-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE SCHEMA building_power;

ALTER SCHEMA building_power OWNER TO root;

CREATE TABLE building_power.building_power (
    sample_time timestamp with time zone NOT NULL,
    electricity_ceiling_fan_energy_consumption double precision,
    electricity_clothes_dryer_energy_consumption double precision,
    electricity_clothes_washer_energy_consumption double precision,
    electricity_cooling_energy_consumption double precision,
    electricity_cooling_fans_pumps_energy_consumption double precision,
    electricity_dishwasher_energy_consumption double precision,
    electricity_freezer_energy_consumption double precision,
    electricity_heating_energy_consumption double precision,
    electricity_heating_fans_pumps_energy_consumption double precision,
    electricity_heating_hp_bkup_energy_consumption double precision,
    electricity_heating_hp_bkup_fa_energy_consumption double precision,
    electricity_hot_water_energy_consumption double precision,
    electricity_lighting_exterior_energy_consumption double precision,
    electricity_lighting_garage_energy_consumption double precision,
    electricity_lighting_interior_energy_consumption double precision,
    electricity_mech_vent_energy_consumption double precision,
    electricity_net_energy_consumption double precision,
    electricity_permanent_spa_heat_energy_consumption double precision,
    electricity_permanent_spa_pump_energy_consumption double precision,
    electricity_plug_loads_energy_consumption double precision,
    electricity_pool_heater_energy_consumption double precision,
    electricity_pool_pump_energy_consumption double precision,
    electricity_pv_energy_consumption double precision,
    electricity_range_oven_energy_consumption double precision,
    electricity_refrigerator_energy_consumption double precision,
    electricity_total_energy_consumption double precision,
    electricity_well_pump_energy_consumption double precision,
    site_energy_net_energy_consumption double precision,
    site_energy_total_energy_consumption double precision,
    load_cooling_energy_delivered_kbtu double precision,
    load_heating_energy_delivered_kbtu double precision,
    load_hot_water_energy_delivered_kbtu double precision,
    outdoor_air_dryblub_temp_c double precision,
    zone_mean_air_temp_air_source_heat_pump_airloop_ret_air_zone_c double precision,
    zone_mean_air_temp_attic_unvented_c double precision,
    zone_mean_air_temp_attic_vented_c double precision,
    zone_mean_air_temp_basement_unconditioned_c double precision,
    zone_mean_air_temp_central_ac_airloop_ret_air_zone_c double precision,
    zone_mean_air_temp_central_ac_and_furnace_airloop_ret_air_zone_ double precision,
    zone_mean_air_temp_conditioned_space_c double precision,
    zone_mean_air_temp_crawlspace_unvented_c double precision,
    zone_mean_air_temp_crawlspace_vented_c double precision,
    zone_mean_air_temp_furnace_airloop_ret_air_zone_c double precision,
    zone_mean_air_temp_garage_c double precision,
    zone_mean_air_temp_ground_source_heat_pump_airloop_ret_air_zone double precision,
    bldg_id bigint NOT NULL,
    source_file text
);

ALTER TABLE building_power.building_power OWNER TO root;

-- Convert to hypertable
SELECT create_hypertable('building_power.building_power', 'sample_time', chunk_time_interval => interval '7 days');

CREATE MATERIALIZED VIEW daily_energy_consumption_by_building
WITH (timescaledb.continuous) AS
SELECT
    bldg_id,
    time_bucket('1 day', recorded_at) AS day,
    sum(electricity_net_energy_consumption) AS electricity_net_energy_consumption,
    sum(electricity_pv_energy_consumption) AS electricity_pv_energy_consumption,
    sum(electricity_total_energy_consumption) AS electricity_total_energy_consumption,
    sum(site_energy_net_energy_consumption) AS site_energy_net_energy_consumption,
    sum(site_energy_total_energy_consumption) AS site_energy_total_energy_consumption
FROM
    energy_data  -- Assumes your source hypertable is named 'energy_data'
GROUP BY
    bldg_id,
    day
WITH NO DATA;

--------------------------------------------------------------------------------
-- 1. SIMULATION METADATA
--------------------------------------------------------------------------------
CREATE TABLE building_power.simulation_outputs (
    simulation_output_id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT now(),
	parameters JSONB NOT NULL,
	pandapower_grid_id BIGINT REFERENCES building_power.pandapower_grids(grid_id) ON DELETE SET NULL
);

CREATE INDEX idx_simulation_outputs_metadata ON building_power.simulation_outputs USING gin (parameters jsonb_path_ops);

--------------------------------------------------------------------------------
-- 2. VARIABLE METADATA
--------------------------------------------------------------------------------
CREATE TABLE building_power.variable (
    variable_id SERIAL PRIMARY KEY,
    simulation_output_id INT NOT NULL REFERENCES building_power.simulation_outputs(simulation_output_id) ON DELETE CASCADE,
    variable_name TEXT NOT NULL,
    unit TEXT,
    extra_info TEXT
);

-- Optional: Index to quickly find variables by name within a simulation
CREATE INDEX ON building_power.variable (simulation_output_id, variable_name);

--------------------------------------------------------------------------------
-- 3. TIME SERIES DATA
--------------------------------------------------------------------------------
CREATE TABLE building_power.output_timeseries (
    variable_id INT NOT NULL REFERENCES building_power.variable(variable_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    quantity DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('building_power.output_timeseries', 'ts', chunk_time_interval => interval '7 days');

-- Index for fast lookup by variable and timestamp
CREATE INDEX ON building_power.output_timeseries (variable_id, ts DESC);

--------------------------------------------------------------------------------
-- 4. COMPRESSION SETTINGS
--------------------------------------------------------------------------------
-- Use TimescaleDB native compression
ALTER TABLE building_power.output_timeseries SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'variable_id'
);

-- Configure automatic compression after data is older than 30 days
SELECT add_compression_policy('building_power.output_timeseries', INTERVAL '30 days');

--------------------------------------------------------------------------------
-- 5. OPTIONAL: Retention Policy
--------------------------------------------------------------------------------
-- Example: keep only last 2 years of raw data
-- SELECT add_retention_policy('timeseries', INTERVAL '2 years');


-- building_power.pandapower_grids definition
DROP TABLE IF EXISTS building_power.pandapower_grids;

CREATE SEQUENCE building_power.grid_name_seq START WITH 1;

CREATE OR REPLACE FUNCTION building_power.generate_sequential_string()
RETURNS TEXT AS $$
DECLARE
    seq_num INT;
BEGIN
    SELECT nextval('grid_name_seq') INTO seq_num;
    RETURN 'GRID_' || LPAD(seq_num::TEXT, 5, '0'); -- Example: "PREFIX_00001"
END;
$$ LANGUAGE plpgsql;

select building_power.generate_sequential_string()

CREATE TABLE building_power.pandapower_grids (
	grid_id bigserial NOT NULL,
	"timestamp" timestamptz DEFAULT now() NULL,
    grid_name text NOT NULL DEFAULT building_power.generate_sequential_string(),   
    grid_description text NULL, 
    created_at timestamptz DEFAULT now() NULL,
	grid_metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
	CONSTRAINT networks_network_name_key UNIQUE (grid_name),	
	CONSTRAINT pandapower_grids_pkey PRIMARY KEY (grid_id)
);

-- building_power.pandapower_grids foreign keys
CREATE INDEX IF NOT EXISTS idx_pandapower_grids_grid_id ON building_power.pandapower_grids(grid_id);
CREATE INDEX idx_grid_metadata ON building_power.pandapower_grids USING gin (grid_metadata jsonb_path_ops);
CREATE INDEX idx_grid_name ON building_power.pandapower_grids USING btree (grid_name);


CREATE TABLE building_power.network_vertices (
    network_vertix_id SERIAL PRIMARY KEY,
    pandapower_grid_id INT NOT NULL REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE,
    pandapower_bus_index INT NOT NULL,
    vertix_label text NULL,
    vertix_metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    UNIQUE (pandapower_grid_id, pandapower_bus_index)
);
CREATE INDEX idx_vertices_metadata ON building_power.network_vertices USING gin (vertix_metadata jsonb_path_ops);
CREATE INDEX idx_vertices_network ON building_power.network_vertices USING btree (network_vertix_id);
-- building_power.network_vertices foreign keys
ALTER TABLE building_power.network_vertices ADD CONSTRAINT network_vertices_network_id_fkey FOREIGN KEY (pandapower_grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- Network edges table with composite foreign keys
CREATE TABLE IF NOT EXISTS building_power.network_edges (
    network_edge_id SERIAL PRIMARY KEY,
    pandapower_grid_id INT NOT NULL REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE,
    source_pandapower_bus_index INT NOT NULL,
    target_pandapower_bus_index INT NOT NULL,
    directed BOOLEAN NOT NULL DEFAULT TRUE,
    edge_metadata JSONB NOT NULL DEFAULT '{}',
    FOREIGN KEY (pandapower_grid_id, source_pandapower_bus_index) REFERENCES building_power.network_vertices(pandapower_grid_id, pandapower_bus_index) ON DELETE CASCADE,
    FOREIGN KEY (pandapower_grid_id, target_pandapower_bus_index) REFERENCES building_power.network_vertices(pandapower_grid_id, pandapower_bus_index) ON DELETE CASCADE
);

-- Indexes for efficient source and target lookups
CREATE INDEX IF NOT EXISTS idx_edges_source ON building_power.network_edges (pandapower_grid_id, source_pandapower_bus_index);
CREATE INDEX IF NOT EXISTS idx_edges_target ON building_power.network_edges (pandapower_grid_id, target_pandapower_bus_index);
CREATE INDEX IF NOT EXISTS idx_edges_network ON building_power.network_edges (pandapower_grid_id);



-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_networks_name ON building_power.networks(network_name);
CREATE INDEX IF NOT EXISTS idx_networks_metadata ON building_power.networks USING GIN (network_metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_vertices_network ON building_power.network_vertices(pandapower_grid_id);
CREATE INDEX IF NOT EXISTS idx_vertices_metadata ON building_power.network_vertices USING GIN (vertix_metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_edges_network ON building_power.network_edges(pandapower_grid_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON building_power.network_edges(pandapower_grid_id, source_pandapower_bus_index);
CREATE INDEX IF NOT EXISTS idx_edges_target ON building_power.network_edges(pandapower_grid_id, target_pandapower_bus_index);
CREATE INDEX IF NOT EXISTS idx_edges_metadata ON building_power.network_edges USING GIN (edge_metadata jsonb_path_ops);
--------------------------------------------------------------------------------
-- DONE
--------------------------------------------------------------------------------


------ Pandapower tables

-- building_power.bus definition

-- Drop table

-- DROP TABLE building_power.bus;

CREATE TABLE building_power.bus (
	grid_id int8 NULL,
	bus_id int8 NULL,
	"name" varchar NULL,
	vn_kv float8 NULL,
	"type" varchar NULL,
	"zone" varchar NULL,
	in_service bool NULL,
	lat float8 NULL,
	lon float8 NULL
);


-- building_power.bus_geodata definition

-- Drop table

-- DROP TABLE building_power.bus_geodata;

CREATE TABLE building_power.bus_geodata (
	grid_id int8 NULL,
	bus_geodata_id int8 NULL,
	x varchar NULL,
	y varchar NULL,
	coords varchar NULL
);


-- building_power.dtypes definition

-- Drop table

-- DROP TABLE building_power.dtypes;

CREATE TABLE building_power.dtypes (
	grid_id int8 NULL,
	dtypes_id int8 NULL,
	"element" varchar NULL,
	"column" varchar NULL,
	dtype varchar NULL
);


-- building_power.ext_grid definition

-- Drop table

-- DROP TABLE building_power.ext_grid;

CREATE TABLE building_power.ext_grid (
	grid_id int8 NULL,
	ext_grid_id int8 NULL,
	"name" varchar NULL,
	bus int8 NULL,
	vm_pu float8 NULL,
	va_degree float8 NULL,
	slack_weight float8 NULL,
	in_service bool NULL,
	s_sc_max_mva float8 NULL,
	s_sc_min_mva float8 NULL,
	rx_min float8 NULL,
	rx_max float8 NULL,
	r0x0_max float8 NULL,
	x0x_max float8 NULL
);


-- building_power.fuse_std_types definition

-- Drop table

-- DROP TABLE building_power.fuse_std_types;

CREATE TABLE building_power.fuse_std_types (
	grid_id int8 NULL,
	fuse_std_types_id varchar NULL,
	fuse_type varchar NULL,
	i_rated_a float8 NULL,
	t_avg varchar NULL,
	t_min varchar NULL,
	t_total varchar NULL,
	x_avg varchar NULL,
	x_min varchar NULL,
	x_total varchar NULL
);


-- building_power.grid_tables definition

-- Drop table

-- DROP TABLE building_power.grid_tables;

CREATE TABLE building_power.grid_tables (
	grid_id int8 NULL,
	grid_tables_id int8 NULL,
	"table" varchar NULL
);


-- building_power.line definition

-- Drop table

-- DROP TABLE building_power.line;

CREATE TABLE building_power.line (
	grid_id int8 NULL,
	line_id int8 NULL,
	"name" varchar NULL,
	std_type varchar NULL,
	from_bus int8 NULL,
	to_bus int8 NULL,
	length_km float8 NULL,
	r_ohm_per_km float8 NULL,
	x_ohm_per_km float8 NULL,
	c_nf_per_km float8 NULL,
	g_us_per_km float8 NULL,
	max_i_ka float8 NULL,
	df float8 NULL,
	"parallel" int8 NULL,
	"type" varchar NULL,
	in_service bool NULL,
	r0_ohm_per_km float8 NULL,
	x0_ohm_per_km float8 NULL,
	c0_nf_per_km float8 NULL
);


-- building_power.line_std_types definition

-- Drop table

-- DROP TABLE building_power.line_std_types;

CREATE TABLE building_power.line_std_types (
	grid_id int8 NULL,
	line_std_types_id varchar NULL,
	c_nf_per_km varchar NULL,
	r_ohm_per_km varchar NULL,
	x_ohm_per_km varchar NULL,
	max_i_ka varchar NULL,
	"type" varchar NULL,
	q_mm2 varchar NULL,
	alpha varchar NULL,
	voltage_rating varchar NULL
);


-- building_power.parameters definition

-- Drop table

-- DROP TABLE building_power.parameters;

CREATE TABLE building_power.parameters (
	grid_id int8 NULL,
	parameters_id int8 NULL,
	"version" varchar NULL,
	format_version varchar NULL,
	converged bool NULL,
	"OPF_converged" bool NULL,
	"name" varchar NULL,
	f_hz float8 NULL,
	sn_mva int8 NULL
);


-- building_power.trafo definition

-- Drop table

-- DROP TABLE building_power.trafo;

CREATE TABLE building_power.trafo (
	grid_id int8 NULL,
	trafo_id int8 NULL,
	"name" varchar NULL,
	std_type varchar NULL,
	hv_bus int8 NULL,
	lv_bus int8 NULL,
	sn_mva float8 NULL,
	vn_hv_kv float8 NULL,
	vn_lv_kv float8 NULL,
	vk_percent float8 NULL,
	vkr_percent float8 NULL,
	pfe_kw float8 NULL,
	i0_percent float8 NULL,
	shift_degree float8 NULL,
	tap_side varchar NULL,
	tap_neutral float8 NULL,
	tap_min float8 NULL,
	tap_max float8 NULL,
	tap_step_percent float8 NULL,
	tap_step_degree float8 NULL,
	tap_pos float8 NULL,
	tap_phase_shifter bool NULL,
	"parallel" int8 NULL,
	df float8 NULL,
	in_service bool NULL,
	vector_group varchar NULL,
	vk0_percent float8 NULL,
	mag0_percent int8 NULL,
	mag0_rx int8 NULL,
	si0_hv_partial float8 NULL,
	vkr0_percent float8 NULL
);


-- building_power.trafo3w_std_types definition

-- Drop table

-- DROP TABLE building_power.trafo3w_std_types;

CREATE TABLE building_power.trafo3w_std_types (
	grid_id int8 NULL,
	trafo3w_std_types_id varchar NULL,
	sn_hv_mva varchar NULL,
	sn_mv_mva varchar NULL,
	sn_lv_mva varchar NULL,
	vn_hv_kv varchar NULL,
	vn_mv_kv varchar NULL,
	vn_lv_kv varchar NULL,
	vk_hv_percent varchar NULL,
	vk_mv_percent varchar NULL,
	vk_lv_percent varchar NULL,
	vkr_hv_percent varchar NULL,
	vkr_mv_percent varchar NULL,
	vkr_lv_percent varchar NULL,
	pfe_kw varchar NULL,
	i0_percent varchar NULL,
	shift_mv_degree varchar NULL,
	shift_lv_degree varchar NULL,
	vector_group varchar NULL,
	tap_side varchar NULL,
	tap_neutral varchar NULL,
	tap_min varchar NULL,
	tap_max varchar NULL,
	tap_step_percent varchar NULL
);


-- building_power.trafo_std_types definition

-- Drop table

-- DROP TABLE building_power.trafo_std_types;

CREATE TABLE building_power.trafo_std_types (
	grid_id int8 NULL,
	trafo_std_types_id varchar NULL,
	i0_percent varchar NULL,
	pfe_kw varchar NULL,
	vkr_percent varchar NULL,
	sn_mva varchar NULL,
	vn_lv_kv varchar NULL,
	vn_hv_kv varchar NULL,
	vk_percent varchar NULL,
	shift_degree varchar NULL,
	vector_group varchar NULL,
	tap_side varchar NULL,
	tap_neutral varchar NULL,
	tap_min varchar NULL,
	tap_max varchar NULL,
	tap_step_degree varchar NULL,
	tap_step_percent varchar NULL,
	tap_phase_shifter varchar NULL
);


-- building_power.bus foreign keys

ALTER TABLE building_power.bus ADD CONSTRAINT bus_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.bus_geodata foreign keys

ALTER TABLE building_power.bus_geodata ADD CONSTRAINT bus_geodata_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.dtypes foreign keys

ALTER TABLE building_power.dtypes ADD CONSTRAINT dtypes_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.ext_grid foreign keys

ALTER TABLE building_power.ext_grid ADD CONSTRAINT ext_grid_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.fuse_std_types foreign keys

ALTER TABLE building_power.fuse_std_types ADD CONSTRAINT fuse_std_types_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.grid_tables foreign keys

ALTER TABLE building_power.grid_tables ADD CONSTRAINT grid_tables_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.line foreign keys

ALTER TABLE building_power.line ADD CONSTRAINT line_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.line_std_types foreign keys

ALTER TABLE building_power.line_std_types ADD CONSTRAINT line_std_types_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.parameters foreign keys

ALTER TABLE building_power.parameters ADD CONSTRAINT parameters_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.trafo foreign keys

ALTER TABLE building_power.trafo ADD CONSTRAINT trafo_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.trafo3w_std_types foreign keys

ALTER TABLE building_power.trafo3w_std_types ADD CONSTRAINT trafo3w_std_types_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;


-- building_power.trafo_std_types foreign keys

ALTER TABLE building_power.trafo_std_types ADD CONSTRAINT trafo_std_types_grid_id_fkey FOREIGN KEY (grid_id) REFERENCES building_power.pandapower_grids(grid_id) ON DELETE CASCADE;