# Duilio 3-Phase Power Network Simulation Parameters Documentation

This document describes the parameters used in the `duilio_3ph_evaluate.py` simulation script for evaluating 3-phase electrical distribution networks with solar PV systems, energy storage, and building loads.

## Parameter File Structure

Parameters are loaded from a JSON configuration file (e.g., `first_test.json`) containing the following fields:

### Basic Simulation Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Unique identifier for the simulation test | `"first_test"` |
| `description` | string | Human-readable description of the test configuration | `"This is the first test configuration for the pandapower simulation."` |
| `output_file` | string | Name of the CSV output file for results | `"first_test.csv"` |

### Time Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `step_size_s` | integer | Simulation time step in seconds | `900` (15 minutes) |
| `start_time` | string | Simulation start timestamp (YYYY-MM-DD HH:MM:SS) | `"2018-01-01 07:00:00"` |
| `simulation_time_s` | integer | Total simulation duration in seconds | `28800` (8 hours) |

### Solar PV Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `panel_efficiency` | float | Solar panel efficiency (0.0 to 1.0) | `0.5` (50% efficiency) |

### Network Topology Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `network_id` | string | Identifier for the network configuration | `"first_test_network"` |
| `require_existing_assets` | boolean | If true, fail if network/profile not in database (use with pre-generated assets) | `true` |
| `use_saved_network_if_exists` | boolean | **(Deprecated)** Legacy flag, use `require_existing_assets` instead | `true` |
| `mv_bus_latitude` | float | Latitude coordinate of the medium voltage bus | `-25.4505` |
| `mv_bus_longitude` | float | Longitude coordinate of the medium voltage bus | `-49.2310` |

### Network Structure Ranges

These parameters define the random generation bounds for the electrical network topology:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `commercial_range` | array[int, int] | [min, max] number of commercial transformer branches | `[2, 5]` |
| `industrial_range` | array[int, int] | [min, max] number of industrial transformer branches | `[2, 5]` |
| `residential_range` | array[int, int] | [min, max] number of residential transformer branches | `[2, 5]` |
| `fork_length_range` | array[int, int] | [min, max] length of distribution line forks in meters | `[200, 800]` |
| `line_buses_range` | array[int, int] | [min, max] number of buses per distribution line | `[5, 15]` |
| `line_forks_range` | array[int, int] | [min, max] number of forks per distribution line | `[1, 4]` |

### Power System Sizing

These parameters control the random generation of electrical loads and generation:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `solar_power_range_per_branch_kW` | array[float, float] | [min, max] total solar PV power per branch in kW | `[0.5, 100]` |
| `solar_power_steps` | integer | Number of discrete steps for solar power distribution | `10` |
| `load_power_range_per_branch_kW` | array[float, float] | [min, max] total electrical load per branch in kW | `[0.5, 100]` |
| `storage_capacity_range_per_branch_kWh` | array[float, float] | [min, max] total battery storage capacity per branch in kWh | `[0.5, 100]` |
| `fixed_load_profile_id` | string (optional) | Source profile ID used to preserve the same load/PV assignment across multiple storage runs | `"Solar-20-Seed-1111"` |
| `preserve_load_when_changing_storage` | boolean (optional) | When true and `fixed_load_profile_id` is set, the simulation reuses the existing load/PV assignment and only updates storage capacities from the specified range. | `true` |
| `initial_capacity_range` | array[float, float] | [min, max] initial battery state of charge in percent | `[0.0, 100.0]` |

### Phase Balancing

| Parameter | Type | Description | Example |
|-----------|------|-------------|--------|
| `balance_phase_loading` | boolean | When `true`, a postprocessing step runs after power profile generation to rebalance phase assignments for 1ph and 2ph consumers, minimising the max-min power imbalance across phases a, b, c within each transformer branch. 3ph consumers are not modified. Default `false`. | `true` |

#### How Phase Balancing Works

During power profile generation, each consumer is randomly assigned to one or more grid phases (a, b, c) depending on its `InstallationType` (1ph, 2ph, or 3ph). This random assignment can lead to significant phase imbalance — for example, most single-phase loads might end up on phase *a*, overloading it while phases *b* and *c* remain underutilised.

When `balance_phase_loading` is enabled, a greedy optimisation runs **after** the initial random assignment:

1. For each transformer, all LV-side consumers are collected.
2. 3ph consumers (inherently balanced) are fixed in place.
3. 1ph and 2ph consumers are sorted by descending load power.
4. Each consumer is reassigned to the phase combination that minimises the resulting max-min imbalance across the three phases.
5. Both `installation_connection_phases` and `inverter_connection_phases` are updated accordingly.

This produces a more realistic and stable network configuration, reducing the likelihood of convergence errors caused by extreme phase imbalance.

## Network Generation Behavior

### Recommended Workflow: Pre-generate Assets

For reproducible simulations with consistent network configurations, use the `RegenerateAssets` utility to pre-generate networks and power profiles **before** running simulations:

```bash
# Pre-generate networks and power profiles
python -m simulation_worker.RegenerateAssets --network --profile --folder ./params

# Then run simulations with require_existing_assets=true
python -m simulation_worker.SimulationWorker --workers 4
```

Set `require_existing_assets: true` in your parameter files to ensure simulations fail fast if assets are missing.

### Asset Loading Behavior

The `require_existing_assets` parameter controls how the simulation handles missing assets:

| Value | Behavior |
|-------|----------|
| `true` | **Strict mode.** Fail with error if network or power profile doesn't exist in database. Recommended when using pre-generated assets. |
| `false` (default) | **Auto-generate mode.** Try to load from database; if not found, generate new assets and save them. Provides backward compatibility. |

### Legacy Parameters (Deprecated)

The following parameters are deprecated but still supported for backward compatibility:

- `use_saved_network_if_exists` - Ignored when `require_existing_assets` is set
- `use_saved_power_profile_if_exists` - Ignored when `require_existing_assets` is set

### RegenerateAssets Utility

Use this utility to manually regenerate networks and/or power profiles:

```bash
# Regenerate both networks and power profiles
python -m simulation_worker.RegenerateAssets --network --profile

# Regenerate only networks  
python -m simulation_worker.RegenerateAssets --network

# Regenerate only power profiles (networks must exist)
python -m simulation_worker.RegenerateAssets --profile

# Specify custom folder
python -m simulation_worker.RegenerateAssets --folder ./my_params --network

# Parallel processing
python -m simulation_worker.RegenerateAssets --network --profile --workers 4

# Dry run to preview files
python -m simulation_worker.RegenerateAssets --network --profile --dry-run
```

## Network Generation Process

The simulation creates a realistic distribution network using these parameters:

1. **Network Loading/Generation Decision**: Based on `use_saved_network_if_exists` parameter, either loads existing network or generates new one.

2. **Transformer Creation**: Creates commercial, industrial, and residential transformer connections from the MV bus based on the respective range parameters.

3. **Line Generation**: For each transformer branch, generates distribution lines with buses and forks according to the line configuration ranges.

4. **Geographic Placement**: Places network components geographically using the MV bus coordinates as reference, with automatic collision avoidance.

5. **Load Assignment**: 
   - Distributes total branch load power among randomly selected buses
   - Queries a PostgreSQL database for real building power consumption profiles
   - Matches building profiles to target power levels within 0.5 kW tolerance

6. **Solar PV Assignment**:
   - Distributes solar generation among buses based on solar power ranges
   - Creates 1-phase, 2-phase, or 3-phase inverters based on power rating
   - Connects solar irradiation models based on geographic coordinates

7. **Storage Assignment**:
   - Sizes battery storage based on Tesla Powerwall 3 characteristics
   - Sets charge/discharge limits proportional to capacity
   - Distributes initial state of charge randomly within specified range

## Database Requirements

The simulation requires a PostgreSQL database with:
- Building power consumption data in `building_power.building_power` table
- TimescaleDB extension for time-series data handling
- Network topology storage capabilities in `network_vertices` and `network_edges` tables
- Pandapower network storage capabilities

## Output Data

The simulation generates time-series data for:
- Bus voltages and power flows (3-phase)
- Line currents and losses (3-phase) 
- Transformer loading
- Solar irradiation and PV generation
- Building load consumption
- Battery storage operation

This documentation provides guidance for configuring realistic distribution network simulations with renewable energy integration and real building load profiles.
