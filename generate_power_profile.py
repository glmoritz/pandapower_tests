import pandapower as pp
import pandas as pd
from enum import Enum
import math
from shapely.geometry import Point as ShapelyPoint, LineString, MultiLineString, MultiPoint as ShapelyMultipoint
from shapely.strtree import STRtree
from shapely import affinity
from shapely import distance as shapely_distance
from geopy.distance import distance as GeopyDistance
from geopy.point import Point as GeopyPoint
from collections import deque
import networkx as nx
import numpy as np
import psycopg2
import json
from shapely import wkt
from pandapower.sql_io import to_sql, from_sql, check_postgresql_catalogue_table
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text


def distribute_loads_to_buses(net, graph, params, db, rng=None):
    """
    Distribute loads, PV systems, and storage to buses in the network.
    
    Args:
        net: Pandapower network
        graph: NetworkX graph representation
        params: Simulation parameters dictionary
        db: Database connection parameters
        rng: numpy.random.Generator instance for reproducible randomness.
             If None, creates a new unseeded generator (not recommended).
    """
    if rng is None:
        # Fallback for backward compatibility, but not recommended
        rng = np.random.default_rng()
        print("WARNING: distribute_loads_to_buses called without RNG - results will not be reproducible!")
    
    # Iterate through each transformer in the network
    for idx, trafo_row in net.trafo.iterrows():
        
        # Get all the lv buses that are fed by this transformer
        lv_buses = list(nx.descendants(graph, trafo_row['lv_bus']))

        # Draw the total power of the loads on this branch
        total_load_power = rng.uniform(params['load_power_range_per_branch_kW'][0], 
                                      params['load_power_range_per_branch_kW'][1])
        
        # Draw the total solar power of the PV systems on this branch
        total_solar_power = rng.uniform(params['solar_power_range_per_branch_kW'][0], 
                                      params['solar_power_range_per_branch_kW'][1])
        
        total_storage_capacity = rng.uniform(params['storage_capacity_range_per_branch_kWh'][0], 
                                      params['storage_capacity_range_per_branch_kWh'][1])
        
        # Select random number of buses and which buses get loads
        num_load_buses = rng.integers(1, len(lv_buses) + 1)  # +1 because high is exclusive
        load_buses = rng.choice(lv_buses, size=num_load_buses, replace=False).tolist()

        #distribute the load power to the buses
        load_distribution = rng.uniform(size=len(load_buses))  
        bus_loads = (total_load_power/load_distribution.sum())*load_distribution

        #distribute the PV generation to the buses
        pv_distribution = rng.uniform(size=len(load_buses)) 
        storage_distribution = rng.uniform(size=len(load_buses))

        pv_peak_power = (pv_distribution/pv_distribution.sum())*total_solar_power
        pv_storage_capacity = (storage_distribution/storage_distribution.sum())*total_storage_capacity        
        
        #connect the PV systems to the buses
        for i, bus in enumerate(load_buses):           
            initial_charge = rng.uniform(params['initial_capacity_range'][0], params['initial_capacity_range'][1])
        
            #the maximum charge and discharge power of the storage system will be based on Tesla Powerwall 3
            MaxChargePower_kW = (8.0/13.5)*pv_storage_capacity[i]  # 8 kW charge power, 13.5 kWh capacity
            MaxDischargePower_kW = (11.5/13.5)*pv_storage_capacity[i]  # 11.5 kW rated power, 13.5 kWh capacity

            # ---- Determine installation type based on load power ----
            # The installation type reflects the house wiring (how many grid phases it is connected to).
            if bus_loads[i] < 3.0:
                installation_type = '1ph'
            elif bus_loads[i] < 8.0:
                installation_type = rng.choice(['1ph', '2ph'])
            else:
                installation_type = rng.choice(['2ph', '3ph'])

            # ---- Determine inverter type based on PV peak power, constrained by installation ----
            # The inverter cannot have more phases than the installation.
            _phase_count = {'1ph': 1, '2ph': 2, '3ph': 3}
            if pv_peak_power[i] < 10.0:
                inverter_type = '1ph'
            elif pv_peak_power[i] < 20.0:
                inverter_type = rng.choice(['1ph', '2ph'])
            else:
                inverter_type = '3ph'

            # Downgrade inverter type if it exceeds the installation
            if _phase_count[inverter_type] > _phase_count[installation_type]:
                inverter_type = installation_type

            # ---- Determine which physical phases the installation connects to ----
            n_inst = _phase_count[installation_type]
            installation_connection_phases = rng.choice(['a', 'b', 'c'], size=n_inst, replace=False).tolist()

            # ---- Determine which installation phases the inverter connects to ----
            n_inv = _phase_count[inverter_type]
            if n_inv == n_inst:
                bus_phases = installation_connection_phases[:]
            else:
                bus_phases = rng.choice(installation_connection_phases, size=n_inv, replace=False).tolist()

            #TODO: Verificar a Norma copel NTC 901100 para definir os limites dos disjuntores residenciais e comerciais,
            #  e se possível coletar dados reais de disjuntores usados em residências brasileiras para definir uma distribuição estatística mais realista.
            #  Por enquanto, estou usando uma distribuição uniforme com um valor padrão baseado em disjuntores típicos, mas isso pode ser melhorado com dados reais.

            # ---- Per-phase breaker limit (kW -> MW) ----
            # Typical residential breakers: 1ph ~32A@230V=7.36kW, 3ph ~25A@230V=5.75kW per phase.
            # The parameter 'breaker_limit_per_phase_kW' can be a scalar or a [min, max] range
            # drawn from the scenario config JSON.  If absent, a default value is derived from
            # the installation type: roughly 20% headroom above the expected per-phase load.
            breaker_cfg = params.get('breaker_limit_per_phase_kW', None)
            if breaker_cfg is None:
                # Derive a sensible default -- assume breaker rated ~20% above average per-phase load
                breaker_kw = bus_loads[i] / _phase_count[installation_type] * 1.2
            elif isinstance(breaker_cfg, list):
                breaker_kw = rng.uniform(breaker_cfg[0], breaker_cfg[1])
            else:
                breaker_kw = float(breaker_cfg)

            household_params = {
                "LoadPower_kW": bus_loads[i],
                "SolarPeakPower_MW": pv_peak_power[i]/1000,
                "StorageCapacity_MWh": pv_storage_capacity[i]/1000,
                "InitialSOC_percent": initial_charge,
                "MaxChargePower_MW": MaxChargePower_kW/1000,
                "MaxDischargePower_MW": MaxDischargePower_kW/1000,
                "InverterType": inverter_type,
                "InstallationType": installation_type,
                "BreakerLimit_MW": breaker_kw / 1000.0,
                "Index": f'bus{bus}'
            }
            graph.nodes[bus]['household_params'] = household_params
            graph.nodes[bus]['installation_connection_phases'] = installation_connection_phases
            graph.nodes[bus]['inverter_connection_phases'] = bus_phases

            #first I need to find a combination of buses that have this average power
            if bus_loads[i] > 4.0: #our database has no loads with average power above 4kW
                min_value_kw = 1.0
                max_value_kw = 4.0
                min_parts_count = int(bus_loads[i] / max_value_kw) + 1                
                max_parts_count = int(bus_loads[i] / min_value_kw) 
                parts_count = rng.integers(min_parts_count, max_parts_count + 1)  # +1 because high is exclusive
                split_loads = [min_value_kw] * parts_count
                
                remainder = bus_loads[i] - sum(split_loads)
                while remainder > 0:
                    remainder_distribution_factors = rng.uniform(size=len(split_loads))
                    remainder_distribution = (remainder/np.sum(remainder_distribution_factors))*remainder_distribution_factors
                    for j in range(len(split_loads)):
                        if split_loads[j] + remainder_distribution[j] <= max_value_kw*1.1:
                            split_loads[j] += remainder_distribution[j]
                            remainder -= remainder_distribution[j]                
            else:
                split_loads = [bus_loads[i]]

            household_params['SplitLoads_target_kW'] = split_loads

            #now execute a query to find buildings that have split_loads[i] as average power consumption
            sql = text("""
                    WITH targets(avg_power) AS 
                    (
                        SELECT unnest(:split_loads)
                    ), 
                    candidate_buildings AS 
                    (
                    SELECT 
                        bldg_id,
                        -- Convert kWh to kW by dividing by time fraction (0.25 for 15min intervals)
                        -- Then average these power values over the period
                        AVG(electricity_total_energy_consumption / 0.25) AS avg_power_kw
                    FROM building_power.building_power
                    WHERE electricity_total_energy_consumption IS NOT NULL
                    AND sample_time >= :start_time
                    AND sample_time <= :end_time
                    GROUP BY bldg_id
                    ) 
                    SELECT DISTINCT ON (t.avg_power) 
                        cb.bldg_id,
                        t.avg_power, 
                        cb.avg_power_kw, 
                        ABS(cb.avg_power_kw - t.avg_power) AS deviation_kw 
                        FROM targets t 
                        JOIN candidate_buildings cb 
                        ON ABS(cb.avg_power_kw - t.avg_power) <= :tolerance
                    ORDER BY t.avg_power, random(), deviation_kw;
                  """)            

            sim_start_dt = datetime.strptime(params['start_time'], '%Y-%m-%d %H:%M:%S')
            sim_end_dt = sim_start_dt + timedelta(seconds=float(params['simulation_time_s']))

            db_url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
            engine = create_engine(db_url)
            with engine.connect() as conn:
                results = conn.execute(sql, {
                    "split_loads": split_loads,
                    "start_time": sim_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": sim_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    "tolerance": 0.5
                }).fetchall()

            bldg_ids_list = [str(row[0]) for row in results]
            graph.nodes[bus]['connected_buildings_ids'] = [row[0] for row in results]
            graph.nodes[bus]['household_params']['AssignedLoadPower_kW'] = [row[2] for row in results]  # actual assigned average powers
            graph.nodes[bus]['household_params']['SplitLoads_target_kW'] = split_loads
            

    # --- Phase balancing postprocessing ---
    if params.get('balance_phase_loading', False):
        print("[INFO] Running phase balancing postprocessing...")
        balance_phase_loading(net, graph)


def balance_phase_loading(net, graph):
    """
    Postprocessing step that rebalances phase assignments for 1ph and 2ph consumers
    to minimize power imbalance across the three phases (a, b, c) within each
    transformer branch.

    Only consumers whose InstallationType is '1ph' or '2ph' are eligible for
    reassignment.  3ph consumers are left untouched because they are inherently
    balanced.

    Algorithm (per transformer):
        1. Compute the total load power contributed to each phase by all
           consumers on the LV side.
        2. Sort consumers by descending load power (largest first — greedy).
        3. For each consumer, evaluate every valid phase assignment and pick
           the one that minimises the resulting max-min phase imbalance.
        4. Update ``installation_connection_phases`` and
           ``inverter_connection_phases`` in the graph node accordingly.

    Args:
        net: pandapower network (for transformer table).
        graph: NetworkX DiGraph with household_params / phase data on nodes.

    Returns:
        dict: Per-transformer summary with phase totals before and after
              balancing and the number of consumers reassigned.
    """
    _phase_count = {'1ph': 1, '2ph': 2, '3ph': 3}
    all_phases = ['a', 'b', 'c']
    summary = {}

    for idx, trafo_row in net.trafo.iterrows():
        lv_buses = list(nx.descendants(graph, trafo_row['lv_bus']))

        # Collect buses with household params
        consumer_buses = []
        for bus in lv_buses:
            attrs = graph.nodes.get(bus, {})
            if 'household_params' not in attrs:
                continue
            hp = attrs['household_params']
            inst_type = hp.get('InstallationType', '3ph')
            load_kw = hp.get('LoadPower_kW', 0.0)
            consumer_buses.append({
                'bus': bus,
                'installation_type': inst_type,
                'load_kw': load_kw,
                'n_inst': _phase_count.get(inst_type, 3),
            })

        if not consumer_buses:
            continue

        # --- Compute initial phase totals ---
        phase_totals_before = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        for cb in consumer_buses:
            inst_phases = graph.nodes[cb['bus']].get('installation_connection_phases', all_phases[:])
            per_phase_kw = cb['load_kw'] / max(len(inst_phases), 1)
            for ph in inst_phases:
                phase_totals_before[ph] += per_phase_kw

        # --- Greedy reassignment ---
        # Reset phase totals — we will rebuild from scratch
        phase_totals = {'a': 0.0, 'b': 0.0, 'c': 0.0}

        # Separate fixed (3ph) and movable (1ph/2ph) consumers
        fixed_consumers = [c for c in consumer_buses if c['installation_type'] == '3ph']
        movable_consumers = [c for c in consumer_buses if c['installation_type'] in ('1ph', '2ph')]

        # Add fixed consumers first
        for cb in fixed_consumers:
            inst_phases = graph.nodes[cb['bus']].get('installation_connection_phases', all_phases[:])
            per_phase_kw = cb['load_kw'] / max(len(inst_phases), 1)
            for ph in inst_phases:
                phase_totals[ph] += per_phase_kw

        # Sort movable consumers by load power descending (greedy: place largest first)
        movable_consumers.sort(key=lambda c: c['load_kw'], reverse=True)

        reassigned_count = 0
        from itertools import combinations
        for cb in movable_consumers:
            n_phases = cb['n_inst']

            # Generate all possible phase assignments for this consumer
            if n_phases == 1:
                candidates = [['a'], ['b'], ['c']]
            elif n_phases == 2:
                candidates = [list(combo) for combo in combinations(all_phases, 2)]
            else:
                candidates = [all_phases[:]]

            best_assignment = None
            best_imbalance = float('inf')

            for candidate in candidates:
                # Simulate adding this consumer with this assignment
                trial = dict(phase_totals)
                per_phase_kw = cb['load_kw'] / n_phases
                for ph in candidate:
                    trial[ph] = trial.get(ph, 0.0) + per_phase_kw
                imbalance = max(trial.values()) - min(trial.values())
                if imbalance < best_imbalance:
                    best_imbalance = imbalance
                    best_assignment = candidate

            # Apply best assignment
            per_phase_kw = cb['load_kw'] / n_phases
            for ph in best_assignment:
                phase_totals[ph] += per_phase_kw

            # Check if the phases actually changed
            old_inst_phases = graph.nodes[cb['bus']].get('installation_connection_phases', [])
            if sorted(best_assignment) != sorted(old_inst_phases):
                reassigned_count += 1

            # Update graph node
            graph.nodes[cb['bus']]['installation_connection_phases'] = best_assignment

            # Update inverter connection phases: inverter phases are a subset of
            # installation phases.  Keep the same count but pick from the new
            # installation phases.
            hp = graph.nodes[cb['bus']]['household_params']
            inv_type = hp.get('InverterType', '1ph')
            n_inv = _phase_count.get(inv_type, 1)
            if n_inv >= n_phases:
                new_inv_phases = best_assignment[:]
            else:
                # Pick the first n_inv phases from the new installation phases
                new_inv_phases = best_assignment[:n_inv]
            graph.nodes[cb['bus']]['inverter_connection_phases'] = new_inv_phases

        # --- Summary ---
        imbalance_before = max(phase_totals_before.values()) - min(phase_totals_before.values())
        imbalance_after = max(phase_totals.values()) - min(phase_totals.values())
        summary[idx] = {
            'phase_totals_before_kW': dict(phase_totals_before),
            'phase_totals_after_kW': dict(phase_totals),
            'imbalance_before_kW': round(imbalance_before, 4),
            'imbalance_after_kW': round(imbalance_after, 4),
            'consumers_total': len(consumer_buses),
            'consumers_movable': len(movable_consumers),
            'consumers_reassigned': reassigned_count,
        }

        print(
            f"  Trafo {idx}: imbalance {imbalance_before:.2f} kW -> {imbalance_after:.2f} kW "
            f"({reassigned_count}/{len(movable_consumers)} consumers reassigned)"
        )

    return summary


def apply_profile_to_graph(engine, grid_id: int, profile_name: str, graph: nx.Graph):      
    """
    Load power profile metadata from database and apply to graph nodes.
    All node properties are stored as a single JSON object.
    """
    def coerce_numeric_strings(obj):
        if isinstance(obj, dict):
            return {k: coerce_numeric_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [coerce_numeric_strings(item) for item in obj]
        if isinstance(obj, str):
            value = obj.strip()
            if not value:
                return obj
            lower_value = value.lower()
            if lower_value in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                return obj
            try:
                if value.isdigit() or (
                    len(value) > 1
                    and value[0] in {"+", "-"}
                    and value[1:].isdigit()
                ):
                    return int(value)
                return float(value)
            except ValueError:
                return obj
        return obj

    # Custom decoder for complex objects
    def decode_custom_objects(obj):
        if isinstance(obj, dict):
            # Handle shapely geometry objects
            if "_shapely_wkt" in obj:
                return wkt.loads(obj["_shapely_wkt"])
            
            # Handle GeopyPoint objects
            elif "_geopy_point" in obj:
                point_data = obj["_geopy_point"]
                latitude = coerce_numeric_strings(point_data["latitude"])
                longitude = coerce_numeric_strings(point_data["longitude"])
                return GeopyPoint(latitude, longitude)
            
            # Process nested dictionaries
            return {k: decode_custom_objects(v) for k, v in obj.items()}
        
        # Handle lists
        elif isinstance(obj, list):
            return [decode_custom_objects(item) for item in obj]
        
        # All other objects pass through unchanged
        return obj
    
    sql = text("""
        SELECT 
               pandapower_bus_index, 
               node_data
        FROM building_power.bus_power_profile_assignments
        WHERE pandapower_grid_id = :grid_id
          AND power_profile_id = (
              SELECT power_profile_id 
              FROM building_power.power_profiles 
              WHERE power_profile_name = :profile_name
                AND pandapower_grid_id = :grid_id
          )
        ORDER BY pandapower_bus_index;
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"grid_id": grid_id, "profile_name": profile_name}).fetchall()
        if not rows:
            return False
        
        for row in rows:
            bus_idx = row[0]
            node_data_json = row[1]  # JSON string from database
            
            if bus_idx not in graph.nodes:
                continue
            
            # Parse JSON and restore all node properties, decoding custom objects
            if isinstance(node_data_json, str):
                node_data = json.loads(node_data_json)
            else:
                node_data = node_data_json
            decoded_data = decode_custom_objects(node_data)
            decoded_data = coerce_numeric_strings(decoded_data)
            if not isinstance(decoded_data, dict):
                continue
            for key, value in decoded_data.items():
                graph.nodes[bus_idx][key] = value
        
        return True


def save_graph_metadata(engine, grid_id: int, profile_name: str, graph: nx.Graph):    
    """
    Traverses the NetworkX graph and saves all node properties as JSON.
    Only saves nodes that have 'household_params' attribute.
    """
    # Custom JSON encoder for complex objects
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if hasattr(o, '__geo_interface__'):
                return {"_shapely_wkt": o.wkt}
            if isinstance(o, GeopyPoint):
                return {"_geopy_point": {"latitude": o.latitude, "longitude": o.longitude}}
            return json.JSONEncoder.default(self, o)

    with engine.begin() as conn:
        # ---------------------------------------------------------
        # 1. Check if profile exists
        # ---------------------------------------------------------
        check_sql = text("""
            SELECT 1 from            
            building_power.power_profiles AS pp            
            WHERE pp.pandapower_grid_id = :grid_id
            AND pp.power_profile_name = :profile_name
            LIMIT 1;
        """)

        exists = conn.execute(check_sql, {
            "grid_id": grid_id,
            "profile_name": profile_name
        }).fetchone()

        # ---------------------------------------------------------
        # 2. If exists, rename old profile
        # ---------------------------------------------------------
        if exists:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            legacy_name = f"{profile_name}_legacy_{ts}"

            rename_sql = text("""
                UPDATE building_power.power_profiles
                SET power_profile_name = :legacy_name
                WHERE pandapower_grid_id = :grid_id
                AND power_profile_name = :profile_name;
            """)

            conn.execute(rename_sql, {
                "legacy_name": legacy_name,
                "grid_id": grid_id,
                "profile_name": profile_name
            })

            print(f"[INFO] Profile existed. Renamed old profile to {legacy_name}")

        # ---------------------------------------------------------
        # 3. Insert power profile and collect node data
        # ---------------------------------------------------------
        # Insert or get the power profile
        profile_sql = text("""
            INSERT INTO building_power.power_profiles (power_profile_name, pandapower_grid_id)
            VALUES (:profile_name, :grid_id)
            ON CONFLICT (pandapower_grid_id, power_profile_name) DO NOTHING
            RETURNING power_profile_id;
        """)

        profile_result = conn.execute(profile_sql, {
            "grid_id": grid_id,
            "profile_name": profile_name
        }).fetchone()
        
        if profile_result:
            power_profile_id = profile_result[0]
        else:
            # If ON CONFLICT DO NOTHING, fetch the existing id
            get_profile_sql = text("""
            SELECT power_profile_id 
            FROM building_power.power_profiles 
            WHERE pandapower_grid_id = :grid_id
            AND power_profile_name = :profile_name;
            """)
            power_profile_id = conn.execute(get_profile_sql, {
            "grid_id": grid_id,
            "profile_name": profile_name
            }).fetchone()[0]

        insert_sql = text("""
            INSERT INTO building_power.bus_power_profile_assignments (
            pandapower_grid_id,
            pandapower_bus_index,
            power_profile_id,
            node_data
            ) VALUES (
            :grid_id,
            :bus_index,
            :power_profile_id,
            CAST(:node_data AS jsonb)
            )
            ON CONFLICT (pandapower_grid_id, pandapower_bus_index, power_profile_id)
            DO UPDATE SET node_data = CAST(:node_data AS jsonb);
        """)

        saved_count = 0
        for bus_id, attrs in graph.nodes(data=True):
            # Convert all node attributes to JSON string
            node_data_json = json.dumps(dict(attrs), cls=CustomJSONEncoder)
            
            conn.execute(insert_sql, {
            "grid_id": grid_id,
            "bus_index": int(bus_id),
            "power_profile_id": power_profile_id,
            "node_data": node_data_json
            })
            saved_count += 1
        
        print(f"[INFO] Created {saved_count} metadata rows for profile '{profile_name}'.")




def save_power_profile_to_database(graph: nx.Graph, net, db_connection, grid_name):
    conn = psycopg2.connect(**db_connection)
    cur = conn.cursor()
    grid_catalogue_name = 'pandapower_grids'
    
    # Create a custom JSON encoder for complex objects
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, o):
            # Handle shapely geometry objects
            if hasattr(o, '__geo_interface__'):
                return {"_shapely_wkt": o.wkt}
            
            # Handle GeopyPoint objects
            if isinstance(o, GeopyPoint):
                return {"_geopy_point": {"latitude": o.latitude, "longitude": o.longitude}}
                
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, o)
    
    try:
        
        # Step 1: Create or get network record
        cur.execute(f"""
            SELECT grid_id from building_power.{grid_catalogue_name} WHERE grid_name = %s            
        """,
          (
            grid_name, 
            )
        )

        if cur is not None:        
            old_grid_id = cur.fetchone()        
            if old_grid_id is not None:
                # Step 2: Instead of deleting, we'll just update the grid name to maintain history
                # This updates the existing grid to have a unique historical name
                historical_grid_name = f"{grid_name}_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
                
                cur.execute(f"""
                    UPDATE building_power.{grid_catalogue_name}
                    SET grid_name = %s
                    WHERE grid_id = %s
                """, (historical_grid_name, old_grid_id))
                
                # Commit the name update so we can create a new grid with the original name
                conn.commit()                
        
        pandapower_grid_id = to_sql(net=net,
                         conn=conn,
                         schema="building_power", 
                         grid_catalogue_name=grid_catalogue_name
                        )      
        
        # Step 3: Insert vertices using pandapower bus index as the key
        for node, attrs in graph.nodes(data=True):
            label = attrs.get("name", f"Bus_{node}")
            metadata = dict(attrs)
            
            # Serialize complex objects in metadata
            metadata_json = json.dumps(metadata, cls=CustomJSONEncoder)
            
            cur.execute("""
                INSERT INTO building_power.network_vertices (pandapower_grid_id, pandapower_bus_index, vertix_label, vertix_metadata)
                VALUES (%s, %s, %s, %s);
            """, (pandapower_grid_id, int(node), label, metadata_json))

        # Step 4: Insert edges
        for u, v, attrs in graph.edges(data=True):
            edge_metadata = json.dumps(attrs, cls=CustomJSONEncoder)
            cur.execute("""
                INSERT INTO building_power.network_edges (pandapower_grid_id, source_pandapower_bus_index, target_pandapower_bus_index, directed, edge_metadata)
                VALUES (%s, %s, %s, %s, %s);
            """, (
                pandapower_grid_id,
                int(u),
                int(v),
                isinstance(graph, nx.DiGraph),
                edge_metadata
            ))       
        
        
        # Step 6: Link the pandapower network to our network record        
        # Update the record with the grid name and network_id
        cur.execute(f"""
            UPDATE building_power.{grid_catalogue_name}
            SET grid_name = %s,
            grid_description = %s,
            grid_metadata = %s
            WHERE grid_id = %s;
        """, (
            grid_name, 
            f"Network generated with {len(graph.nodes)} buses and {len(graph.edges)} connections",
            json.dumps({
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "is_directed": isinstance(graph, nx.DiGraph),
            "saved_at": pd.Timestamp.now().isoformat()
            }),
            pandapower_grid_id
        ))
        
        conn.commit()
        print(f"Saved network '{grid_name}' with grid_id={pandapower_grid_id}")
        return pandapower_grid_id
        
    except Exception as e:
        conn.rollback()
        print(f"Failed to save network: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def load_network_from_database(db_connection, network_name):
    conn = psycopg2.connect(**db_connection)
    cur = conn.cursor()
    grid_catalogue_name = 'pandapower_grids'

    def coerce_numeric_strings(obj):
        if isinstance(obj, dict):
            return {k: coerce_numeric_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [coerce_numeric_strings(item) for item in obj]
        if isinstance(obj, str):
            value = obj.strip()
            if not value:
                return obj
            lower_value = value.lower()
            if lower_value in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                return obj
            try:
                if value.isdigit() or (
                    len(value) > 1
                    and value[0] in {"+", "-"}
                    and value[1:].isdigit()
                ):
                    return int(value)
                return float(value)
            except ValueError:
                return obj
        return obj

    # Create a custom JSON decoder for complex objects
    def custom_object_hook(obj):
        if isinstance(obj, dict):
            # Handle shapely geometry objects
            if "_shapely_wkt" in obj:
                return wkt.loads(obj["_shapely_wkt"])
            
            # Handle GeopyPoint objects
            elif "_geopy_point" in obj:
                point_data = obj["_geopy_point"]
                latitude = coerce_numeric_strings(point_data["latitude"])
                longitude = coerce_numeric_strings(point_data["longitude"])
                return GeopyPoint(latitude, longitude)
            
            # Process nested dictionaries
            return {k: custom_object_hook(v) for k, v in obj.items()}
        
        # Handle lists
        elif isinstance(obj, list):
            return [custom_object_hook(item) for item in obj]
        
        # All other objects pass through unchanged
        return obj

    try:
        # Step 1: Get network_id from network name
        cur.execute(f"""
            SELECT grid_id FROM building_power.{grid_catalogue_name}
            WHERE grid_name = %s;
        """, (network_name,))
        
        result = cur.fetchone()
        if result is None:
            raise Exception(f"Network '{network_name}' not found")
        
        pandapower_grid_id = result[0]
        
        # Step 2: Load graph structure from our network tables
        graph = nx.DiGraph()
        
        # Load vertices
        cur.execute("""
            SELECT pandapower_bus_index, vertix_label, vertix_metadata 
            FROM building_power.network_vertices 
            WHERE pandapower_grid_id = %s
            ORDER BY pandapower_bus_index;
        """, (pandapower_grid_id,))
        
        for bus_index, label, metadata_json in cur.fetchall():
            # Use the custom decoder to parse the JSON
            metadata = custom_object_hook(metadata_json)
            metadata = coerce_numeric_strings(metadata)
            if not isinstance(metadata, dict):
                metadata = {}
            node_attrs = dict(metadata)
            if label is not None:
                node_attrs["name"] = label
            graph.add_node(bus_index, **node_attrs)

        # Load edges
        cur.execute("""
            SELECT source_pandapower_bus_index, target_pandapower_bus_index, directed, edge_metadata 
            FROM building_power.network_edges 
            WHERE pandapower_grid_id = %s;
        """, (pandapower_grid_id,))
        
        for source_bus, target_bus, is_directed, metadata_json in cur.fetchall():
            # Use the custom decoder to parse the JSON
            metadata = custom_object_hook(metadata_json)
            metadata = coerce_numeric_strings(metadata)
            if not isinstance(metadata, dict):
                metadata = {}
            graph.add_edge(source_bus, target_bus, **metadata)
            if not is_directed:
                graph.add_edge(target_bus, source_bus, **metadata)

        # Step3: Load the pandapower network
        grid_catalogue_name = 'pandapower_grids'
        
        # Get the grid_id for this network
        cur.execute(f"""
            SELECT grid_id FROM building_power.{grid_catalogue_name}
            WHERE grid_name = %s;
        """, (network_name,))
        
        grid_result = cur.fetchone()
        if grid_result is None:
            raise Exception(f"Pandapower grid not found for network '{network_name}'")
        
        pandapower_grid_id = grid_result[0]

        pandapower_grid = from_sql(
                        conn=conn,
                        grid_id=pandapower_grid_id,
                        grid_catalogue_name=grid_catalogue_name,
                        schema="building_power"
                    )
        
        print(f"Loaded network '{network_name}' with {len(graph.nodes)} buses and {len(graph.edges)} connections")
        return pandapower_grid_id, pandapower_grid, graph
            
    except Exception as e:
        print(f"Failed to load network: {e}")
        raise
    finally:
        cur.close()
        conn.close()