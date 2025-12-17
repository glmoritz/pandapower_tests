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

            #the maximum power per phase on single/double phase systems will be 10kWp
            if pv_peak_power[i] < 10.0:
                inverter_type = '1ph'
            elif pv_peak_power[i] < 20.0:
                inverter_type = rng.choice(['1ph', '2ph'])
            else:
                inverter_type = '3ph'

            household_params = {
                "LoadPower_kW": bus_loads[i],
                "SolarPeakPower_MW": pv_peak_power[i]/1000,
                "StorageCapacity_MWh": pv_storage_capacity[i]/1000,
                "InitialSOC_percent": initial_charge,
                "MaxChargePower_MW": MaxChargePower_kW/1000,
                "MaxDischargePower_MW": MaxDischargePower_kW/1000,
                "InverterType": inverter_type,
                "Index": f'bus{bus}'
            }
            graph.nodes[bus]['household_params'] = household_params

            #connect the household inverter to the bus (in random phases)
            inverter_phases = ['a', 'b', 'c']

            if inverter_type == '1ph':
                bus_phases = [rng.choice(['a', 'b', 'c'])]                
            elif inverter_type == '2ph':
                bus_phases = rng.choice(['a', 'b', 'c'], size=2, replace=False).tolist()                                                                
            else:
                bus_phases = ['a', 'b', 'c']

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
            


def apply_profile_to_graph(engine, grid_id: int, profile_name: str, graph: nx.Graph):      
    """
    Load power profile metadata from database and apply to graph nodes.
    All node properties are stored as a single JSON object.
    """
    # Custom decoder for complex objects
    def decode_custom_objects(obj):
        if isinstance(obj, dict):
            # Handle shapely geometry objects
            if "_shapely_wkt" in obj:
                return wkt.loads(obj["_shapely_wkt"])
            
            # Handle GeopyPoint objects
            elif "_geopy_point" in obj:
                point_data = obj["_geopy_point"]
                return GeopyPoint(point_data["latitude"], point_data["longitude"])
            
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
                historical_grid_name = f"{grid_name}_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                
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

    # Create a custom JSON decoder for complex objects
    def custom_object_hook(obj):
        if isinstance(obj, dict):
            # Handle shapely geometry objects
            if "_shapely_wkt" in obj:
                return wkt.loads(obj["_shapely_wkt"])
            
            # Handle GeopyPoint objects
            elif "_geopy_point" in obj:
                point_data = obj["_geopy_point"]
                return GeopyPoint(point_data["latitude"], point_data["longitude"])
            
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