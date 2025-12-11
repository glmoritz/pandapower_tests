import mosaik
import mosaik.util
#from pv_configurations import generate_configurations, Scenarios
import simbench
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import pandapower as pp
import pandas as pd
import matplotlib
import nest_asyncio
import numpy as np
import re
import sys
import os
import networkx as nx
import pandapower.networks as pn
from simulation_worker.SimulationWorker import find_and_lock_param_file
from create_random_network import generate_pandapower_net, load_network_from_database, save_network_to_database
from generate_power_profile import distribute_loads_to_buses, apply_profile_to_graph, save_graph_metadata
import time
import random
from pandapower.create import create_load
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

# Add local-mosaik-pandapower-2.src.mosaik_components to Python path
pandapower_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'local-mosaik-pandapower-2', 'src'))
if pandapower_module_path not in sys.path:
    sys.path.insert(0, pandapower_module_path)

irradiation_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'irradiation_module'))
if irradiation_module_path not in sys.path:
    sys.path.insert(0, irradiation_module_path)

house_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'household_producer'))
if house_module_path not in sys.path:
    sys.path.insert(0, house_module_path)

postgres_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'postgres_model'))
if postgres_module_path not in sys.path:
    sys.path.insert(0, postgres_module_path)

import irradiation_model


def getElementbyName(grid, name):
    """
    Get the element by name from the DataFrame.
    """
    for element in grid.children:
        if element.extra_info['name'] == name:
            return element
    return None    
#matplotlib.use("Qt5Agg")  # Use "Qt5Agg" if you have PyQt5 installed           

def run_simulation(params):
    # Simulator backends
    SIM_CONFIG = {
        'ChargerSim': {        
            'python': 'mosaik_csv:CSV'
        },  
        'PVSim': {
            'python': 'mosaik_components.pv.pvsimulator:PVSimulator'
        },
        'CSV_writer': {
            'python': 'mosaik_csv_writer:CSVWriter',
        },
        "Pandapower": {
            'python': 'mosaik_components.pandapower:Simulator'
        },        
        'SolarIrradiation': {
            'python': 'irradiation_model.SolarIrradiationModel:SolarIrradiationModel'
        },
        'HouseholdProducer': {
            'python': 'household_producer.HouseholdProducerModel:HouseholdProducerModel'
        },
        'PostgresReaderModel': {
            'python': 'postgres_model.PostgresReaderModel:PostgresReaderModel'
        },
        'PostgresWriterModel': {
            'python': 'postgres_model.PostgresWriterModel:PostgresWriterModel'
        }
    }    

    world = mosaik.World(SIM_CONFIG)

    load_dotenv()

    db = {
        "dbname": os.getenv("POSTGRES_DB_NAME", "duilio"),
        "user": os.getenv("POSTGRES_DB_USER", "root"),
        "password": os.getenv("POSTGRES_DB_PASSWORD", "skamasfrevrest"),
        "host": os.getenv("POSTGRES_DB_HOST", "103.0.2.7"),
        "port": int(os.getenv("POSTGRES_DB_PORT", "5433"))
    }
    db_url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    engine = create_engine(db_url)

    net, graph = None, None
    if params['use_saved_network_if_exists']:
        try:
            pandapower_grid_id, net, graph = load_network_from_database(db, params['network_id'])
            print(f"Loaded existing network '{params['network_id']}' from database.")
        except Exception as e:
            print(f"Network '{params['network_id']} does not exist': {e}")
            pandapower_grid_id, net, graph = None, None, None
    
    if net is None or graph is None:
        net, graph = generate_pandapower_net(
            CommercialRange=params['commercial_range'],
            IndustrialRange=params['industrial_range'],
            ResidencialRange=params['residential_range'],
            ForkLengthRange=params['fork_length_range'],
            LineBusesRange=params['line_buses_range'],
            LineForksRange=params['line_forks_range'],
            mv_bus_coordinates=(float(params['mv_bus_latitude']),float(params['mv_bus_longitude']))
        )
        net.ext_grid['r0x0_max'] = 5.0
        net.ext_grid['x0x_max'] = 5.0

        # Add a new column to the net.line DataFrame
        net.line['r0_ohm_per_km'] = net.line['r_ohm_per_km'] * 3
        net.line['x0_ohm_per_km'] = net.line['x_ohm_per_km'] * 3
        net.line['c0_nf_per_km'] = net.line['c_nf_per_km'] * 3
        net.trafo['vector_group'] = 'Dyn'
        net.trafo['vk0_percent'] = net.trafo['vk_percent']
        net.trafo['mag0_percent'] = 100
        net.trafo['mag0_rx'] = 0 
        net.trafo['si0_hv_partial'] = 0.9
        net.trafo['vkr0_percent'] = net.trafo['vkr_percent']         
        pandapower_grid_id = save_network_to_database(
            graph = graph,
            net= net,
            db_connection = db,
            grid_name = params['network_id'])             
        print(f"Saved new network '{params['network_id']}' to database.")

    profile_loaded = False
    if params['use_saved_power_profile_if_exists']:
        profile_loaded = apply_profile_to_graph(
            engine,
            pandapower_grid_id,
            params['power_profile_id'],
            graph
        )        
        if profile_loaded:
            print(f"Loaded existing power profile '{params['power_profile_id']}' from database.")
        else:
            print(f"Power profile '{params['power_profile_id']}' does not exist.")
            
    
    if not profile_loaded:
        print(f"Generating Power profile '{params['power_profile_id']}'.")
        distribute_loads_to_buses(net, graph, params, db)  
        save_graph_metadata(engine, pandapower_grid_id, params['power_profile_id'], graph)
        print(f"Saved power profile '{params['power_profile_id']}' to database.")
    
    # Create PV system with certain configuration
    pv_sim = world.start(
                        "PVSim",
                        start_date=params['start_time'],
                        step_size=int(params['step_size_s']))
    
    #Irradiation model
    irradiation_sim = world.start("SolarIrradiation", sim_start=params['start_time'], time_step=int(params['step_size_s']), date_format="%Y-%m-%d %H:%M:%S", type="time-based")

    #Household model
    household_sim = world.start("HouseholdProducer", sim_start=params['start_time'], time_step=int(params['step_size_s']), date_format="%Y-%m-%d %H:%M:%S", type="hybrid")    

    #Instantiate the power network    
    pp_sim = world.start("Pandapower", step_size=int(params['step_size_s']), asymmetric_flow=True)
 
    #Power Consumption Model    
    sim_start_dt = datetime.strptime(params['start_time'], '%Y-%m-%d %H:%M:%S')
    sim_end_dt = sim_start_dt + timedelta(seconds=float(params['simulation_time_s']))
    power_consumption_sim = world.start(  "PostgresReaderModel",                                         
                                            time_resolution=1,     
                                            time_step=int(params['step_size_s']),
                                            sim_start=params['start_time'],
                                            sim_end=sim_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                            db_connection=db,
                                            sql_rows=['Power[kW]'],
                                            date_format='%Y-%m-%d %H:%M:%S',
                                            type="time-based")                                          

    result_output_model = world.start("PostgresWriterModel",                                                                                     
                                            start_date=params['start_time'],                                            
                                            db_connection=db,                                                                                                                                    
                                            write_to_db=True,
                                            simulation_params=params,
                                            output_csv=True,
                                            output_file=f'{params['results_dir']}/{params['output_file']}',
                                            time_resolution=1)                                              
    result_writer = result_output_model.PostgresWriterModel(buff_size=int(params['step_size_s']))

    # Save grid version used in this simulation
    sql = text("""UPDATE building_power.simulation_outputs
              SET pandapower_grid_id = :grid_id
              WHERE simulation_output_id = (
                  SELECT simulation_output_id 
                  FROM building_power.simulation_outputs 
                  WHERE parameters->>'network_id' = :network_id
                  ORDER BY simulation_output_id DESC 
                  LIMIT 1
              );
            """)        
    with engine.connect() as conn:
        conn.execute(sql, {"grid_id": pandapower_grid_id, "network_id": params['network_id']})
        conn.commit()  

    grid = pp_sim.Grid(net=net)
    
    extra_info = pp_sim.get_extra_info()
    loads = [e for e in grid.children if e.type == "Load"]    
    buses = [e for e in grid.children if e.type == "Bus"] 
    lines = [e for e in grid.children if e.type == "Line"] 
    trafos = [e for e in grid.children if e.type == "Transformer"]
    external_grid = [e for e in grid.children if e.type == "ExternalGrid"]
    generators = [e for e in grid.children if e.type == "StaticGen"]     

    #output load Powers
    for load in loads:
        world.connect(load, result_writer, "P[MW]")

    #output bus powers
    for bus in buses:
        world.connect(bus, result_writer, "P_a[MW]")    
        world.connect(bus, result_writer, "Vm_a[pu]")
        world.connect(bus, result_writer, "P_b[MW]")    
        world.connect(bus, result_writer, "Vm_b[pu]")
        world.connect(bus, result_writer, "P_c[MW]")    
        world.connect(bus, result_writer, "Vm_c[pu]") 
        world.connect(bus, result_writer, "Unbalance[%]")        
        graph.nodes[bus.extra_info['index']]['bus_element'] = bus
        
    for trafo in trafos:
        world.connect(trafo, result_writer, "Loading[%]") 
        world.connect(trafo, result_writer, "P_a_lv[MW]") 
        world.connect(trafo, result_writer, "P_b_lv[MW]") 
        world.connect(trafo, result_writer, "P_c_lv[MW]") 
        world.connect(trafo, result_writer, "Q_a_lv[MVar]") 
        world.connect(trafo, result_writer, "Q_b_lv[MVar]") 
        world.connect(trafo, result_writer, "Q_c_lv[MVar]")         

    #output line information
    for line in lines:
        world.connect(line, result_writer, "I_a_from[kA]")
        world.connect(line, result_writer, "I_b_from[kA]")
        world.connect(line, result_writer, "I_c_from[kA]")
        world.connect(line, result_writer, "I_n_from[kA]")
        world.connect(line, result_writer, "Pl_a[MW]")
        world.connect(line, result_writer, "Pl_b[MW]")        
        world.connect(line, result_writer, "Pl_c[MW]")
        world.connect(line, result_writer, "Loading[%]") 
        world.connect(line, result_writer, "Loading[%]") 
    
    for ext_grid in external_grid:
        world.connect(ext_grid, result_writer, "P_a[MW]")            
        world.connect(ext_grid, result_writer, "P_b[MW]")            
        world.connect(ext_grid, result_writer, "P_c[MW]")  

    for bus in buses:        
        graph.nodes[bus.extra_info['index']]['bus_element'] = bus    

    
    # For each branch add the loads as specified in the parameters
    #first, get the roots of all branches (the children of the root bus)
    for trafo in trafos:
        # Get all the lv buses that are fed by this transformer
        lv_buses = list(nx.descendants(graph, net.trafo.at[trafo.extra_info['index'], 'lv_bus']))
        
        #connect the PV systems to the buses
        for i, bus in enumerate(lv_buses):

            if 'household_params' in graph.nodes[bus]:
                #plot solar panel power            
                irradiation_model = irradiation_sim.SolarIrradiation(latitude=graph.nodes[bus]['latlon'][0], longitude=graph.nodes[bus]['latlon'][1])
                
                # Extract only the parameters expected by HouseholdProducer constructor
                hp = graph.nodes[bus]['household_params']
                household_constructor_params = {
                    'SolarPeakPower_MW': hp['SolarPeakPower_MW'],
                    'StorageCapacity_MWh': hp['StorageCapacity_MWh'],
                    'InitialSOC_percent': hp['InitialSOC_percent'],
                    'MaxChargePower_MW': hp['MaxChargePower_MW'],
                    'MaxDischargePower_MW': hp['MaxDischargePower_MW'],
                    'InverterType': hp['InverterType']
                }
                house_model = household_sim.HouseholdProducer(**household_constructor_params)
                #connect the irradiance model to the household model
                world.connect(irradiation_model, result_writer, "DNI[W/m2]")
                world.connect(irradiation_model, house_model, ("DNI[W/m2]","Irradiance[W/m2]"))

                #output house statistics to simulation results
                #world.connect(house_model, result_writer, 'EnergyExported[MWh]')
                #world.connect(house_model, result_writer, 'EnergyImported[MWh]')
                #world.connect(house_model, result_writer, 'PVEnergyGeneration[MWh]')
                #world.connect(house_model, result_writer, 'EnergyConsumption[MWh]')
                world.connect(house_model, result_writer, 'SOC[MWh]')
                #world.connect(house_model, result_writer, 'BatteryEnergyStored[MWh]')
                #world.connect(house_model, result_writer, 'BatteryEnergyConsumed[MWh]')
                world.connect(house_model, result_writer, 'PVPowerGeneration[MW]')
                world.connect(house_model, result_writer, 'BatteryPower[MW]')            
     
                #connect the household inverter to the bus (in random phases)
                inverter_phases = ['a', 'b', 'c']

                if 'inverter_connection_phases' in graph.nodes[bus]:                
                    for k, bus_phase in enumerate(graph.nodes[bus]['inverter_connection_phases']):
                        world.connect(house_model,graph.nodes[bus]['bus_element'], (f'Q_{inverter_phases[k]}_load[MVar]',f'Q_{bus_phase}_load[MVar]'))
                        world.connect(house_model,graph.nodes[bus]['bus_element'], (f'P_{inverter_phases[k]}_load[MW]',f'P_{bus_phase}_load[MW]'))
                        world.connect(house_model, result_writer, f'P_{bus_phase}_load[MW]')            

                if 'connected_buildings_ids' in graph.nodes[bus]: 
                    bldg_ids = ', '.join([str(i) for i in graph.nodes[bus]['connected_buildings_ids']])

                    #conect the household to the power consumption of the bus            
                    interval_str = f'{params['step_size_s']} seconds'            
                    sql_query = f"""
                            SELECT time_bucket(INTERVAL '{interval_str}', sample_time) AS bucket,
                            SUM(electricity_total_energy_consumption) / 0.25 AS \"Power[kW]\"
                            FROM building_power.building_power
                            WHERE bldg_id IN ({bldg_ids})
                                AND sample_time >= '{sim_start_dt.strftime('%Y-%m-%d %H:%M:%S')}'
                                AND sample_time <= '{sim_end_dt.strftime('%Y-%m-%d %H:%M:%S')}'
                                AND electricity_total_energy_consumption IS NOT NULL
                            GROUP BY bucket
                            ORDER BY bucket                
                            """

                    power_model = power_consumption_sim.PostgresReader(query=[sql_query])

                    world.connect(power_model, house_model, ("Power[kW]","PowerConsumption[MW]"), transform=lambda kw_val: kw_val / 1000) # Convert kW to MW
                    world.connect(power_model, result_writer, 'Power[kW]')            
                    
            
    world.run(until=float(params['simulation_time_s']))
  


if __name__ == "__main__":
    # Load parameters from JSON file
    params = None
    while params is None:
        params = find_and_lock_param_file()

        if params is None:
            time.sleep(10)
        else:
            # Run the simulation with the loaded parameters
            run_simulation(params)

