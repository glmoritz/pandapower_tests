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
from create_random_network import generate_pandapower_net
import time
import random
from pandapower.create import create_load
from datetime import datetime, timedelta
import psycopg2

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

postgres_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'postgres_reader_model'))
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
            'python': 'postgres_reader_model.PostgresReaderModel:PostgresReaderModel'
        }
    }    

    world = mosaik.World(SIM_CONFIG)

    net, graph = generate_pandapower_net(
        CommercialRange=params['commercial_range'],
        IndustrialRange=params['industrial_range'],
        ResidencialRange=params['residential_range'],
        ForkLengthRange=params['fork_length_range'],
        LineBusesRange=params['line_buses_range'],
        LineForksRange=params['line_forks_range'],
        mv_bus_coordinates=(float(params['mv_bus_latitude']),float(params['mv_bus_longitude']))
    )

    # Power data output to test
    csv_sim_writer = world.start('CSV_writer', start_date = params['start_time'], output_file=f'{params['results_dir']}/{params['output_file']}')
    csv_writer = csv_sim_writer.CSVWriter(buff_size = params['step_size_s'])

    # Create PV system with certain configuration
    pv_sim = world.start(
                        "PVSim",
                        start_date=params['start_time'],
                        step_size=int(params['step_size_s']))

    # Create PV system
    # pv_model = pv_sim.PV.create(1, 
    #                             latitude=LAT,
    #                             area=AREA,
    #                             efficiency=params['panel_efficiency'], 
    #                             el_tilt=EL, 
    #                             az_tilt=AZ
    #                             )

    
    #Configure charger component
    # charger_sim = world.start("ChargerSim", sim_start=params['start_time'], datafile=CHARGER_DATA)
    # charger_model = charger_sim.Charger1.create(1)

    #Irradiation model
    irradiation_sim = world.start("SolarIrradiation", sim_start=params['start_time'], time_step=int(params['step_size_s']), date_format="%Y-%m-%d %H:%M:%S", type="time-based")

    #Household model
    household_sim = world.start("HouseholdProducer", sim_start=params['start_time'], time_step=int(params['step_size_s']), date_format="%Y-%m-%d %H:%M:%S", type="hybrid")    

    #Instantiate the power network    
    pp_sim = world.start("Pandapower", step_size=int(params['step_size_s']), asymmetric_flow=True)
 
    #Power Consumption Model
    db = {
        "dbname": "duilio",
        "user": "root",
        "password": "skamasfrevrest",
        "host": "103.0.1.37",
        "port": 5433  # or your port
    }
    sim_start_dt = datetime.strptime(params['start_time'], '%Y-%m-%d %H:%M:%S')
    sim_end_dt = sim_start_dt + timedelta(seconds=float(params['simulation_time_s']))
    power_consumption_sim = world.start(  "PostgresReaderModel",                                         
                                            time_resolution=int(params['step_size_s']),     
                                            time_step=int(params['step_size_s']),
                                            sim_start=params['start_time'],
                                            sim_end=sim_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                            db_connection=db,
                                            sql_rows=['power_kw'],
                                            date_format='%Y-%m-%d %H:%M:%S',
                                            type="time-based")                                          

    grid = pp_sim.Grid(net=net)

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
    extra_info = pp_sim.get_extra_info()
    loads = [e for e in grid.children if e.type == "Load"]    
    buses = [e for e in grid.children if e.type == "Bus"] 
    lines = [e for e in grid.children if e.type == "Line"] 
    trafos = [e for e in grid.children if e.type == "Transformer"]
    generators = [e for e in grid.children if e.type == "StaticGen"]     

    #output load Powers
    for load in loads:
        world.connect(load, csv_writer, "P[MW]")

    #output bus powers
    for bus in buses:
        world.connect(bus, csv_writer, "P_a[MW]")    
        world.connect(bus, csv_writer, "Vm_a[pu]")
        world.connect(bus, csv_writer, "P_b[MW]")    
        world.connect(bus, csv_writer, "Vm_b[pu]")
        world.connect(bus, csv_writer, "P_c[MW]")    
        world.connect(bus, csv_writer, "Vm_c[pu]")
        graph.nodes[bus.extra_info['index']]['bus_element'] = bus
        
    for trafo in trafos:
        world.connect(trafo, csv_writer, "Loading[%]")  

    #output line information
    for line in lines:
        world.connect(line, csv_writer, "I_a_from[kA]")
        world.connect(line, csv_writer, "I_b_from[kA]")
        world.connect(line, csv_writer, "I_c_from[kA]")
        world.connect(line, csv_writer, "Pl_a[MW]")
        world.connect(line, csv_writer, "Pl_b[MW]")
        world.connect(line, csv_writer, "Pl_c[MW]")
        
        #world.connect(line, csv_writer, "Pin[MW]")
        # world.connect(line, csv_writer, "Pout[MW]")
        # world.connect(line, csv_writer, "Pout[MW]")
        # world.connect(line, csv_writer, "VmIn[pu]")
        # world.connect(line, csv_writer, "VmOut[pu]")
        # world.connect(line, csv_writer, "QIn[MVar]")
        # world.connect(line, csv_writer, "QOut[MVar]")

    #connect charger csv file to charger connection
    #world.connect(charger_model[0],charger1, ("P[MW]","P[MW]"))
    
    #world.connect(irr_model1, pv_model[0],"DNI[W/m2]")
                    
    # world.connect(
    #                     pv_model[0],
    #                     csv_writer,
    #                     "P[MW]",
    #                 )
    
    # Run simulation

    #connect charger csv file to charger connection
    #params['solar_power_range_per_branch_kW']

    #"solar_power_range_per_branch_kW": [0.5, 100], 
    
    # For each branch add the loads as specified in the parameters

    #first, get the roots of all branches (the children of the root bus)
    
    
    for trafo in trafos:
        # Get all the lv buses that are fed by this transformer
        lv_buses = list(nx.descendants(graph, net.trafo.at[trafo.extra_info['index'], 'lv_bus']))

        # Draw the total power of the loads on this branch
        total_load_power = random.uniform(params['load_power_range_per_branch_kW'][0], 
                                      params['load_power_range_per_branch_kW'][1])
        
        # Draw the total solar power of the PV systems on this branch
        total_solar_power = random.uniform(params['solar_power_range_per_branch_kW'][0], 
                                      params['solar_power_range_per_branch_kW'][1])
        
        total_storage_capacity = random.uniform(params['storage_capacity_range_per_branch_kWh'][0], 
                                      params['storage_capacity_range_per_branch_kWh'][1])
        
        

        load_buses = random.sample(lv_buses, random.randint(1, len(lv_buses)))

        #pv_buses = random.sample(lv_buses, random.randint(1, len(lv_buses)))

        #distribute the load power to the buses
        load_distribution = np.random.uniform(size=len(load_buses))  
        bus_loads = (total_load_power/load_distribution.sum())*load_distribution

        #distribute the PV generation to the buses
        pv_distribution = np.random.uniform(size=len(load_buses)) 
        storage_distribution = np.random.uniform(size=len(load_buses))

        pv_peak_power = (pv_distribution/pv_distribution.sum())*total_solar_power
        pv_storage_capacity = (storage_distribution/storage_distribution.sum())*total_storage_capacity        
        
        #connect the PV systems to the buses
        for i, bus in enumerate(load_buses):
            
            #plot solar panel power            
            irradiation_model = irradiation_sim.SolarIrradiation(latitude=graph.nodes[bus]['latlon'][0], longitude=graph.nodes[bus]['latlon'][1])

            initial_charge = random.uniform(params['initial_capacity_range'][0], params['initial_capacity_range'][1])
        
            #the maximum charge and discharge power of the storage system will be based on Tesla Powerwall 3
            MaxChargePower_kW = (8.0/13.5)*pv_storage_capacity[i]  # 8 kW charge power, 13.5 kWh capacity
            MaxDischargePower_kW = (11.5/13.5)*pv_storage_capacity[i]  # 11.5 kW rated power, 13.5 kWh capacity

            #the maximum power per phase on single/double phase systems will be 10kWp
            if pv_peak_power[i] < 10.0:
                inverter_type = '1ph'
            elif pv_peak_power[i] < 20.0:
                inverter_type = random.choice(['1ph', '2ph'])
            else:
                inverter_type = '3ph'

            house_model = household_sim.HouseholdProducer(
                                            SolarPeakPower_MW=pv_peak_power[i]/1000,
                                            StorageCapacity_MWh=pv_storage_capacity[i]/1000,
                                            InitialSOC_percent=initial_charge,
                                            MaxChargePower_MW=MaxChargePower_kW/1000,
                                            MaxDischargePower_MW=MaxDischargePower_kW/1000,
                                            InverterType=inverter_type
                                            )
            
            #connect the irradiance model to the household model
            world.connect(irradiation_model, csv_writer, "DNI[W/m2]")
            world.connect(irradiation_model, house_model, ("DNI[W/m2]","Irradiance[W/m2]"))

            #connect the household inverter to the bus (in random phases)
            inverter_phases = ['a', 'b', 'c']

            if inverter_type == '1ph':
                bus_phases = [random.choice(['a', 'b', 'c'])]                
            elif inverter_type == '2ph':
                bus_phases = random.sample(['a', 'b', 'c'], 2)                                                                
            else:
                bus_phases = ['a', 'b', 'c']
                
            for k, bus_phase in enumerate(bus_phases):
                world.connect(house_model,graph.nodes[bus]['bus_element'], (f'Q_{inverter_phases[k]}_load[MVar]',f'Q_{bus_phase}_load[MVar]'))
                world.connect(house_model,graph.nodes[bus]['bus_element'], (f'P_{inverter_phases[k]}_load[MW]',f'P_{bus_phase}_load[MW]'))

            #first I need to find a combination of buses that have this average power
            if bus_loads[i] > 4: #our database has no loads with average power above 4kW
                min_value_kw = 1.0
                max_value_kw = 4.0
                min_parts_count = int(bus_loads[i] / max_value_kw) + 1                
                max_parts_count = int(bus_loads[i] / min_value_kw) 
                parts_count = random.randint(min_parts_count, max_parts_count)
                split_loads = [min_value_kw] * parts_count
                
                remainder = bus_loads[i] - sum(split_loads)
                while remainder > 0:
                    remainder_distribution_factors = np.random.uniform(size=len(split_loads))
                    remainder_distribution = (remainder/np.sum(remainder_distribution_factors))*remainder_distribution_factors
                    for j in range(len(split_loads)):
                        if split_loads[j] + remainder_distribution[j] <= max_value_kw*1.1:
                            split_loads[j] += remainder_distribution[j]
                            remainder -= remainder_distribution[j]                
            else:
                split_loads = [bus_loads[i]]

            #now execute a query to find buildings that have split_loads[i] as average power consumption
            sql = """
                    WITH targets(avg_power) AS (
                        SELECT unnest(%s::float8[])  -- List of target average powers
                    ), candidate_buildings AS (
                        SELECT 
                            bldg_id,
                            AVG(electricity_total_energy_consumption) / 24 AS avg_daily_power_kw
                        FROM building_power.daily_energy
                        WHERE electricity_total_energy_consumption IS NOT NULL
                        AND \"day\" >= %s
                        AND \"day\" <= %s
                        GROUP BY bldg_id
                    )
                    SELECT DISTINCT ON (t.avg_power) 
                        cb.bldg_id,
                        t.avg_power,                        
                        cb.avg_daily_power_kw,
                        ABS(cb.avg_daily_power_kw - t.avg_power) AS deviation_kw
                    FROM targets t
                    JOIN candidate_buildings cb 
                        ON ABS(cb.avg_daily_power_kw - t.avg_power) <= %s
                    ORDER BY t.avg_power, random(), deviation_kw;
                    """            

            with psycopg2.connect(**db) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (split_loads, sim_start_dt.strftime('%Y-%m-%d %H:%M:%S'), sim_end_dt.strftime('%Y-%m-%d %H:%M:%S'), 0.5))  # 0.5 kW tolerance
                    results = cur.fetchall()

            bldg_ids_list = [str(row[0]) for row in results]
            bldg_ids = ', '.join(bldg_ids_list)

            #conect the household to the power consumption of the bus            
            interval_str = f'{params['step_size_s']} seconds'            
            sql_query = f"""
                    SELECT time_bucket(INTERVAL '{interval_str}', sample_time) AS bucket,
                    SUM(electricity_total_energy_consumption) / 0.25 AS power_kw
                    FROM building_power.building_power
                    WHERE bldg_id IN ({bldg_ids})
                         AND sample_time >= '{sim_start_dt.strftime('%Y-%m-%d %H:%M:%S')}'
                         AND sample_time <= '{sim_end_dt.strftime('%Y-%m-%d %H:%M:%S')}'
                         AND electricity_total_energy_consumption IS NOT NULL
                    GROUP BY bucket
                    ORDER BY bucket                
                    """

            power_model = power_consumption_sim.PostgresReader(query=[sql_query])

            world.connect(power_model, house_model, ("power_kw","PowerConsumption[MW]"), transform=lambda kw_val: kw_val / 1000) # Convert kW to MW
            
            graph.nodes[bus]['household_element'] = house_model
            graph.nodes[bus]['irradiation_model'] = irradiation_model
            graph.nodes[bus]['power_model'] = power_model
            graph.nodes[bus]['pv_model'] = irradiation_model
            graph.nodes[bus]['buildings'] = bldg_ids_list
        
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

