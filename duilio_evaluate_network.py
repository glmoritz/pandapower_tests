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
import pandapower.networks as pn
from simulation_worker.SimulationWorker import find_and_lock_param_file
from create_random_network import generate_pandapower_net
import time

# Add local-mosaik-pandapower-2.src.mosaik_components to Python path
pandapower_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'local-mosaik-pandapower-2', 'src'))
if pandapower_module_path not in sys.path:
    sys.path.insert(0, pandapower_module_path)

irradiation_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'irradiation_module'))
if irradiation_module_path not in sys.path:
    sys.path.insert(0, irradiation_module_path)

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
        }
    }

    world = mosaik.World(SIM_CONFIG)

    net = generate_pandapower_net(
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
                        step_size=params['step_size_s'])

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
    irradiation_sim = world.start("SolarIrradiation", sim_start=params['start_time'], time_step=params['step_size_s'], date_format="%Y-%m-%d %H:%M:%S", type="time-based")
    irr_model1 = irradiation_sim.SolarIrradiation(latitude=-25.4484, longitude=-49.2323)

    #Instantiate the power network    
    pp_sim = world.start("Pandapower", step_size=params['step_size_s'], asymmetric_flow=True)
 
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

    #plot irradiance power
    world.connect(irr_model1, csv_writer, "DNI[W/m2]")

    world.connect(irr_model1, pv_model[0],"DNI[W/m2]")
                    
    world.connect(
                        pv_model[0],
                        csv_writer,
                        "P[MW]",
                    )
    
    # Run simulation

    #connect charger csv file to charger connection
    world.connect(charger_model[0],getElementbyName(grid,'Bus C20'), ("P[MW]","P_a_load[MW]"))
        
    world.run(until=params['simulation_time_s'])

    


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

