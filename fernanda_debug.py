import mosaik
import mosaik.util
from pv_configurations import generate_configurations, Scenarios
import simbench
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import pandapower as pp
import pandas as pd
import matplotlib
import nest_asyncio
import numpy as np
import os
import sys

# Add local-mosaik-pandapower-2.src.mosaik_components to Python path
module_path = os.path.abspath(os.path.join(os.getcwd(), 'local-mosaik-pandapower-2', 'src'))

if module_path not in sys.path:
    sys.path.insert(0, module_path)


def getElementbyName(grid, name):
    """
    Get the element by name from the DataFrame.
    """
    for element in grid.children:
        if element.extra_info['name'] == name:
            return element
    return None    
#matplotlib.use("Qt5Agg")  # Use "Qt5Agg" if you have PyQt5 installed

# Simulator backends
SIM_CONFIG = {
    'ChargerSim': {        
        'python': 'mosaik_csv:CSV'
    },  
    # 'PVSim': {
    #     'python': 'mosaik_components.pv.photovoltaic_simulator:PVSimulator'
    # },
    'CSV_writer': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    "Pandapower": {
        'python': 'mosaik_components.pandapower:Simulator'
    }
}

START = "2024-12-01 00:15:00"
END = 3600 * 72
STEP_SIZE = 60 * 15
CHARGER_DATA = "./dados-carregamento-normal.csv"

#this is needed to use mosaik on ipynbs
nest_asyncio.apply()

world = mosaik.World(SIM_CONFIG)



#Configure charger component
charger_sim = world.start("ChargerSim", sim_start=START, datafile=CHARGER_DATA)
charger_model = charger_sim.Charger1.create(1)
#meteo_model = meteo_sim.Braunschweig.create(1)

# Create PV system
# pv_count = 5
# pv_config = {str(i) : generate_configurations(Scenarios.HOUSE) for i in range(pv_count)}
# pv_sim = world.start(
#             "PVSim",
#             start_date=START,
#             step_size=STEP_SIZE,
#             pv_data=pv_config,
#         )
# pv_model = pv_sim.PVSim.create(pv_count)

# Power data output to test
csv_sim_writer = world.start('CSV_writer', start_date = START,
                                        output_file='results.csv')
csv_writer = csv_sim_writer.CSVWriter(buff_size = STEP_SIZE)

#Instantiate the power network
# create empty net
pp_sim = world.start("Pandapower", step_size=90)

net = pp.from_json("example.json")

grid = pp_sim.Grid(net=net)

extra_info = pp_sim.get_extra_info()


loads = [e for e in grid.children if e.type == "Load"]    
buses = [e for e in grid.children if e.type == "Bus"] 
lines = [e for e in grid.children if e.type == "Line"] 


#connect a charger to B5
#charger1 = pp_sim.ControlledGen(bus=buses[1].extra_info['index'])
#charger1 = pp_sim.ControlledGen(bus=getElementbyName(grid,'B5').extra_info['index'])
generators = [e for e in grid.children if e.type == "StaticGen"] 

#output load Powers
for load in loads:
    world.connect(load, csv_writer, "P[MW]")

#output bus powers
for bus in buses:
    world.connect(bus, csv_writer, "P[MW]")

#output line information
for line in lines:
    world.connect(line, csv_writer, "I[kA]")
    #world.connect(line, csv_writer, "Pin[MW]")
    # world.connect(line, csv_writer, "Pout[MW]")
    # world.connect(line, csv_writer, "Pout[MW]")
    # world.connect(line, csv_writer, "VmIn[pu]")
    # world.connect(line, csv_writer, "VmOut[pu]")
    # world.connect(line, csv_writer, "QIn[MVar]")
    # world.connect(line, csv_writer, "QOut[MVar]")

#connect charger csv file to charger connection
world.connect(charger_model[0],getElementbyName(grid,'B5'), ("P[MW]","P_load[MW]"))

#plot charger power
world.connect(charger_model[0], csv_writer, "P[MW]")

    
world.run(until=END)