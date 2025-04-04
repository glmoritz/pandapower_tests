import mosaik
import mosaik.util
from pv_configurations import generate_configurations, Scenarios
import simbench
import pandapower.plotting as plot
import matplotlib.pyplot as plt


# Simulator backends
SIM_CONFIG = {
    'MeteoSim': {        
        'python': 'mosaik_csv:CSV'
    },  
    'PVSim': {
        'python': 'mosaik_components.pv.photovoltaic_simulator:PVSimulator'
    },
    'CSV_writer': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    "Pandapower": {
        'python': 'mosaik_components.pandapower:Simulator'
    }
}

START = "2020-01-01 08:00:00"
END = 3600 * 12
STEP_SIZE = 60 * 15
METEO_DATA = "./solar_data/Braunschweig_meteodata_2020_15min.csv"

def main():
    world = mosaik.World(SIM_CONFIG)

    # Configure weather component
    meteo_sim = world.start("MeteoSim", sim_start=START, datafile=METEO_DATA)
    meteo_model = meteo_sim.Braunschweig.create(1)

    # Create PV system
    pv_count = 5
    pv_config = {str(i) : generate_configurations(Scenarios.HOUSE) for i in range(pv_count)}
    pv_sim = world.start(
                "PVSim",
                start_date=START,
                step_size=STEP_SIZE,
                pv_data=pv_config,
            )
    pv_model = pv_sim.PVSim.create(pv_count)

    # Power data output to test
    csv_sim_writer = world.start('CSV_writer', start_date = START,
                                           output_file='results.csv')
    csv_writer = csv_sim_writer.CSVWriter(buff_size = STEP_SIZE)

    #Instantiate the power network
    pp_sim = world.start("Pandapower", step_size=900)
    net = simbench.get_simbench_net(sb_code_info='1-LV-urban6--2-sw')        

    # Extract the load buses and their respective power values
    load_buses = net.load.bus.values
    load_names = net.load.name.values  # Load names
    bus_names = net.bus.loc[load_buses, 'name'].values  # Bus names where loads are connected


    fig, ax = plt.subplots()
    plot.simple_plot(net, ax=ax, show_plot=False)

    # Annotate the plot with bus and load names
    for idx, bus in enumerate(load_buses):
        load_name = load_names[idx]
        bus_name = bus_names[idx]
        # Bus coordinates (from the bus table)
        x = net.bus_geodata.loc[bus, 'x']
        y = net.bus_geodata.loc[bus, 'y']
        
        # Annotate the plot with the bus and load names at the bus location
        ax.text(x, y, f'Bus: {bus_name}\nLoad: {load_name}', 
                fontsize=5, color='blue')


    # Save the plot to a PNG file
    output_file = "simbench_load_plot.png"
    plt.savefig(output_file, dpi=300)  # Set the DPI for high resolution

    # Optionally, close the plot to free up memory
    plt.close()
 
    grid = pp_sim.Grid(net=net)
    pv_gen = pp_sim.ControlledGen(bus=net.bus[net.bus['name'] == 'LV6.201 Bus 53'].index[0])
    extra_info = pp_sim.get_extra_info()
    
    loads = [e for e in grid.children if e.type == "Load"]    
    buses = [e for e in grid.children if e.type == "Bus"] 
    
    for load in loads:
        world.connect(load, csv_writer, "P[MW]")
    
    for bus in buses:
        world.connect(bus, csv_writer, "P[MW]")

    #connect the solar power plant to Bus53    
    world.connect(pv_model[0], 
                  pv_gen, 
                  ("P[MW]", "P[MW]")                  
                )

    world.connect(
                    meteo_model[0],
                    pv_model[0],
                    ("GlobalRadiation", "GHI[W/m2]"),
                    ("AirPressHourly", "Air[Pa]"),
                    ("AirTemperature", "Air[C]"),
                    ("WindSpeed", "Wind[m/s]"),
                )

    world.connect(
                    pv_model[0],
                    csv_writer,
                    "P[MW]",
                )

    # Run simulation
    world.run(until=END)
    
        
    #grid = pandapower.Grid()


    # Connect the solar panel to the grid
    #world.connect(solar_panel, grid, ('power_output', 'solar_p_mw'))

    # Run simulation
    #mosaik.util.connect_many_to_one([solar_panel], grid)
    

if __name__ == '__main__':
    main()
