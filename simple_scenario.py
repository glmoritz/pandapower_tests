import mosaik
import mosaik.util
from pv_configurations import generate_configurations, Scenarios
import simbench
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import pandapower as pp
import pandas as pd
import matplotlib

#matplotlib.use("Qt5Agg")  # Use "Qt5Agg" if you have PyQt5 installed


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
END = 3600 * 5
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
    # create empty net
    pp_sim = world.start("Pandapower", step_size=90)
    net = pp.create_empty_network()

    # create buses
    b1 = pp.create_bus(net, vn_kv=11.4, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=0.22, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=0.22, name="Bus 3")
    b4 = pp.create_bus(net, vn_kv=0.22, name="Bus 4")

    # create bus elements
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b3, p_mw=0.0010, q_mvar=0.0000, name="Load")
    #pp.create_load(net, bus=b4, p_mw=0.001, q_mvar=0.0000, name="Load")

    # create branch elements
    #pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=0.4,
        vn_hv_kv=11.4,
        vn_lv_kv=0.220,
        vk_percent=6.0,
        vkr_percent=1.425,
        pfe_kw=1.35,
        i0_percent=0.3375,
        name="Custom 11.4kV to 0.22kV Transformer"
    )

    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=1.0, name="Transformer Line",std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=b3, to_bus=b4, length_km=1.0, name="Photovoltaic Line",std_type="NAYY 4x50 SE")
    
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
    output_file = "simbench_load_plot_2.png"
    plt.savefig(output_file, dpi=300)  # Set the DPI for high resolution

    # Optionally, close the plot to free up memory
    plt.close()
 
    grid = pp_sim.Grid(net=net)
    #pv_gen = pp.create_gen(net, bus=b4, p_mw=0.0, q_mvar=0.0, name="PV System")
    pv_gen = pp_sim.ControlledGen(bus=b4)
    #pv_gen = pp_sim.Gen(bus=b4)
    #pv_gen = pp.create_sgen(net, bus=b4, p_mw=0.0, q_mvar=0.0, name="PV System")
    #extra_info = pp_sim.get_extra_info()
    
    loads = [e for e in grid.children if e.type == "Load"]    
    buses = [e for e in grid.children if e.type == "Bus"] 
    lines = [e for e in grid.children if e.type == "Line"] 
    
    for load in loads:
        world.connect(load, csv_writer, "P[MW]")
    
    for bus in buses:
        world.connect(bus, csv_writer, "P[MW]")

    for line in lines:
        world.connect(line, csv_writer, "I[kA]")
        world.connect(line, csv_writer, "Pin[MW]")
        world.connect(line, csv_writer, "Pout[MW]")
        world.connect(line, csv_writer, "Pout[MW]")
        world.connect(line, csv_writer, "VmIn[pu]")
        world.connect(line, csv_writer, "VmOut[pu]")
        world.connect(line, csv_writer, "QIn[MVar]")
        world.connect(line, csv_writer, "QOut[MVar]")

        
    #connect the solar power plant to Bus4   
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
    
    
        
    # Load the CSV file
    filename = "results.csv"
    df = pd.read_csv(filename)

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract power and current columns (assuming their naming convention includes 'P' or 'I')
    power_columns = [col for col in df.columns if "P[MW]" in col]
    current_columns = [col for col in df.columns if "I[kA]" in col]

    # Prepare subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot Power in kW
    for column in power_columns:
        axs[0].plot(df['date'], df[column] * 1000, label=f"{column} [kW]")
    axs[0].set_ylabel("Power [kW]")
    axs[0].legend()

    # Plot Current in A
    for column in current_columns:
        axs[1].plot(df['date'], df[column] * 1000, label=f"{column} [A]")
    axs[1].set_ylabel("Current [A]")
    axs[1].set_xlabel("Date")
    axs[1].legend()    

    # Add grid and adjust layout
    for ax in axs.flat:
        ax.grid(True)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig("results_subplots.pdf")
    plt.show()
    #plt.close()
    

if __name__ == '__main__':
    main()
