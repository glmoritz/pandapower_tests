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

START = "2024-12-01 01:00:00"
END = 3600 * 24
STEP_SIZE = 60 * 15
CHARGER_DATA = "./dados-carregamento-normal.csv"

def main():
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
    
    net = pp.from_json("my_network.json")
    
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
    output_file = "powerfactoryLDF.png"
    plt.savefig(output_file, dpi=300)  # Set the DPI for high resolution

    # Optionally, close the plot to free up memory
    plt.close()
    #charger = pp.create_gen(net, bus=buses[1], p_mw=0.0, q_mvar=0.0, name="Charger 1")

    grid = pp_sim.Grid(net=net)
    
    extra_info = pp_sim.get_extra_info()
    
    loads = [e for e in grid.children if e.type == "Load"]    
    buses = [e for e in grid.children if e.type == "Bus"] 
    lines = [e for e in grid.children if e.type == "Line"] 

    
    
    charger1 = pp_sim.ControlledGen(bus=buses[1].extra_info['index'])
    #pv_gen = pp_sim.Gen(bus=b4)
    #pv_gen = pp.create_sgen(net, bus=b4, p_mw=0.0, q_mvar=0.0, name="PV System")
    
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

    world.connect(charger_model[0],charger1, ("P[MW]","P[MW]"))
    world.connect(charger_model[0], csv_writer, "P[MW]")

    # Run simulation
    
    world.run(until=END)
    
    # Load the generated CSV file
    filename = "results.csv"  # The original CSV file
    output_filename = "results_renamed.csv"  # The new CSV file with renamed columns

    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(filename)

    # Rename the columns dynamically based on `lines[x].extra_info['name']`
    # Assuming you have access to the `lines` list and their `extra_info` here
    column_mapping = {}
    for line in lines:
        line_name = line.extra_info.get('name', line.eid)
        column_mapping[f"Pandapower-0.{line.eid}-I[kA]"] = f"{line_name}_I[kA]"
        column_mapping[f"Pandapower-0.{line.eid}-Pin[MW]"] = f"{line_name}_Pin[MW]"
        column_mapping[f"Pandapower-0.{line.eid}-Pout[MW]"] = f"{line_name}_Pout[MW]"
        column_mapping[f"Pandapower-0.{line.eid}-VmIn[pu]"] = f"{line_name}_VmIn[pu]"
        column_mapping[f"Pandapower-0.{line.eid}-VmOut[pu]"] = f"{line_name}_VmOut[pu]"
        column_mapping[f"Pandapower-0.{line.eid}-QIn[MVar]"] = f"{line_name}_QIn[MVar]"
        column_mapping[f"Pandapower-0.{line.eid}-QOut[MVar]"] = f"{line_name}_QOut[MVar]"

    for load in loads:
        load_name = load.extra_info.get('name', load.eid)
        column_mapping[f"Pandapower-0.{load.eid}-P[MW]"] = f"{load_name}_P[MW]"
    
    for bus in buses:
        bus_name = load.extra_info.get('name', bus.eid)
        column_mapping[f"Pandapower-0.{bus.eid}-P[MW]"] = f"{bus_name}_P[MW]"


    # Apply the column renaming
    df.rename(columns=column_mapping, inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_filename, index=False)

    print(f"Renamed CSV saved as {output_filename}")
        
    # Load the CSV file
    #filename = "resultsPowerFactoryLDF.csv"
    #df = pd.read_csv(filename)

    # Convert the 'date' column to datetime format
    #df['date'] = pd.to_datetime(df['date'])

    # Extract power and current columns (assuming their naming convention includes 'P' or 'I')
    # power_columns = [col for col in df.columns if "P[MW]" in col]
    # current_columns = [col for col in df.columns if "I[kA]" in col]

    # # Prepare subplots
    # fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # # Plot Power in kW
    # for column in power_columns:
    #     axs[0].plot(df['date'], df[column] * 1000, label=f"{column} [kW]")
    # axs[0].set_ylabel("Power [kW]")
    # axs[0].legend()

    # # Plot Current in A
    # for column in current_columns:
    #     axs[1].plot(df['date'], df[column] * 1000, label=f"{column} [A]")
    # axs[1].set_ylabel("Current [A]")
    # axs[1].set_xlabel("Date")
    # axs[1].legend()    

    # # Add grid and adjust layout
    # for ax in axs.flat:
    #     ax.grid(True)
    # plt.tight_layout()

    # # Save the plot to a file
    # plt.savefig("results_subplots.pdf")
    # plt.show()
    #plt.close()
    

if __name__ == '__main__':
    main()
