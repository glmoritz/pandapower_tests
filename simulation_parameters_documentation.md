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
| `use_saved_network_if_exists` | boolean | If true, load existing network from database; if false, generate new network and overwrite existing one | `true` |
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
| `initial_capacity_range` | array[float, float] | [min, max] initial battery state of charge in percent | `[0.0, 100.0]` |

## Network Generation Behavior

The `use_saved_network_if_exists` parameter controls network loading/generation behavior:

- **`true`**: The simulation attempts to load an existing network with the specified `network_id` from the database. If found, the existing network topology is used. If not found, a new random network is generated and saved to the database.

- **`false`**: The simulation always generates a new random network using the specified parameters and saves it to the database. If a network with the same `network_id` already exists in the database, it will be overwritten.

This allows for reproducible simulations when using saved networks, or for testing different network configurations when generating new networks.

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
