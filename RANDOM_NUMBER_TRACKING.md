# Random Number Usage Tracking

This document tracks all uses of random number generation in the pandapower_tests project. This is important for reproducibility and debugging.

## Summary

Random numbers are used throughout the project for:
1. **Network topology generation** - Creating random network structures
2. **Power load distribution** - Assigning random loads to buses
3. **Solar irradiation modeling** - Stochastic cloud state transitions
4. **Spatial positioning** - Random placement of network nodes
5. **Database queries** - Random selection of building power profiles

---

## 1. Main Simulation Entry Point

### File: `duilio_3ph_evaluate.py`

**Line 76-77: Random Seed Initialization**
```python
#init the random seed
rng = np.random.default_rng(seed = params.get('random_seed', 1234567890))
```
- **Purpose**: Initialize NumPy random number generator with seed from parameters
- **Controllability**: ✅ Yes - controlled by `params['random_seed']`
- **Default Value**: 1234567890
- **Impact**: High - This should control all NumPy random operations
- **⚠️ ISSUE**: The `rng` variable is created but **NOT USED** anywhere in the code! All subsequent random calls use the global random state instead.

---

## 2. Power Profile Generation

### File: `generate_power_profile.py`

#### Line 3, 5: Module Imports
```python
import random
```
- Imported twice (redundant)

#### Line 32-33: Load Power Generation
```python
total_load_power = random.uniform(params['load_power_range_per_branch_kW'][0], 
                                  params['load_power_range_per_branch_kW'][1])
```
- **Purpose**: Generate random total load power per transformer branch
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform
- **Parameters**: From config `load_power_range_per_branch_kW`

#### Line 36-37: Solar Power Generation
```python
total_solar_power = random.uniform(params['solar_power_range_per_branch_kW'][0], 
                                   params['solar_power_range_per_branch_kW'][1])
```
- **Purpose**: Generate random total solar power per branch
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform
- **Parameters**: From config `solar_power_range_per_branch_kW`

#### Line 39-40: Storage Capacity Generation
```python
total_storage_capacity = random.uniform(params['storage_capacity_range_per_branch_kWh'][0], 
                                        params['storage_capacity_range_per_branch_kWh'][1])
```
- **Purpose**: Generate random storage capacity per branch
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform
- **Parameters**: From config `storage_capacity_range_per_branch_kWh`

#### Line 43: Bus Selection
```python
load_buses = random.sample(lv_buses, random.randint(1, len(lv_buses)))
```
- **Purpose**: Randomly select buses for load placement AND randomly determine how many buses
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: 
  - `random.randint()`: Uniform discrete
  - `random.sample()`: Sampling without replacement

#### Line 46: Load Distribution
```python
load_distribution = np.random.uniform(size=len(load_buses))
```
- **Purpose**: Random distribution factors for loads across buses
- **Controllability**: ❌ No - uses global NumPy random state
- **Distribution**: Uniform [0, 1)

#### Line 50-51: PV and Storage Distribution
```python
pv_distribution = np.random.uniform(size=len(load_buses))
storage_distribution = np.random.uniform(size=len(load_buses))
```
- **Purpose**: Random distribution factors for PV and storage across buses
- **Controllability**: ❌ No - uses global NumPy random state
- **Distribution**: Uniform [0, 1)

#### Line 58: Initial State of Charge
```python
initial_charge = random.uniform(params['initial_capacity_range'][0], 
                                params['initial_capacity_range'][1])
```
- **Purpose**: Random initial battery charge
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform
- **Parameters**: From config `initial_capacity_range`

#### Line 68: Inverter Type Selection
```python
inverter_type = random.choice(['1ph', '2ph'])
```
- **Purpose**: Randomly choose between single-phase and two-phase inverter
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete
- **Condition**: Only when `10.0 <= pv_peak_power[i] < 20.0`

#### Line 87: Single-Phase Connection
```python
bus_phases = [random.choice(['a', 'b', 'c'])]
```
- **Purpose**: Random phase selection for single-phase inverter
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete (3 choices)

#### Line 89: Two-Phase Connection
```python
bus_phases = random.sample(['a', 'b', 'c'], 2)
```
- **Purpose**: Random two-phase selection for two-phase inverter
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform sampling without replacement

#### Line 101: Load Splitting
```python
parts_count = random.randint(min_parts_count, max_parts_count)
```
- **Purpose**: Randomly determine how many parts to split large loads into
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete
- **Condition**: When `bus_loads[i] > 4.0` kW

#### Line 106: Remainder Distribution
```python
remainder_distribution_factors = np.random.uniform(size=len(split_loads))
```
- **Purpose**: Random factors for distributing remainder load across split parts
- **Controllability**: ❌ No - uses global NumPy random state
- **Distribution**: Uniform [0, 1)

#### Line 144: Database Query Randomization
```sql
ORDER BY t.avg_power, random(), deviation_kw;
```
- **Purpose**: Random tie-breaking in SQL query for building selection
- **Controllability**: ❌ No - uses PostgreSQL's random() function
- **Impact**: Affects which building power profiles are selected

---

## 3. Network Generation

### File: `create_random_network.py`

#### Line 3, 5: Module Imports
```python
import random
```
- Imported twice (redundant)

#### Line 329: Commented Seed
```python
#random.seed(3333)
```
- **Status**: ⚠️ COMMENTED OUT - seed is not set!
- **Impact**: Network generation is not reproducible

#### Line 409: Fork Count
```python
Forks = random.randint(*LineForksRange)
```
- **Purpose**: Random number of forks per line branch
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete
- **Parameters**: From function parameter `LineForksRange`

#### Line 412: Bus Count
```python
NumBuses = random.randint(*LineBusesRange)
```
- **Purpose**: Random number of buses per line
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete
- **Parameters**: From function parameter `LineBusesRange`

#### Line 418: Fork Bus Distribution
```python
fork_bus_counts[random.randint(0, Forks - 1)] += 1
```
- **Purpose**: Randomly assign buses to forks
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete

#### Line 443: Working Bus Selection
```python
working_bus = random.choice(added_busses)
```
- **Purpose**: Randomly select bus to connect next fork
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete

#### Line 453-455: Transformer Counts
```python
Ct = random.randint(*CommercialRange)
It = random.randint(*IndustrialRange)
Rt = random.randint(*ResidencialRange)
```
- **Purpose**: Random number of each transformer type
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Uniform discrete
- **Parameters**: From function parameters

#### Line 734: Angular Perturbation
```python
delta_theta = random.gauss(0, max_angle)
```
- **Purpose**: Random angular deviation for node positioning
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Gaussian (Normal)
- **Mean**: 0
- **Std Dev**: `max_angle`

#### Line 756: Radial Perturbation
```python
delta_r = random.gauss(0, max_distance)
```
- **Purpose**: Random radial distance variation for node positioning
- **Controllability**: ❌ No - uses global `random` module
- **Distribution**: Gaussian (Normal)
- **Mean**: 0
- **Std Dev**: `max_distance`

---

## 4. Network Generation (Backup Version)

### File: `create_random_network_bk.py`

Similar random usage patterns to `create_random_network.py`:

#### Line 97: Fork Count
```python
Forks = random.randint(*LineForksRange)
```

#### Line 100: Fork Lengths
```python
Fork_Lengths = [random.randint(*ForkLengthRange) for _ in range(Forks)]
```

#### Line 103: Bus Count
```python
NumBuses = random.randint(*LineBusesRange)
```

#### Line 109: Fork Assignment
```python
fork_bus_counts[random.randint(0, Forks - 1)] += 1
```

#### Line 120: Cut Points for Line Segments
```python
cuts = sorted([random.uniform(0, total_length) for _ in range(n_hops - 1)])
```

#### Line 150: Working Bus Selection
```python
working_bus = random.choice(added_busses)
```

#### Line 160-162: Transformer Counts
```python
Ct = random.randint(*CommercialRange)
It = random.randint(*IndustrialRange)
Rt = random.randint(*ResidencialRange)
```

#### Line 302: Radius Generation
```python
radius = random.uniform(max_radius + min_distance_m * 1.5, max_radius + min_distance_m * 3)
```

#### Line 310: Angle Generation
```python
angle = i * angle_step + random.uniform(-angle_step*0.9, angle_step*0.9)
```

#### Line 336: Angle Adjustment
```python
angle += random.uniform(-angle_step/8, angle_step/8)
```

#### Line 448: Random Destination Angle
```python
angle = random.uniform(0, 360)
```

#### Line 488: Angle Offset
```python
angle_offset = random.uniform(-angle_step / 2, angle_step / 2)
```

#### Line 496: Shuffle Angles
```python
random.shuffle(angles_to_try)
```

---

## 5. Network Generation (Fixed Version)

### File: `create_random_network_fixed.py`

#### Line 440: Load Power Randomization
```python
load_p_mw = random.uniform(0.4, 1.0) * default_max_load / 1000
```
- **Purpose**: Randomize load between 40% and 100% of default max
- **Controllability**: ❌ No

#### Line 541-542: Initial Position
```python
radius = random.uniform(10, 100)
angle = random.uniform(0, 360)
```

#### Line 567, 569: Angle Selection
```python
angle = random.uniform(0, 360)
# or
angle = random.uniform(angle_range[0], angle_range[1])
```

#### Line 572: Radius with Jitter
```python
radius = layout_radius * (depth + 1) * random.uniform(0.8, 1.2)
```

#### Line 579-580: Position Offset
```python
offset_x = random.uniform(-10, 10)
offset_y = random.uniform(-10, 10)
```

#### Line 629-630: Subtree Shift
```python
shift_angle = random.uniform(0, 2 * np.pi)
shift_distance = random.uniform(0, max_distance)
```

#### Line 636: Rotation Angle
```python
rotation_angle = random.uniform(-30, 30)
```

#### Line 696-697: Branching Logic
```python
if random.random() < branch_probability:
    num_branches = random.randint(1, min(max_branches, target_bus_count - current_node_count))
```

#### Line 720: Distance Generation
```python
distance = random.uniform(30, 50)
```

#### Line 737: Subtree Randomization Decision
```python
if random.random() < 0.3:
```

---

## 6. Solar Irradiation Model

### File: `irradiation_model/SolarIrradiationModel.py`

#### Line 94: Initial Cloud State
```python
'cloud_state': np.random.choice(self.cloud_states),
```
- **Purpose**: Initialize cloud state randomly for each solar entity
- **Controllability**: ❌ No - uses global NumPy random state
- **Distribution**: Uniform discrete
- **States**: [0, 1, 2] representing clear, partly cloudy, cloudy

#### Line 123: Markov Chain Transition
```python
next_state = np.random.choice(self.cloud_states, p=P_n[current_state])
```
- **Purpose**: Stochastic cloud state transition using Markov chain
- **Controllability**: ❌ No - uses global NumPy random state
- **Distribution**: Discrete with transition probabilities
- **Transition Matrix**:
  ```
  [0.8, 0.15, 0.05]  # From clear
  [0.1, 0.8,  0.1 ]  # From partly cloudy
  [0.05, 0.15, 0.8]  # From cloudy
  ```

---

## 7. PostgreSQL Writer Model

### File: `postgres_model/PostgresWriterModel.py`

#### Line 5: Import
```python
import random
```
- **Note**: Imported but not visibly used in the first 50 lines shown

---

## Critical Issues and Recommendations

### 🚨 Critical Issues

1. **Unused Random Seed in Main Simulation**
   - `duilio_3ph_evaluate.py` Line 77 creates `rng` but never uses it
   - All random operations use unseeded global state
   - **Result**: Simulations are NOT reproducible even with `random_seed` parameter

2. **Multiple Random Generators**
   - Mix of `random` (Python stdlib) and `np.random` (NumPy)
   - No coordination between them
   - Setting one seed doesn't affect the other

3. **PostgreSQL Random Function**
   - Database `random()` function has its own seed state
   - Cannot be controlled from Python code
   - Different results on different database servers

4. **Commented-Out Seed**
   - `create_random_network.py` Line 329 has commented seed
   - Network generation is not reproducible

### ✅ Recommendations

1. **Implement Proper Seeding**
   ```python
   # In duilio_3ph_evaluate.py
   seed = params.get('random_seed', 1234567890)
   random.seed(seed)
   np.random.seed(seed)
   ```

2. **Use Single RNG Throughout**
   ```python
   # Create RNG and pass it to all functions
   rng = np.random.default_rng(seed=seed)
   # Then use: rng.uniform(), rng.choice(), etc.
   ```

3. **Document Random State Requirements**
   - Each function that uses randomness should accept `rng` parameter
   - Document which random generator each function uses

4. **Database Query Determinism**
   - Replace `ORDER BY random()` with deterministic sorting
   - Or use seeded PostgreSQL random: `SELECT setseed(0.5)`

5. **Add Reproducibility Tests**
   - Unit tests that verify same seed → same results
   - Integration tests for full simulation reproducibility

### 📊 Random Number Usage Summary

| File | Random Calls | Seeded? | Impact |
|------|--------------|---------|---------|
| `duilio_3ph_evaluate.py` | 1 (unused) | ⚠️ Yes (but unused) | High |
| `generate_power_profile.py` | 14+ | ❌ No | High |
| `create_random_network.py` | 10+ | ❌ No (commented) | High |
| `create_random_network_bk.py` | 15+ | ❌ No | Medium |
| `create_random_network_fixed.py` | 15+ | ❌ No | Medium |
| `irradiation_model/SolarIrradiationModel.py` | 2 | ❌ No | Medium |
| SQL Queries | 1+ | ❌ No | Low |

### 🎯 Priority Actions

1. **URGENT**: Fix unused `rng` variable in `duilio_3ph_evaluate.py`
2. **HIGH**: Implement consistent seeding across all modules
3. **MEDIUM**: Refactor to use single RNG instance passed as parameter
4. **LOW**: Replace SQL `random()` with deterministic ordering

---

## Configuration Parameters Affecting Randomness

From simulation parameters, these control the ranges but not the seed:

- `random_seed`: Intended seed (currently not working)
- `load_power_range_per_branch_kW`: Range for load power
- `solar_power_range_per_branch_kW`: Range for solar power
- `storage_capacity_range_per_branch_kWh`: Range for storage
- `initial_capacity_range`: Range for initial battery charge
- `commercial_range`: Range for commercial transformer count
- `industrial_range`: Range for industrial transformer count
- `residential_range`: Range for residential transformer count
- `fork_length_range`: Range for fork lengths
- `line_buses_range`: Range for buses per line
- `line_forks_range`: Range for forks per line

---

## ✅ FIXED - Implementation Status (2025-12-11)

### Successfully Implemented Independent RNG Streams

The project now uses **SeedSequence-based independent RNG streams** for reproducible, isolated randomness:

#### **Key Changes Made:**

1. **`duilio_3ph_evaluate.py`** - Main simulation coordinator
   - Creates master seed from `params['random_seed']`
   - Uses `SeedSequence.spawn()` to create independent child seeds
   - Passes dedicated RNG to each component:
     - `rng_network_generation` → network topology generation
     - `rng_power_profile` → power distribution
     - `rng_irradiation` → solar irradiation model
     - Additional streams reserved for future use

2. **`generate_power_profile.py`** - Power distribution
   - Added `rng` parameter to `distribute_loads_to_buses()`
   - Replaced all `random.*` calls with `rng.*` methods
   - Replaced all `np.random.*` calls with `rng.*` methods
   - ✅ **14 random calls** now controlled

3. **`create_random_network.py`** - Network topology
   - Added `rng` parameter to `generate_pandapower_net()`
   - Replaced all `random.randint()` with `rng.integers()`
   - Replaced `random.choice()` with `rng.choice()`
   - Replaced `random.gauss()` with `rng.normal()`
   - Passed RNG through nested functions: `add_transformer_branch()`, `generate_network_coordinates()`, `randomize_branch_positions()`, `randomize_position()`
   - ✅ **10+ random calls** now controlled

4. **`irradiation_model/SolarIrradiationModel.py`** - Solar model
   - Added `seed` parameter to `init()` method
   - Created internal `self.rng` for cloud state transitions
   - Replaced `np.random.choice()` with `self.rng.choice()`
   - ✅ **2 random calls** now controlled

#### **Architecture Benefits:**

1. **Reproducibility**: Same master seed → identical simulation results
2. **Independence**: Changing RNG usage in one component doesn't affect others
3. **Debuggability**: Can reproduce specific scenarios for debugging
4. **Modularity**: Each function can be tested independently
5. **Maintainability**: Clear seed flow through the codebase

#### **Example Usage:**

```python
# In duilio_3ph_evaluate.py
master_seed = params.get('random_seed', 1234567890)
seed_sequence = np.random.SeedSequence(master_seed)
child_seeds = seed_sequence.spawn(10)

# Each component gets its own independent RNG
rng_network = np.random.default_rng(child_seeds[0])
rng_power = np.random.default_rng(child_seeds[1])

# Pass to functions
net, graph = generate_pandapower_net(..., rng=rng_network)
distribute_loads_to_buses(net, graph, params, db, rng=rng_power)
```

#### **Testing:**

Run `test_reproducibility.py` to verify:
- Same seed produces identical results
- Independent streams don't interfere with each other
- Network generation is reproducible

---

*Last Updated: 2025-12-11*
*Status: ✅ Core simulation components now use independent RNG streams*
*Project: pandapower_tests*
