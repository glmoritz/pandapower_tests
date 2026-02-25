# %%
# SELECT variable_id, simulation_id, element_type, element_index, variable_name, unit
# FROM building_power.simulation_variable
# where simulation_id = $1
# ;

# SELECT sv.variable_name ,sv.element_type, sv.element_index , ts.variable_id, bucket, value*1000 as power_kw 
# FROM building_power.simulation_timeseries ts
# inner join building_power.simulation_variable sv on ts.variable_id = sv.variable_id
# where ts.variable_id = $1 order by bucket;

# SELECT DISTINCT time_bucket('1 day', sample_time) AS day
#         FROM building_power.building_power
#         WHERE bldg_id = :bldg_id
#         ORDER BY day
        
#         SELECT time_bucket(INTERVAL '{interval_str}', sample_time) AS bucket,
#                             SUM(electricity_total_energy_consumption) / 0.25 AS \"Power[kW]\"
#                             FROM building_power.building_power
#                             WHERE bldg_id IN ({bldg_ids})
#                                 AND sample_time >= '{sim_start_dt.strftime('%Y-%m-%d %H:%M:%S')}'
#                                 AND sample_time <= '{sim_end_dt.strftime('%Y-%m-%d %H:%M:%S')}'
#                                 AND electricity_total_energy_consumption IS NOT NULL
#                             GROUP BY bucket
#                             ORDER BY bucket 


import pandapower as pp
import numpy as np
import pandas as pd
import copy
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

PICKLE_FILE = "debug_net.p"

# ============================================================
# 1. LOAD NETWORK
# ============================================================
net = pp.from_pickle(PICKLE_FILE)
print("=" * 80)
print(f"3-PHASE POWER FLOW CONVERGENCE DEBUGGER")
print(f"pandapower version: {pp.__version__}")
print("=" * 80)
print(f"\nNetwork: {len(net.bus)} buses, {len(net.line)} lines, "
      f"{len(net.trafo)} trafos, {len(net.ext_grid)} ext_grid(s)")
print(f"  Loads (symmetric): {len(net.load)}")
print(f"  Sgens (symmetric): {len(net.sgen)}")
print(f"  Asymmetric loads:  {len(net.asymmetric_load)}")
print(f"  Asymmetric sgens:  {len(net.asymmetric_sgen)}")

# ============================================================
# 2. PARAMETER VALIDATION - Zero-sequence data checks
# ============================================================
print("\n" + "=" * 80)
print("PARAMETER VALIDATION")
print("=" * 80)

# %%
print(net)

#%%
issues = []

# --- Line zero-sequence ---
for col in ['r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km']:
    nans = net.line[col].isna().sum()
    zeros = (net.line[col] == 0).sum()
    vmin, vmax = net.line[col].min(), net.line[col].max()
    status = "OK" if nans == 0 else "BAD"
    if col != 'c0_nf_per_km' and zeros > 0:
        status = "WARNING"
    print(f"  line.{col}: {status} ({nans} NaN, {zeros} zeros, range=[{vmin:.4f}, {vmax:.4f}])")
    if nans > 0:
        issues.append(f"line.{col} has {nans} NaN values")
    if col != 'c0_nf_per_km' and zeros > 0:
        issues.append(f"line.{col} has {zeros} zero values (may cause singular matrix)")

# --- Transformer zero-sequence ---
print("\n  Transformer zero-sequence parameters:")
for idx in net.trafo.index:
    t = net.trafo.loc[idx]
    vk0 = t.get('vk0_percent', None)
    vkr0 = t.get('vkr0_percent', None)
    mag0 = t.get('mag0_percent', None)
    mag0_rx = t.get('mag0_rx', None)
    si0 = t.get('si0_hv_partial', None)
    vg = t.get('vector_group', None)
    xn = t.get('xn_ohm', None)
    pfe = t.get('pfe_kw', None)
    i0 = t.get('i0_percent', None)

    print(f"    Trafo {idx} [{t['name']}]: vg={vg}, sn={t.sn_mva*1000:.0f}kVA, "
          f"vk0={vk0}%, vkr0={vkr0}%, mag0={mag0}%, mag0_rx={mag0_rx}, "
          f"si0_hv={si0}, xn={xn}, pfe={pfe}kW, i0={i0}%")

    if mag0_rx == 0:
        issues.append(f"Trafo {idx}: mag0_rx=0 (zero R/X ratio -> potential singular zero-seq matrix)")
    if pfe == 0 and i0 == 0:
        issues.append(f"Trafo {idx}: pfe_kw=0 AND i0_percent=0 (no magnetizing branch)")
    if vk0 is None or pd.isna(vk0):
        issues.append(f"Trafo {idx}: vk0_percent is missing/NaN")

# --- Ext grid zero-sequence ---
print("\n  External grid zero-sequence:")
for idx in net.ext_grid.index:
    eg = net.ext_grid.loc[idx]
    print(f"    ExtGrid {idx}: r0x0_max={eg.get('r0x0_max',None)}, "
          f"x0x_max={eg.get('x0x_max',None)}, "
          f"s_sc_max={eg.get('s_sc_max_mva',None)} MVA")

# ============================================================
# 3. TOTAL LOAD PER TRANSFORMER PER PHASE
# ============================================================
print("\n" + "=" * 80)
print("LOAD DISTRIBUTION PER TRANSFORMER PER PHASE")
print("=" * 80)

# Build bus-to-trafo mapping using network topology
def get_downstream_buses(net, trafo_lv_bus, all_line_connections):
    """BFS to find all buses downstream of a trafo LV bus."""
    visited = {trafo_lv_bus}
    queue = [trafo_lv_bus]
    while queue:
        bus = queue.pop(0)
        for neighbor in all_line_connections.get(bus, []):
            if neighbor not in visited and neighbor != 0:  # don't cross to MV bus
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

# Build adjacency from lines
line_adj = {}
for _, line in net.line[net.line.in_service].iterrows():
    fb, tb = int(line.from_bus), int(line.to_bus)
    line_adj.setdefault(fb, []).append(tb)
    line_adj.setdefault(tb, []).append(fb)

print(f"\n{'Trafo':>6} | {'LV Bus':>6} | {'P_a (kW)':>10} | {'P_b (kW)':>10} | "
      f"{'P_c (kW)':>10} | {'Total (kW)':>10} | {'Sn (kVA)':>8} | {'Load %':>7} | {'Unbalance':>10}")
print("-" * 100)

for trafo_idx in net.trafo.index:
    lv_bus = int(net.trafo.at[trafo_idx, 'lv_bus'])
    sn_kva = net.trafo.at[trafo_idx, 'sn_mva'] * 1000
    downstream = get_downstream_buses(net, lv_bus, line_adj)

    # Sum asymmetric loads on downstream buses
    mask = net.asymmetric_load['bus'].isin(downstream) & net.asymmetric_load['in_service']
    pa = net.asymmetric_load.loc[mask, 'p_a_mw'].sum() * 1000
    pb = net.asymmetric_load.loc[mask, 'p_b_mw'].sum() * 1000
    pc = net.asymmetric_load.loc[mask, 'p_c_mw'].sum() * 1000
    total = pa + pb + pc
    loading = total / sn_kva * 100
    phases = [pa, pb, pc]
    max_phase = max(phases)
    min_phase = min(phases)
    unbal = (max_phase - min_phase) / (total / 3) * 100 if total > 0 else 0

    print(f"{trafo_idx:>6} | {lv_bus:>6} | {pa:>10.2f} | {pb:>10.2f} | "
          f"{pc:>10.2f} | {total:>10.2f} | {sn_kva:>8.0f} | {loading:>6.1f}% | {unbal:>9.0f}%")

    if unbal > 100:
        issues.append(f"Trafo {trafo_idx}: EXTREME phase unbalance ({unbal:.0f}%), "
                      f"A={pa:.1f}kW B={pb:.1f}kW C={pc:.1f}kW")

# Per-bus detail
print("\n  Per-bus asymmetric load detail (non-zero only):")
for bus_idx in net.asymmetric_load['bus'].unique():
    al = net.asymmetric_load[net.asymmetric_load['bus'] == bus_idx]
    pa = al['p_a_mw'].sum() * 1000
    pb = al['p_b_mw'].sum() * 1000
    pc = al['p_c_mw'].sum() * 1000
    if pa + pb + pc > 0:
        bus_name = net.bus.at[bus_idx, 'name'] if bus_idx in net.bus.index else '?'
        print(f"    Bus {bus_idx:>3} ({bus_name}): "
              f"A={pa:>7.2f}kW  B={pb:>7.2f}kW  C={pc:>7.2f}kW  "
              f"Total={pa+pb+pc:.2f}kW")

# ============================================================
# 4. BALANCED POWER FLOW TEST (sanity check)
# ============================================================
print("\n" + "=" * 80)
print("BALANCED (SINGLE-PHASE) POWER FLOW TEST")
print("=" * 80)
try:
    net_bal = copy.deepcopy(net)
    pp.runpp(net_bal)
    print(f"  Result: CONVERGED")
    print(f"  Voltage range: {net_bal.res_bus.vm_pu.min():.4f} - {net_bal.res_bus.vm_pu.max():.4f} pu")
    print(f"  Max line loading:  {net_bal.res_line.loading_percent.max():.2f}%")
    print(f"  Max trafo loading: {net_bal.res_trafo.loading_percent.max():.2f}%")
except Exception as e:
    print(f"  Result: FAILED - {e}")
    issues.append(f"Even balanced power flow fails: {e}")

# ============================================================
# 5. TRY MULTIPLE 3-PHASE SOLVER CONFIGURATIONS
# ============================================================
print("\n" + "=" * 80)
print("3-PHASE POWER FLOW - MULTIPLE SOLVER ATTEMPTS")
print("=" * 80)

test_configs = [
    {"label": "Default (auto init, tol=1e-8, max_iter=30)",
     "kwargs": {}},
    {"label": "More iterations (max_iter=100)",
     "kwargs": {"max_iteration": 100}},
    {"label": "Many iterations (max_iter=500)",
     "kwargs": {"max_iteration": 500}},
    {"label": "Flat start + 100 iter",
     "kwargs": {"init": "flat", "max_iteration": 100}},
    {"label": "Relaxed tolerance (1e-4)",
     "kwargs": {"tolerance_mva": 1e-4, "max_iteration": 100}},
    {"label": "Very relaxed tolerance (1e-2)",
     "kwargs": {"tolerance_mva": 1e-2, "max_iteration": 100}},
    {"label": "Disable connectivity check",
     "kwargs": {"check_connectivity": False, "max_iteration": 100}},
    {"label": "No enforce_q_lims + 100 iter",
     "kwargs": {"enforce_q_lims": False, "max_iteration": 100}},
    {"label": "No voltage angle calculation",
     "kwargs": {"calculate_voltage_angles": False, "max_iteration": 100}},
    {"label": "switch_rx_ratio=0.5",
     "kwargs": {"switch_rx_ratio": 0.5, "max_iteration": 100}},
]

converged_any = False
for cfg in test_configs:
    net_t = copy.deepcopy(net)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp_3ph(net_t, **cfg["kwargs"])
        # Check for NaN in results (false convergence)
        has_nan = net_t.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']].isna().any().any()
        if has_nan:
            print(f"  [WARN  ] {cfg['label']}: 'converged' but results contain NaN (false convergence)")
        else:
            min_v = min(net_t.res_bus_3ph['vm_a_pu'].min(),
                        net_t.res_bus_3ph['vm_b_pu'].min(),
                        net_t.res_bus_3ph['vm_c_pu'].min())
            max_v = max(net_t.res_bus_3ph['vm_a_pu'].max(),
                        net_t.res_bus_3ph['vm_b_pu'].max(),
                        net_t.res_bus_3ph['vm_c_pu'].max())
            print(f"  [  OK  ] {cfg['label']}: CONVERGED (V range: {min_v:.4f}-{max_v:.4f} pu)")
            converged_any = True
    except Exception as e:
        print(f"  [FAILED] {cfg['label']}: {e}")

# ============================================================
# 6. ISOLATE PROBLEM: TEST EACH TRAFO BRANCH ALONE
# ============================================================
print("\n" + "=" * 80)
print("BRANCH ISOLATION TEST (one trafo at a time)")
print("=" * 80)

for test_trafo in range(len(net.trafo)):
    net_t = copy.deepcopy(net)
    for t in range(len(net.trafo)):
        if t != test_trafo:
            net_t.trafo.at[t, 'in_service'] = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp_3ph(net_t, max_iteration=100)
        min_v = min(net_t.res_bus_3ph['vm_a_pu'].min(),
                    net_t.res_bus_3ph['vm_b_pu'].min(),
                    net_t.res_bus_3ph['vm_c_pu'].min())
        print(f"  Trafo {test_trafo} alone: CONVERGED (min V={min_v:.4f} pu)")
    except:
        lv_bus = int(net.trafo.at[test_trafo, 'lv_bus'])
        downstream = get_downstream_buses(net, lv_bus, line_adj)
        mask = net.asymmetric_load['bus'].isin(downstream)
        total_kw = (net.asymmetric_load.loc[mask, ['p_a_mw', 'p_b_mw', 'p_c_mw']].sum().sum()) * 1000
        print(f"  Trafo {test_trafo} alone: FAILED (total load: {total_kw:.1f} kW)")
        issues.append(f"Trafo {test_trafo} branch diverges even in isolation (load={total_kw:.1f}kW)")

# ============================================================
# 7. VOLTAGE COLLAPSE ANALYSIS - find max loadability per branch (isolated)
# ============================================================
print("\n" + "=" * 80)
print("VOLTAGE COLLAPSE / MAX LOADABILITY ANALYSIS (each branch isolated)")
print("=" * 80)

for trafo_idx in net.trafo.index:
    lv_bus = int(net.trafo.at[trafo_idx, 'lv_bus'])
    downstream = get_downstream_buses(net, lv_bus, line_adj)
    mask = net.asymmetric_load['bus'].isin(downstream) & net.asymmetric_load['in_service']
    original_total = (net.asymmetric_load.loc[mask, ['p_a_mw', 'p_b_mw', 'p_c_mw']].sum().sum()) * 1000

    if original_total == 0:
        print(f"\n  Trafo {trafo_idx}: no load, skipping")
        continue

    # Create isolated branch: disable all other trafos
    def make_isolated(net_orig, trafo_idx, scale=1.0, mask=None):
        net_t = copy.deepcopy(net_orig)
        for t in range(len(net_t.trafo)):
            if t != trafo_idx:
                net_t.trafo.at[t, 'in_service'] = False
        if mask is not None and scale != 1.0:
            for col in ['p_a_mw', 'p_b_mw', 'p_c_mw']:
                net_t.asymmetric_load.loc[mask, col] = net_orig.asymmetric_load.loc[mask, col] * scale
        return net_t

    # Quick check if full load works in isolation
    net_t = make_isolated(net, trafo_idx)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp_3ph(net_t, max_iteration=100)
        can_full = True
    except:
        can_full = False

    if not can_full:
        lo_scale, hi_scale = 0.0, 1.0
        for _ in range(12):
            mid = (lo_scale + hi_scale) / 2
            net_t = make_isolated(net, trafo_idx, scale=mid, mask=mask)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pp.runpp_3ph(net_t, max_iteration=100)
                lo_scale = mid
            except:
                hi_scale = mid

        max_kw = original_total * lo_scale
        print(f"\n  Trafo {trafo_idx}: original={original_total:.1f}kW, "
              f"max stable={max_kw:.1f}kW ({lo_scale:.1%} of original)")
        if lo_scale < 1.0:
            issues.append(f"Trafo {trafo_idx}: load ({original_total:.1f}kW) exceeds max stable "
                          f"({max_kw:.1f}kW, {lo_scale:.1%}) in isolated branch")

        # Show voltages at the max stable point
        net_t = make_isolated(net, trafo_idx, scale=lo_scale, mask=mask)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp_3ph(net_t, max_iteration=100)
            print(f"  Voltages at max stable load ({lo_scale:.1%}):")
            for bus_idx in sorted(downstream):
                va = net_t.res_bus_3ph.at[bus_idx, 'vm_a_pu']
                vb = net_t.res_bus_3ph.at[bus_idx, 'vm_b_pu']
                vc = net_t.res_bus_3ph.at[bus_idx, 'vm_c_pu']
                min_v = min(va, vb, vc)
                flag = " <<<< LOW" if min_v < 0.7 else (" << LOW" if min_v < 0.85 else "")
                bus_name = net.bus.at[bus_idx, 'name']
                print(f"    Bus {bus_idx:>3} Va={va:.4f} Vb={vb:.4f} Vc={vc:.4f}{flag}  ({bus_name})")
        except:
            pass
    else:
        print(f"\n  Trafo {trafo_idx}: original={original_total:.1f}kW -> converges OK in isolation")

# ============================================================
# 8. FEEDER LENGTH & IMPEDANCE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("FEEDER IMPEDANCE ANALYSIS (longest path per trafo)")
print("=" * 80)

for trafo_idx in net.trafo.index:
    lv_bus = int(net.trafo.at[trafo_idx, 'lv_bus'])
    downstream = get_downstream_buses(net, lv_bus, line_adj)

    # Find the longest path (max total impedance) from trafo LV bus
    # Simple DFS approach for radial network
    def dfs_max_path(bus, visited, net, line_adj):
        max_len = 0
        max_r = 0
        max_x = 0
        farthest_bus = bus
        for neighbor in line_adj.get(bus, []):
            if neighbor not in visited and neighbor in downstream:
                visited.add(neighbor)
                line_data = net.line[((net.line.from_bus == bus) & (net.line.to_bus == neighbor)) |
                                     ((net.line.from_bus == neighbor) & (net.line.to_bus == bus))]
                if not line_data.empty:
                    l = line_data.iloc[0]
                    seg_len = l.length_km
                    seg_r = l.length_km * l.r_ohm_per_km
                    seg_x = l.length_km * l.x_ohm_per_km
                    child_len, child_r, child_x, child_bus = dfs_max_path(neighbor, visited, net, line_adj)
                    if seg_len + child_len > max_len:
                        max_len = seg_len + child_len
                        max_r = seg_r + child_r
                        max_x = seg_x + child_x
                        farthest_bus = child_bus
                visited.discard(neighbor)
        return max_len, max_r, max_x, farthest_bus

    max_len, max_r, max_x, farthest = dfs_max_path(lv_bus, {lv_bus}, net, line_adj)
    V_phase = net.bus.at[lv_bus, 'vn_kv'] * 1000 / np.sqrt(3)
    # Max power before voltage collapse (rough estimate: P_max ≈ V²/Z for single phase)
    Z = np.sqrt(max_r**2 + max_x**2)
    P_max_est = V_phase**2 / (4 * Z) / 1000 if Z > 0 else float('inf')  # kW, theoretical max

    print(f"\n  Trafo {trafo_idx}: longest feeder to bus {farthest}")
    print(f"    Length: {max_len*1000:.0f}m, R={max_r:.4f}Ω, X={max_x:.4f}Ω, |Z|={Z:.4f}Ω")
    print(f"    V_phase={V_phase:.1f}V")
    print(f"    Theoretical 1-phase max power (P=V²/4Z): ~{P_max_est:.1f} kW")
    if P_max_est < 30:
        issues.append(f"Trafo {trafo_idx}: theoretical max single-phase load to bus {farthest} "
                      f"is only ~{P_max_est:.1f}kW (feeder {max_len*1000:.0f}m)")

# ============================================================
# 9. SUMMARY OF ISSUES
# ============================================================
print("\n" + "=" * 80)
print(f"SUMMARY: {len(issues)} ISSUE(S) FOUND")
print("=" * 80)
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("  No issues detected.")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)
print("""
The 3-phase power flow diverges due to VOLTAGE COLLAPSE on a single phase.

MECHANISM:
  - Bus 15 has a 22.53 kW load concentrated entirely on PHASE C
  - This bus is at the end of a ~694m radial feeder (Trafo 2 branch)
  - The single-phase current (~97.6A at 231V) causes extreme voltage drop
    through the feeder impedance (R≈0.34Ω, X≈0.20Ω per phase)
  - At the tipping point (~21.7 kW), phase C voltage at bus 15 drops to
    ~0.53 pu (122V) while phases A & B rise above 1.0 pu (neutral shift)
  - Beyond ~21.7 kW, no stable operating point exists -> Newton-Raphson diverges

CONTRIBUTING FACTORS:
  1. EXTREME UNBALANCE: 22.53 kW on a single phase (100% unbalance)
  2. LONG FEEDER: 694m from transformer to the load bus
  3. LOAD EXCEEDS STABILITY LIMIT: actual load (22.53 kW) > max stable (~21.7 kW)
  4. The 'dgstrf info 1' messages indicate the admittance matrix becomes
     singular during iterations as voltages approach zero on the loaded phase

POSSIBLE FIXES (in order of likelihood):
  1. Check if 22.53 kW on one phase is realistic for a residential bus
     - This could be a unit error (kW vs MW) or aggregation issue
     - Typical single-phase residential load: 2-5 kW
  2. Distribute the load across multiple phases (use 3-phase connection)
  3. Reduce feeder length or use lower-impedance cables
  4. Add voltage regulation or capacitor banks
  5. Check the load generation code for unit conversion errors
""")