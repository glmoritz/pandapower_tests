"""
Convergence Debugger
--------------------
Automated analysis of 3-phase power flow convergence failures.
Extracted from debug_convergence.py into a reusable module that
returns structured data (dict) suitable for JSON serialization.

Usage:
    from convergence_debugger import analyze_convergence
    report = analyze_convergence(net)          # from a pandapowerNet
    report = analyze_convergence_from_pickle("debug_net.p")  # from file
"""

import copy
import warnings
import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_downstream_buses(net, trafo_lv_bus, line_adj):
    """BFS to find all buses downstream of a trafo LV bus."""
    visited = {trafo_lv_bus}
    queue = [trafo_lv_bus]
    while queue:
        bus = queue.pop(0)
        for neighbor in line_adj.get(bus, []):
            if neighbor not in visited and neighbor != 0:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def _build_line_adjacency(net):
    """Build a bus→neighbours adjacency dict from in-service lines."""
    adj = {}
    for _, line in net.line[net.line.in_service].iterrows():
        fb, tb = int(line.from_bus), int(line.to_bus)
        adj.setdefault(fb, []).append(tb)
        adj.setdefault(tb, []).append(fb)
    return adj


# ------------------------------------------------------------------
# 1. Network overview
# ------------------------------------------------------------------

def _network_overview(net):
    return {
        "buses": len(net.bus),
        "lines": len(net.line),
        "trafos": len(net.trafo),
        "ext_grids": len(net.ext_grid),
        "loads_symmetric": len(net.load),
        "sgens_symmetric": len(net.sgen),
        "asymmetric_loads": len(net.asymmetric_load),
        "asymmetric_sgens": len(net.asymmetric_sgen),
    }


# ------------------------------------------------------------------
# 2. Parameter validation
# ------------------------------------------------------------------

def _validate_parameters(net):
    issues = []
    line_checks = []

    for col in ['r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km']:
        nans = int(net.line[col].isna().sum())
        zeros = int((net.line[col] == 0).sum())
        vmin = float(net.line[col].min()) if not net.line[col].isna().all() else None
        vmax = float(net.line[col].max()) if not net.line[col].isna().all() else None
        status = "OK" if nans == 0 else "BAD"
        if col != 'c0_nf_per_km' and zeros > 0:
            status = "WARNING"
        line_checks.append({
            "column": col, "status": status,
            "nan_count": nans, "zero_count": zeros,
            "min": vmin, "max": vmax,
        })
        if nans > 0:
            issues.append(f"line.{col} has {nans} NaN values")
        if col != 'c0_nf_per_km' and zeros > 0:
            issues.append(f"line.{col} has {zeros} zero values (may cause singular matrix)")

    trafo_checks = []
    for idx in net.trafo.index:
        t = net.trafo.loc[idx]
        info = {
            "index": int(idx),
            "name": str(t.get("name", "")),
            "vector_group": str(t.get("vector_group", None)),
            "sn_kva": float(t.sn_mva * 1000),
            "vk0_percent": _safe_float(t.get("vk0_percent")),
            "vkr0_percent": _safe_float(t.get("vkr0_percent")),
            "mag0_percent": _safe_float(t.get("mag0_percent")),
            "mag0_rx": _safe_float(t.get("mag0_rx")),
            "si0_hv_partial": _safe_float(t.get("si0_hv_partial")),
            "xn_ohm": _safe_float(t.get("xn_ohm")),
            "pfe_kw": _safe_float(t.get("pfe_kw")),
            "i0_percent": _safe_float(t.get("i0_percent")),
        }
        trafo_checks.append(info)

        if info["mag0_rx"] == 0:
            issues.append(f"Trafo {idx}: mag0_rx=0 (zero R/X ratio -> potential singular zero-seq matrix)")
        if info["pfe_kw"] == 0 and info["i0_percent"] == 0:
            issues.append(f"Trafo {idx}: pfe_kw=0 AND i0_percent=0 (no magnetizing branch)")
        if info["vk0_percent"] is None:
            issues.append(f"Trafo {idx}: vk0_percent is missing/NaN")

    ext_grid_checks = []
    for idx in net.ext_grid.index:
        eg = net.ext_grid.loc[idx]
        ext_grid_checks.append({
            "index": int(idx),
            "r0x0_max": _safe_float(eg.get("r0x0_max")),
            "x0x_max": _safe_float(eg.get("x0x_max")),
            "s_sc_max_mva": _safe_float(eg.get("s_sc_max_mva")),
        })

    return {
        "line_zero_sequence": line_checks,
        "trafo_zero_sequence": trafo_checks,
        "ext_grid_zero_sequence": ext_grid_checks,
        "issues": issues,
    }


def _safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# 3. Load distribution per transformer per phase
# ------------------------------------------------------------------

def _load_distribution(net, line_adj):
    issues = []
    trafo_loads = []

    for trafo_idx in net.trafo.index:
        lv_bus = int(net.trafo.at[trafo_idx, 'lv_bus'])
        sn_kva = float(net.trafo.at[trafo_idx, 'sn_mva'] * 1000)
        downstream = _get_downstream_buses(net, lv_bus, line_adj)

        mask = net.asymmetric_load['bus'].isin(downstream) & net.asymmetric_load['in_service']
        pa = float(net.asymmetric_load.loc[mask, 'p_a_mw'].sum() * 1000)
        pb = float(net.asymmetric_load.loc[mask, 'p_b_mw'].sum() * 1000)
        pc = float(net.asymmetric_load.loc[mask, 'p_c_mw'].sum() * 1000)
        total = pa + pb + pc
        loading = total / sn_kva * 100 if sn_kva > 0 else 0
        phases = [pa, pb, pc]
        unbal = (max(phases) - min(phases)) / (total / 3) * 100 if total > 0 else 0

        entry = {
            "trafo_index": int(trafo_idx),
            "lv_bus": lv_bus,
            "p_a_kw": round(pa, 2),
            "p_b_kw": round(pb, 2),
            "p_c_kw": round(pc, 2),
            "total_kw": round(total, 2),
            "sn_kva": round(sn_kva, 0),
            "loading_percent": round(loading, 1),
            "unbalance_percent": round(unbal, 0),
        }
        trafo_loads.append(entry)

        if unbal > 100:
            issues.append(
                f"Trafo {trafo_idx}: EXTREME phase unbalance ({unbal:.0f}%), "
                f"A={pa:.1f}kW B={pb:.1f}kW C={pc:.1f}kW"
            )

    return {"trafo_loads": trafo_loads, "issues": issues}


# ------------------------------------------------------------------
# 4. Balanced power flow test
# ------------------------------------------------------------------

def _balanced_pf_test(net):
    try:
        net_bal = copy.deepcopy(net)
        pp.runpp(net_bal)
        return {
            "converged": True,
            "vm_pu_min": round(float(net_bal.res_bus.vm_pu.min()), 4),
            "vm_pu_max": round(float(net_bal.res_bus.vm_pu.max()), 4),
            "max_line_loading_percent": round(float(net_bal.res_line.loading_percent.max()), 2),
            "max_trafo_loading_percent": round(float(net_bal.res_trafo.loading_percent.max()), 2),
        }
    except Exception as e:
        return {"converged": False, "error": str(e)}


# ------------------------------------------------------------------
# 5. 3-phase solver attempts
# ------------------------------------------------------------------

_SOLVER_CONFIGS = [
    {"label": "Default", "kwargs": {}},
    {"label": "max_iter=100", "kwargs": {"max_iteration": 100}},
    {"label": "max_iter=500", "kwargs": {"max_iteration": 500}},
    {"label": "flat_start+100iter", "kwargs": {"init": "flat", "max_iteration": 100}},
    {"label": "tol=1e-4+100iter", "kwargs": {"tolerance_mva": 1e-4, "max_iteration": 100}},
    {"label": "tol=1e-2+100iter", "kwargs": {"tolerance_mva": 1e-2, "max_iteration": 100}},
    {"label": "no_connectivity_check", "kwargs": {"check_connectivity": False, "max_iteration": 100}},
    {"label": "no_enforce_q_lims", "kwargs": {"enforce_q_lims": False, "max_iteration": 100}},
    {"label": "no_voltage_angles", "kwargs": {"calculate_voltage_angles": False, "max_iteration": 100}},
    {"label": "switch_rx=0.5", "kwargs": {"switch_rx_ratio": 0.5, "max_iteration": 100}},
]


def _solver_attempts(net):
    results = []
    any_converged = False
    for cfg in _SOLVER_CONFIGS:
        net_t = copy.deepcopy(net)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp_3ph(net_t, **cfg["kwargs"])
            has_nan = bool(
                net_t.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']].isna().any().any()
            )
            if has_nan:
                results.append({"label": cfg["label"], "status": "false_convergence"})
            else:
                min_v = float(min(
                    net_t.res_bus_3ph['vm_a_pu'].min(),
                    net_t.res_bus_3ph['vm_b_pu'].min(),
                    net_t.res_bus_3ph['vm_c_pu'].min(),
                ))
                max_v = float(max(
                    net_t.res_bus_3ph['vm_a_pu'].max(),
                    net_t.res_bus_3ph['vm_b_pu'].max(),
                    net_t.res_bus_3ph['vm_c_pu'].max(),
                ))
                results.append({
                    "label": cfg["label"], "status": "converged",
                    "vm_min": round(min_v, 4), "vm_max": round(max_v, 4),
                })
                any_converged = True
        except Exception as e:
            results.append({"label": cfg["label"], "status": "failed", "error": str(e)})

    return {"attempts": results, "any_converged": any_converged}


# ------------------------------------------------------------------
# 6. Branch isolation test
# ------------------------------------------------------------------

def _branch_isolation(net, line_adj):
    results = []
    for test_trafo in range(len(net.trafo)):
        net_t = copy.deepcopy(net)
        for t in range(len(net.trafo)):
            if t != test_trafo:
                net_t.trafo.at[t, 'in_service'] = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp_3ph(net_t, max_iteration=100)
            min_v = float(min(
                net_t.res_bus_3ph['vm_a_pu'].min(),
                net_t.res_bus_3ph['vm_b_pu'].min(),
                net_t.res_bus_3ph['vm_c_pu'].min(),
            ))
            results.append({
                "trafo_index": test_trafo, "converged": True,
                "vm_min": round(min_v, 4),
            })
        except Exception:
            lv_bus = int(net.trafo.at[test_trafo, 'lv_bus'])
            downstream = _get_downstream_buses(net, lv_bus, line_adj)
            mask = net.asymmetric_load['bus'].isin(downstream)
            total_kw = float(
                net.asymmetric_load.loc[mask, ['p_a_mw', 'p_b_mw', 'p_c_mw']].sum().sum() * 1000
            )
            results.append({
                "trafo_index": test_trafo, "converged": False,
                "total_load_kw": round(total_kw, 1),
            })
    return results


# ------------------------------------------------------------------
# 7. Voltage collapse / max loadability
# ------------------------------------------------------------------

def _voltage_collapse(net, line_adj):
    results = []
    for trafo_idx in net.trafo.index:
        lv_bus = int(net.trafo.at[trafo_idx, 'lv_bus'])
        downstream = _get_downstream_buses(net, lv_bus, line_adj)
        mask = net.asymmetric_load['bus'].isin(downstream) & net.asymmetric_load['in_service']
        original_total = float(
            net.asymmetric_load.loc[mask, ['p_a_mw', 'p_b_mw', 'p_c_mw']].sum().sum() * 1000
        )

        if original_total == 0:
            results.append({
                "trafo_index": int(trafo_idx),
                "original_total_kw": 0,
                "status": "no_load",
            })
            continue

        def make_isolated(net_orig, tidx, scale=1.0, m=None):
            net_t = copy.deepcopy(net_orig)
            for t in range(len(net_t.trafo)):
                if t != tidx:
                    net_t.trafo.at[t, 'in_service'] = False
            if m is not None and scale != 1.0:
                for col in ['p_a_mw', 'p_b_mw', 'p_c_mw']:
                    net_t.asymmetric_load.loc[m, col] = net_orig.asymmetric_load.loc[m, col] * scale
            return net_t

        # Full-load isolation test
        net_t = make_isolated(net, trafo_idx)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp_3ph(net_t, max_iteration=100)
            can_full = True
        except Exception:
            can_full = False

        if can_full:
            results.append({
                "trafo_index": int(trafo_idx),
                "original_total_kw": round(original_total, 1),
                "status": "ok_isolated",
            })
            continue

        # Binary search for max stable load
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            net_t = make_isolated(net, trafo_idx, scale=mid, m=mask)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pp.runpp_3ph(net_t, max_iteration=100)
                lo = mid
            except Exception:
                hi = mid

        max_kw = original_total * lo

        # Voltage snapshot at max stable
        bus_voltages = []
        net_t = make_isolated(net, trafo_idx, scale=lo, m=mask)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp_3ph(net_t, max_iteration=100)
            for bus_idx in sorted(downstream):
                va = float(net_t.res_bus_3ph.at[bus_idx, 'vm_a_pu'])
                vb = float(net_t.res_bus_3ph.at[bus_idx, 'vm_b_pu'])
                vc = float(net_t.res_bus_3ph.at[bus_idx, 'vm_c_pu'])
                bus_voltages.append({
                    "bus": int(bus_idx),
                    "name": str(net.bus.at[bus_idx, 'name']),
                    "vm_a_pu": round(va, 4),
                    "vm_b_pu": round(vb, 4),
                    "vm_c_pu": round(vc, 4),
                    "vm_min_pu": round(min(va, vb, vc), 4),
                })
        except Exception:
            pass

        results.append({
            "trafo_index": int(trafo_idx),
            "original_total_kw": round(original_total, 1),
            "max_stable_kw": round(max_kw, 1),
            "max_stable_fraction": round(lo, 4),
            "status": "voltage_collapse",
            "bus_voltages_at_limit": bus_voltages,
        })

    return results


# ------------------------------------------------------------------
# 8. Feeder impedance analysis
# ------------------------------------------------------------------

def _feeder_impedance(net, line_adj):
    results = []
    for trafo_idx in net.trafo.index:
        lv_bus = int(net.trafo.at[trafo_idx, 'lv_bus'])
        downstream = _get_downstream_buses(net, lv_bus, line_adj)

        def dfs_max_path(bus, visited):
            max_len = max_r = max_x = 0
            farthest_bus = bus
            for neighbor in line_adj.get(bus, []):
                if neighbor not in visited and neighbor in downstream:
                    visited.add(neighbor)
                    line_data = net.line[
                        ((net.line.from_bus == bus) & (net.line.to_bus == neighbor)) |
                        ((net.line.from_bus == neighbor) & (net.line.to_bus == bus))
                    ]
                    if not line_data.empty:
                        l = line_data.iloc[0]
                        seg_len = l.length_km
                        seg_r = l.length_km * l.r_ohm_per_km
                        seg_x = l.length_km * l.x_ohm_per_km
                        c_len, c_r, c_x, c_bus = dfs_max_path(neighbor, visited)
                        if seg_len + c_len > max_len:
                            max_len = seg_len + c_len
                            max_r = seg_r + c_r
                            max_x = seg_x + c_x
                            farthest_bus = c_bus
                    visited.discard(neighbor)
            return max_len, max_r, max_x, farthest_bus

        max_len, max_r, max_x, farthest = dfs_max_path(lv_bus, {lv_bus})
        V_phase = float(net.bus.at[lv_bus, 'vn_kv']) * 1000 / np.sqrt(3)
        Z = np.sqrt(max_r**2 + max_x**2)
        P_max_est = V_phase**2 / (4 * Z) / 1000 if Z > 0 else float('inf')

        results.append({
            "trafo_index": int(trafo_idx),
            "farthest_bus": int(farthest),
            "length_m": round(max_len * 1000, 0),
            "r_ohm": round(max_r, 4),
            "x_ohm": round(max_x, 4),
            "z_ohm": round(Z, 4),
            "v_phase_v": round(V_phase, 1),
            "theoretical_max_1ph_kw": round(P_max_est, 1),
        })

    return results


# ------------------------------------------------------------------
# 9. Root-cause summary generator
# ------------------------------------------------------------------

def _generate_summary(report):
    """Build a human-readable root-cause summary from the structured report."""
    lines = []
    all_issues = report.get("parameter_issues", []) + report.get("load_distribution_issues", [])

    # Identify voltage-collapse branches
    vc_branches = [
        v for v in report.get("voltage_collapse", [])
        if v.get("status") == "voltage_collapse"
    ]
    if vc_branches:
        lines.append("VOLTAGE COLLAPSE detected on the following transformer branches:")
        for vc in vc_branches:
            lines.append(
                f"  - Trafo {vc['trafo_index']}: load {vc['original_total_kw']:.1f} kW "
                f"exceeds max stable {vc['max_stable_kw']:.1f} kW "
                f"({vc['max_stable_fraction']:.1%} of original)"
            )
            low_buses = [
                b for b in vc.get("bus_voltages_at_limit", [])
                if b["vm_min_pu"] < 0.85
            ]
            if low_buses:
                lines.append("    Low-voltage buses at stability limit:")
                for b in low_buses:
                    lines.append(
                        f"      Bus {b['bus']} ({b['name']}): "
                        f"Va={b['vm_a_pu']:.4f} Vb={b['vm_b_pu']:.4f} Vc={b['vm_c_pu']:.4f}"
                    )

    # Unbalance issues
    unbal = [
        t for t in report.get("load_distribution", {}).get("trafo_loads", [])
        if t["unbalance_percent"] > 50
    ]
    if unbal:
        lines.append("\nHIGH PHASE UNBALANCE:")
        for t in unbal:
            lines.append(
                f"  - Trafo {t['trafo_index']}: unbalance {t['unbalance_percent']:.0f}%, "
                f"A={t['p_a_kw']:.1f}kW B={t['p_b_kw']:.1f}kW C={t['p_c_kw']:.1f}kW"
            )

    # Feeder issues
    long_feeders = [
        f for f in report.get("feeder_impedance", [])
        if f["theoretical_max_1ph_kw"] < 30
    ]
    if long_feeders:
        lines.append("\nLONG/HIGH-IMPEDANCE FEEDERS:")
        for f in long_feeders:
            lines.append(
                f"  - Trafo {f['trafo_index']}: feeder to bus {f['farthest_bus']} "
                f"is {f['length_m']:.0f}m, theoretical 1-phase max ~{f['theoretical_max_1ph_kw']:.1f} kW"
            )

    if all_issues:
        lines.append(f"\nADDITIONAL ISSUES ({len(all_issues)}):")
        for issue in all_issues:
            lines.append(f"  - {issue}")

    if not lines:
        lines.append("No specific root cause identified from automated analysis.")

    return "\n".join(lines)


# ==================================================================
# Main entry points
# ==================================================================

def analyze_convergence(net):
    """Run all convergence diagnostics on a pandapowerNet.

    Returns a dict with structured debug information, suitable for
    JSON serialization and inclusion in simulation result files.
    """
    line_adj = _build_line_adjacency(net)

    overview = _network_overview(net)
    params = _validate_parameters(net)
    load_dist = _load_distribution(net, line_adj)
    balanced = _balanced_pf_test(net)
    solver = _solver_attempts(net)
    isolation = _branch_isolation(net, line_adj)
    collapse = _voltage_collapse(net, line_adj)
    feeder = _feeder_impedance(net, line_adj)

    report = {
        "network_overview": overview,
        "parameter_validation": params,
        "parameter_issues": params["issues"],
        "load_distribution": load_dist,
        "load_distribution_issues": load_dist["issues"],
        "balanced_pf_test": balanced,
        "solver_attempts": solver,
        "branch_isolation": isolation,
        "voltage_collapse": collapse,
        "feeder_impedance": feeder,
    }

    report["summary"] = _generate_summary(report)
    report["total_issues"] = len(report["parameter_issues"]) + len(report["load_distribution_issues"])

    return report


def analyze_convergence_from_pickle(pickle_path):
    """Load a pickled pandapowerNet and run convergence analysis."""
    net = pp.from_pickle(pickle_path)
    return analyze_convergence(net)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python convergence_debugger.py <pickle_file> [output.json]")
        sys.exit(1)

    pickle_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Analyzing {pickle_file} ...")
    report = analyze_convergence_from_pickle(pickle_file)

    print(f"\n{'='*80}")
    print(report["summary"])
    print(f"{'='*80}")
    print(f"Total issues: {report['total_issues']}")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Full report saved to {output_file}")
    else:
        print(json.dumps(report, indent=2, default=str))
