#!/usr/bin/env python3
"""
Analyze failed simulation results, diagnose convergence issues, and generate
new JSON queue files with adjusted parameters to fix them.

For each failed simulation, reads the convergence_debug data, classifies the
root cause, and generates a new config with appropriate load reductions.

Usage:
    python fix_and_requeue.py
    python fix_and_requeue.py --results-dir results/duilio_debug/03_results
    python fix_and_requeue.py --dry-run   # preview fixes without writing files
"""

import argparse
import json
import os
import re
import sys


def find_result_files(dirs):
    """Find all result JSON files in the given directories (recursive)."""
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, filenames in os.walk(d):
            for fn in filenames:
                if fn.endswith(".json"):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def load_failed_results(dirs):
    """Load all result JSONs that indicate convergence failures."""
    failed = []
    for fp in find_result_files(dirs):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            status = data.get("status", "")
            if status in ("convergence_error", "error") and not data.get("converged", False):
                data["_source_file"] = fp
                failed.append(data)
        except (json.JSONDecodeError, OSError):
            pass
    return failed


def diagnose_failure(result):
    """Diagnose the root cause and return a fix strategy.

    Returns a dict with:
        root_cause: str,
        fix_strategy: str,
        load_reduction_factor: float (1.0 = no change, 0.5 = halve loads),
        force_balanced: bool,
        reduce_fork_length: bool,
        notes: str
    """
    debug = result.get("convergence_debug")
    summary = result.get("convergence_debug_summary", "")

    diagnosis = {
        "root_cause": "unknown",
        "fix_strategy": "reduce_load",
        "load_reduction_factor": 0.8,
        "force_balanced": True,
        "reduce_fork_length": False,
        "notes": "",
    }

    if not debug:
        diagnosis["notes"] = "No convergence debug data available; applying default load reduction"
        return diagnosis

    # Check voltage collapse
    collapse_entries = [
        v for v in debug.get("voltage_collapse", [])
        if v.get("status") == "voltage_collapse"
    ]
    if collapse_entries:
        # Find worst collapse fraction
        fractions = [v.get("max_stable_fraction", 1.0) for v in collapse_entries]
        worst_fraction = min(fractions)

        diagnosis["root_cause"] = "voltage_collapse"
        # Set load to 90% of the max stable fraction (safety margin)
        diagnosis["load_reduction_factor"] = round(worst_fraction * 0.9, 2)
        diagnosis["notes"] = (
            f"Voltage collapse detected. Worst trafo stable at {worst_fraction:.1%} of original load. "
            f"Reducing load to {diagnosis['load_reduction_factor']:.0%}."
        )
        return diagnosis

    # Check phase unbalance
    load_dist = debug.get("load_distribution", {})
    high_unbalance_trafos = [
        t for t in load_dist.get("trafo_loads", [])
        if t.get("unbalance_percent", 0) > 100
    ]
    if high_unbalance_trafos:
        worst_unbalance = max(t.get("unbalance_percent", 0) for t in high_unbalance_trafos)
        diagnosis["root_cause"] = "phase_unbalance"
        diagnosis["fix_strategy"] = "force_balance"
        diagnosis["force_balanced"] = True
        diagnosis["load_reduction_factor"] = 0.9
        diagnosis["notes"] = (
            f"Severe phase unbalance ({worst_unbalance:.0f}%). "
            f"Forcing balanced loading and reducing load to 90%."
        )
        return diagnosis

    # Check feeder impedance
    feeder_issues = [
        f for f in debug.get("feeder_impedance", [])
        if f.get("theoretical_max_1ph_kw", float("inf")) < 30
    ]
    if feeder_issues:
        worst_max = min(f.get("theoretical_max_1ph_kw", float("inf")) for f in feeder_issues)
        diagnosis["root_cause"] = "feeder_impedance"
        diagnosis["fix_strategy"] = "reduce_fork"
        diagnosis["reduce_fork_length"] = True
        diagnosis["load_reduction_factor"] = 0.7
        diagnosis["notes"] = (
            f"High feeder impedance (worst theoretical max: {worst_max:.1f} kW). "
            f"Reducing fork length and load to 70%."
        )
        return diagnosis

    # Check parameter issues
    param_issues = debug.get("parameter_issues", [])
    if param_issues:
        diagnosis["root_cause"] = "parameter_issues"
        diagnosis["fix_strategy"] = "regenerate_network"
        diagnosis["load_reduction_factor"] = 1.0
        diagnosis["notes"] = (
            f"Network parameter issues detected ({len(param_issues)} issues). "
            f"Regenerating network with fresh parameters."
        )
        return diagnosis

    # Check if all solvers failed
    solver = debug.get("solver_attempts", [])
    any_converged = any(s.get("status") == "converged" for s in solver)
    if solver and not any_converged:
        diagnosis["root_cause"] = "all_solvers_failed"
        diagnosis["load_reduction_factor"] = 0.6
        diagnosis["notes"] = (
            "All solver configurations failed. "
            "Aggressive load reduction to 60%."
        )
        return diagnosis

    # Fallback: generic load reduction
    diagnosis["notes"] = "Could not classify root cause; applying default load reduction to 80%."
    return diagnosis


def parse_name(name):
    """Parse Solar-X-Storage-Y-seedZ[-fixN] from filename."""
    m = re.match(r"Solar-(\d+)-Storage-(\d+)-seed(\d+)(?:-fix(\d+))?", name)
    if m:
        return {
            "solar_kw": int(m.group(1)),
            "storage_kwh": int(m.group(2)),
            "seed": int(m.group(3)),
            "fix_attempt": int(m.group(4)) if m.group(4) else 0,
        }
    return None


def generate_fixed_config(result, diagnosis, original_params_dir):
    """Generate a new JSON config with fixes applied based on diagnosis."""
    name = result.get("name", result.get("base_name", ""))
    parsed = parse_name(name)
    if not parsed:
        return None, None

    next_fix = parsed["fix_attempt"] + 1
    new_base = f"Solar-{parsed['solar_kw']}-Storage-{parsed['storage_kwh']}-seed{parsed['seed']}-fix{next_fix}"
    new_name = f"{new_base}.json"

    # Always reconstruct from standardized params (don't reuse old inconsistent configs)
    config = _reconstruct_config(parsed, result)

    # Apply fixes
    config["name"] = new_name
    config["output_file"] = f"{new_base}.csv"
    config["random_seed"] = parsed["seed"]

    # Apply load reduction
    factor = diagnosis["load_reduction_factor"]
    orig_load = config.get("load_power_range_per_branch_kW", [5, 5])
    new_load = [round(v * factor, 1) for v in orig_load]
    # Ensure min load is at least 1 kW
    new_load = [max(1.0, v) for v in new_load]
    config["load_power_range_per_branch_kW"] = new_load

    # Force balanced loading if recommended
    if diagnosis["force_balanced"]:
        config["balance_phase_loading"] = True

    # Reduce fork length if recommended
    if diagnosis["reduce_fork_length"]:
        orig_fork = config.get("fork_length_range", [30, 80])
        config["fork_length_range"] = [
            max(10, int(orig_fork[0] * 0.5)),
            max(20, int(orig_fork[1] * 0.5)),
        ]

    # Force regeneration of network and profile
    config["use_saved_network_if_exists"] = False
    config["use_saved_power_profile_if_exists"] = False

    # Update network_id and power_profile_id to avoid collisions with original
    config["network_id"] = f"{config.get('network_id', 'medium_test_network')}_fix{next_fix}"
    config["power_profile_id"] = f"{config.get('power_profile_id', '')}_fix{next_fix}"

    return new_name, config


def _reconstruct_config(parsed, result):
    """Reconstruct a config dict from parsed name and result data."""
    from generate_queue_files import SOLAR_CONFIGS, STORAGE_CONFIGS, FIXED_PARAMS, STANDARDIZED_PARAMS

    solar_range = SOLAR_CONFIGS.get(parsed["solar_kw"], [parsed["solar_kw"], parsed["solar_kw"]])
    storage_range = STORAGE_CONFIGS.get(parsed["storage_kwh"], [parsed["storage_kwh"], parsed["storage_kwh"]])

    config = {
        "random_seed": parsed["seed"],
        "power_profile_id": f"Solar-{parsed['solar_kw']}-Storage-{parsed['storage_kwh']}",
        "solar_power_range_per_branch_kW": list(solar_range),
        "storage_capacity_range_per_branch_kWh": list(storage_range),
    }
    config.update(FIXED_PARAMS)
    config.update(STANDARDIZED_PARAMS)
    return config


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(script_dir, "results", "duilio_debug", "03_results")
    default_queue = os.path.join(script_dir, "results", "duilio_debug", "01_params")
    default_original = os.path.join(script_dir, "results", "duilio_debug", "05_queue")

    parser = argparse.ArgumentParser(description="Fix and re-queue failed simulations")
    parser.add_argument(
        "--results-dir",
        default=default_results,
        help=f"Directory containing result JSON files (default: {default_results})",
    )
    parser.add_argument(
        "--queue-dir",
        default=default_queue,
        help=f"Directory to write fixed queue files (default: {default_queue})",
    )
    parser.add_argument(
        "--original-params-dir",
        default=default_original,
        help=f"Directory with original param files for reference (default: {default_original})",
    )
    parser.add_argument(
        "--scan-finished",
        action="store_true",
        help="Also scan 04_finished/ subdirectories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview fixes without writing files",
    )
    args = parser.parse_args()

    # Collect directories to scan
    scan_dirs = [args.results_dir]
    if args.scan_finished:
        base = os.path.dirname(args.results_dir)
        scan_dirs.append(os.path.join(base, "04_finished"))

    # Load failed results
    failed = load_failed_results(scan_dirs)
    if not failed:
        print("No failed simulation results found.")
        return

    print(f"Found {len(failed)} failed simulation(s)")
    print("=" * 80)

    # Diagnose and generate fixes
    fixes = []
    for result in failed:
        name = result.get("name", result.get("base_name", "unknown"))
        diagnosis = diagnose_failure(result)

        print(f"\n{name}")
        print(f"  Root cause: {diagnosis['root_cause']}")
        print(f"  Strategy:   {diagnosis['fix_strategy']}")
        print(f"  Load factor: {diagnosis['load_reduction_factor']:.0%}")
        print(f"  Notes: {diagnosis['notes']}")

        new_name, config = generate_fixed_config(result, diagnosis, args.original_params_dir)
        if not new_name:
            print(f"  ERROR: Could not parse name '{name}', skipping")
            continue

        orig_load = result.get("load_power_range_per_branch_kW",
                               result.get("solar_power_range", [5, 5]))
        print(f"  Fix: {name} -> {new_name}")
        print(f"  Load: {orig_load} -> {config['load_power_range_per_branch_kW']}")

        fixes.append((new_name, config, diagnosis))

    print(f"\n{'=' * 80}")
    print(f"Total fixes to apply: {len(fixes)}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    if not fixes:
        return

    # Write fixed configs to queue
    os.makedirs(args.queue_dir, exist_ok=True)
    written = 0
    for new_name, config, _ in fixes:
        filepath = os.path.join(args.queue_dir, new_name)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)
        written += 1

    print(f"\nWrote {written} fixed config files to {args.queue_dir}")

    # Write fix report
    report_path = os.path.join(args.queue_dir, "_fix_report.json")
    report = []
    for new_name, config, diagnosis in fixes:
        report.append({
            "original": diagnosis.get("notes", ""),
            "fixed_file": new_name,
            "root_cause": diagnosis["root_cause"],
            "fix_strategy": diagnosis["fix_strategy"],
            "load_reduction": diagnosis["load_reduction_factor"],
            "new_load": config["load_power_range_per_branch_kW"],
        })
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Fix report: {report_path}")


if __name__ == "__main__":
    main()
