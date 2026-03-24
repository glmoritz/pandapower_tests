#!/usr/bin/env python3
"""
Analyze convergence results from simulation runs.

Scans result JSON files from the simulation output directories, aggregates
convergence statistics per Solar-Storage combination, and generates summary
reports as CSV files and console output.

Usage:
    python analyze_convergence_results.py
    python analyze_convergence_results.py --results-dir results/duilio_debug/03_results
    python analyze_convergence_results.py --scan-finished   # also scan 04_finished/
"""

import argparse
import csv
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


def parse_result(filepath):
    """Parse a result JSON file and extract key fields."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract solar and storage from name or power_profile_id
    name = data.get("name", data.get("base_name", os.path.basename(filepath)))

    # Try to parse Solar-X-Storage-Y-seedZ pattern
    m = re.match(r"Solar-(\d+)-Storage-(\d+)-seed(\d+)", name)
    if m:
        solar_kw = int(m.group(1))
        storage_kwh = int(m.group(2))
        seed = int(m.group(3))
    else:
        # Fallback: extract from params
        solar_range = data.get("solar_power_range", data.get("solar_power_range_per_branch_kW"))
        storage_range = data.get("storage_capacity_range", data.get("storage_capacity_range_per_branch_kWh"))
        solar_kw = solar_range[0] if solar_range else None
        storage_kwh = storage_range[0] if storage_range else None
        seed = data.get("random_seed")

    status = data.get("status", "unknown")
    converged = data.get("converged", status == "success")
    error_type = data.get("error_type", "")
    error_msg = data.get("error", "")
    duration_s = data.get("duration_s")

    # Convergence debug info
    debug = data.get("convergence_debug")
    debug_summary = data.get("convergence_debug_summary", "")
    debug_total_issues = data.get("convergence_debug_total_issues")

    # Extract root cause classification from convergence_debug
    root_cause = classify_root_cause(debug, debug_summary, error_type)

    return {
        "file": filepath,
        "name": name,
        "solar_kw": solar_kw,
        "storage_kwh": storage_kwh,
        "seed": seed,
        "status": status,
        "converged": converged,
        "error_type": error_type,
        "error_msg": error_msg[:200] if error_msg else "",
        "duration_s": duration_s,
        "root_cause": root_cause,
        "debug_summary": debug_summary[:500] if debug_summary else "",
        "debug_total_issues": debug_total_issues,
    }


def classify_root_cause(debug, debug_summary, error_type):
    """Classify the root cause of a convergence failure."""
    if not debug and not debug_summary:
        if error_type:
            return error_type
        return "unknown"

    causes = []

    if debug:
        # Check voltage collapse
        collapse = debug.get("voltage_collapse", [])
        for vc in collapse:
            if vc.get("status") == "voltage_collapse":
                causes.append("voltage_collapse")
                break

        # Check phase unbalance
        load_dist = debug.get("load_distribution", {})
        for trafo in load_dist.get("trafo_loads", []):
            unbalance = trafo.get("unbalance_percent", 0)
            if unbalance and unbalance > 100:
                causes.append("phase_unbalance")
                break

        # Check feeder impedance
        feeder = debug.get("feeder_impedance", [])
        for f in feeder:
            max_kw = f.get("theoretical_max_1ph_kw", float("inf"))
            if max_kw < 30:
                causes.append("feeder_impedance")
                break

        # Check parameter issues
        param_issues = debug.get("parameter_issues", [])
        if param_issues:
            causes.append("parameter_issues")

        # Check solver attempts - did any converge?
        solver = debug.get("solver_attempts", [])
        any_converged = any(
            s.get("status") == "converged" for s in solver
        )
        if solver and not any_converged:
            if not causes:
                causes.append("all_solvers_failed")

    if not causes:
        # Try to infer from summary text
        summary_lower = (debug_summary or "").lower()
        if "voltage collapse" in summary_lower:
            causes.append("voltage_collapse")
        elif "unbalanc" in summary_lower:
            causes.append("phase_unbalance")
        elif "impedance" in summary_lower:
            causes.append("feeder_impedance")
        elif "singular" in summary_lower:
            causes.append("singular_matrix")
        else:
            causes.append(error_type or "unknown")

    return "; ".join(causes) if causes else "unknown"


def group_results(results):
    """Group results by Solar-Storage combination."""
    groups = {}
    for r in results:
        key = (r["solar_kw"], r["storage_kwh"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    return groups


def compute_summary(groups):
    """Compute convergence statistics per Solar-Storage group."""
    rows = []
    for (solar, storage), results in sorted(groups.items()):
        total = len(results)
        converged = sum(1 for r in results if r["converged"])
        failed = total - converged
        rate = (converged / total * 100) if total > 0 else 0

        # Failure breakdown
        failure_causes = {}
        failed_seeds = []
        for r in results:
            if not r["converged"]:
                failed_seeds.append(str(r["seed"]))
                cause = r["root_cause"]
                failure_causes[cause] = failure_causes.get(cause, 0) + 1

        rows.append({
            "solar_kw": solar,
            "storage_kwh": storage,
            "total_runs": total,
            "converged": converged,
            "failed": failed,
            "convergence_rate_pct": round(rate, 1),
            "failed_seeds": ", ".join(failed_seeds) if failed_seeds else "",
            "failure_causes": "; ".join(f"{k}({v})" for k, v in sorted(failure_causes.items())) if failure_causes else "",
        })

    return rows


def print_summary_table(summary_rows):
    """Print a nicely formatted summary table to stdout."""
    print("\n" + "=" * 100)
    print("CONVERGENCE STATISTICS SUMMARY")
    print("=" * 100)
    print(f"{'Solar(kW)':<12} {'Storage(kWh)':<14} {'Total':<7} {'OK':<5} {'Fail':<6} {'Rate(%)':<9} {'Failure Causes'}")
    print("-" * 100)

    total_all = 0
    converged_all = 0
    for row in summary_rows:
        total_all += row["total_runs"]
        converged_all += row["converged"]
        print(f"{row['solar_kw']:<12} {row['storage_kwh']:<14} {row['total_runs']:<7} "
              f"{row['converged']:<5} {row['failed']:<6} {row['convergence_rate_pct']:<9} "
              f"{row['failure_causes']}")

    print("-" * 100)
    overall_rate = (converged_all / total_all * 100) if total_all > 0 else 0
    print(f"{'TOTAL':<12} {'':<14} {total_all:<7} {converged_all:<5} {total_all - converged_all:<6} {round(overall_rate, 1):<9}")
    print("=" * 100)

    # Print failed seeds detail
    any_failures = any(row["failed"] > 0 for row in summary_rows)
    if any_failures:
        print("\nFAILED SIMULATIONS DETAIL:")
        print("-" * 100)
        for row in summary_rows:
            if row["failed_seeds"]:
                print(f"  Solar-{row['solar_kw']}-Storage-{row['storage_kwh']}: "
                      f"seeds [{row['failed_seeds']}] -> {row['failure_causes']}")
        print()


def write_summary_csv(summary_rows, filepath):
    """Write summary CSV (one row per Solar-Storage combo)."""
    fieldnames = ["solar_kw", "storage_kwh", "total_runs", "converged", "failed",
                  "convergence_rate_pct", "failed_seeds", "failure_causes"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_detailed_csv(results, filepath):
    """Write detailed CSV (one row per simulation)."""
    fieldnames = ["name", "solar_kw", "storage_kwh", "seed", "status", "converged",
                  "error_type", "root_cause", "duration_s", "debug_total_issues",
                  "debug_summary", "error_msg", "file"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(script_dir, "results", "duilio_debug", "03_results")

    parser = argparse.ArgumentParser(description="Analyze convergence results from simulation runs")
    parser.add_argument(
        "--results-dir",
        default=default_results,
        help=f"Directory containing result JSON files (default: {default_results})",
    )
    parser.add_argument(
        "--scan-finished",
        action="store_true",
        help="Also scan 04_finished/ subdirectories for result files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSV files (default: same as results-dir)",
    )
    args = parser.parse_args()

    # Collect directories to scan
    scan_dirs = [args.results_dir]
    if args.scan_finished:
        base = os.path.dirname(args.results_dir)
        finished_dir = os.path.join(base, "04_finished")
        scan_dirs.append(finished_dir)

    # Find and parse all result files
    result_files = find_result_files(scan_dirs)
    if not result_files:
        print(f"No result JSON files found in: {scan_dirs}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files in {len(scan_dirs)} directories")

    results = []
    parse_errors = 0
    for fp in result_files:
        try:
            results.append(parse_result(fp))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  Warning: could not parse {fp}: {e}")
            parse_errors += 1

    if parse_errors:
        print(f"  ({parse_errors} files could not be parsed)")

    print(f"Parsed {len(results)} simulation results")

    # Group and compute statistics
    groups = group_results(results)
    summary_rows = compute_summary(groups)

    # Print to console
    print_summary_table(summary_rows)

    # Write CSVs
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "convergence_summary.csv")
    detailed_csv = os.path.join(output_dir, "convergence_detailed.csv")

    write_summary_csv(summary_rows, summary_csv)
    write_detailed_csv(results, detailed_csv)

    print(f"\nSummary CSV: {summary_csv}")
    print(f"Detailed CSV: {detailed_csv}")


if __name__ == "__main__":
    main()
