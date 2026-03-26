#!/usr/bin/env python3
"""
Generate simulation queue JSON files.

Modes:
    - matrix: generate one config per Solar × Storage × Seed combination.
    - fixed-storage-sweep: generate a baseline profile batch plus a storage-sweep
      batch that reuses the same randomized load/PV assignment for all storage
      levels of a given solar+seed combination.

Usage:
    python generate_queue_files.py
    python generate_queue_files.py --output-dir some/dir
    python generate_queue_files.py --mode fixed-storage-sweep
    python generate_queue_files.py --mode fixed-storage-sweep \
        --output-dir results/duilio_debug/01_params \
        --baseline-dir results/duilio_debug/00_baseline_profiles
"""

import argparse
import json
import os

# --- Parameter matrix ---

SOLAR_CONFIGS = {
    5:  [5, 8],
    10: [10, 15],
    20: [20, 25],
}

STORAGE_CONFIGS = {
    0:  [0, 0],
    5:  [5, 5],
    10: [10, 10],
    20: [20, 20],
    30: [30, 30],
    40: [40, 40],
    50: [50, 50],
    60: [60, 60],
    70: [70, 70],
    80: [80, 80],
    90: [90, 90],
    100: [100, 100],
}

SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888,
         9999, 10101, 11111, 12222, 13333, 14444, 15555, 16666]

# --- Fixed parameters (common to all configs) ---

FIXED_PARAMS = {
    "description": "This is the first test configuration for the pandapower simulation.",
    "step_size_s": 900,
    "start_time": "2018-01-01 00:00:00",
    "simulation_time_s": 2592000,
    "panel_efficiency": 0.5,
    "use_saved_network_if_exists": True,
    "use_saved_power_profile_if_exists": True,
    "mv_bus_latitude": -25.4505,
    "mv_bus_longitude": -49.231,
    "network_id": "medium_test_network",
    "commercial_range": [0, 0],
    "industrial_range": [0, 0],
    "residential_range": [5, 5],
    "line_buses_range": [3, 5],
    "line_forks_range": [1, 2],
    "solar_power_steps": 10,
    "load_power_range_per_branch_kW": [15, 15],
    "initial_capacity_range": [50.0, 50.0],
}

# --- Standardized parameters (user chose option 3: breaker=4, balance=true, fork=[30,80]) ---

STANDARDIZED_PARAMS = {
    "fork_length_range": [50, 80],
    "breaker_limit_per_phase_kW": 8,
    "balance_phase_loading": True,
}


def build_common_config(seed, name, solar_kw, storage_range_kwh, power_profile_id):
    """Build config payload shared by all generation modes."""
    base_name = name.replace(".json", "")
    config = {
        "random_seed": seed,
        "name": name,
        "output_file": f"{base_name}.csv",
        "power_profile_id": power_profile_id,
        "solar_power_range_per_branch_kW": SOLAR_CONFIGS[solar_kw],
        "storage_capacity_range_per_branch_kWh": list(storage_range_kwh),
    }
    config.update(FIXED_PARAMS)
    config.update(STANDARDIZED_PARAMS)
    return config


def generate_config(solar_kw, storage_kwh, seed):
    """Generate a single simulation config dict for the full matrix mode."""
    name = f"High-Consumption-Solar-{solar_kw}-Storage-{storage_kwh}-seed{seed}.json"
    config = build_common_config(
        seed=seed,
        name=name,
        solar_kw=solar_kw,
        storage_range_kwh=STORAGE_CONFIGS[storage_kwh],
        power_profile_id=f"Solar-{solar_kw}-Storage-{storage_kwh}-Seed-{seed}",
    )
    config["fixed_load_profile_id"] = f"Solar-{solar_kw}-Seed-{seed}"
    return name, config


def generate_baseline_config(solar_kw, seed, baseline_storage_kwh):
    """Generate the baseline profile used by a fixed storage sweep."""
    fixed_profile_id = f"Solar-{solar_kw}-Seed-{seed}"
    name = f"Baseline-Solar-{solar_kw}-Seed-{seed}.json"
    config = build_common_config(
        seed=seed,
        name=name,
        solar_kw=solar_kw,
        storage_range_kwh=[baseline_storage_kwh, baseline_storage_kwh],
        power_profile_id=fixed_profile_id,
    )
    config["fixed_load_profile_id"] = fixed_profile_id
    config["require_existing_assets"] = False
    return name, config


def generate_fixed_sweep_config(solar_kw, storage_kwh, seed):
    """Generate a storage-sweep config that reuses a baseline load/PV profile."""
    fixed_profile_id = f"Solar-{solar_kw}-Seed-{seed}"
    name = f"High-Consumption-Solar-{solar_kw}-Storage-{storage_kwh}-seed{seed}.json"
    config = build_common_config(
        seed=seed,
        name=name,
        solar_kw=solar_kw,
        storage_range_kwh=STORAGE_CONFIGS[storage_kwh],
        power_profile_id=f"Solar-{solar_kw}-Storage-{storage_kwh}-Seed-{seed}",
    )
    config["fixed_load_profile_id"] = fixed_profile_id
    config["preserve_load_when_changing_storage"] = True
    config["require_existing_assets"] = True
    return name, config


def write_config(output_dir, name, config):
    filepath = os.path.join(output_dir, name)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)


def print_common_summary():
    print(f"\nMatrix: {len(SOLAR_CONFIGS)} solar × {len(STORAGE_CONFIGS)} storage × {len(SEEDS)} seeds")
    print(f"Solar levels: {sorted(SOLAR_CONFIGS.keys())} kW")
    print(f"Storage levels: {sorted(STORAGE_CONFIGS.keys())} kWh")
    print(f"Seeds: {SEEDS}")
    print(
        f"\nStandardized params: breaker={STANDARDIZED_PARAMS['breaker_limit_per_phase_kW']}kW, "
        f"balance={STANDARDIZED_PARAMS['balance_phase_loading']}, "
        f"fork={STANDARDIZED_PARAMS['fork_length_range']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate simulation queue JSON files")
    parser.add_argument(
        "--mode",
        choices=["matrix", "fixed-storage-sweep"],
        default="matrix",
        help=(
            "Generation mode. 'matrix' writes one config per solar/storage/seed. "
            "'fixed-storage-sweep' also writes baseline profile configs so all storage levels "
            "for a given solar+seed reuse the same randomized load/PV assignment."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results", "duilio_debug", "01_params"),
        help="Directory to write JSON files to (default: results/duilio_debug/01_params/)",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help=(
            "Directory for baseline profile JSON files when using --mode fixed-storage-sweep. "
            "Default: <output-dir>/../00_baseline_profiles"
        ),
    )
    parser.add_argument(
        "--baseline-storage-kwh",
        type=float,
        default=0.0,
        help=(
            "Storage level used when generating baseline profiles in fixed-storage-sweep mode. "
            "Use 0 to freeze only load/PV placement; use a non-zero value to also freeze storage ratios."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "matrix":
        count = 0
        for solar_kw in sorted(SOLAR_CONFIGS.keys()):
            for storage_kwh in sorted(STORAGE_CONFIGS.keys()):
                for seed in SEEDS:
                    name, config = generate_config(solar_kw, storage_kwh, seed)
                    write_config(args.output_dir, name, config)
                    count += 1

        print(f"Generated {count} JSON files in {args.output_dir}")
        print_common_summary()
        return

    baseline_dir = args.baseline_dir
    if baseline_dir is None:
        baseline_dir = os.path.join(os.path.dirname(args.output_dir), "00_baseline_profiles")

    os.makedirs(baseline_dir, exist_ok=True)

    baseline_count = 0
    sweep_count = 0

    for solar_kw in sorted(SOLAR_CONFIGS.keys()):
        for seed in SEEDS:
            name, config = generate_baseline_config(solar_kw, seed, args.baseline_storage_kwh)
            write_config(baseline_dir, name, config)
            baseline_count += 1

    for solar_kw in sorted(SOLAR_CONFIGS.keys()):
        for storage_kwh in sorted(STORAGE_CONFIGS.keys()):
            for seed in SEEDS:
                name, config = generate_fixed_sweep_config(solar_kw, storage_kwh, seed)
                write_config(args.output_dir, name, config)
                sweep_count += 1

    print(f"Generated {baseline_count} baseline JSON files in {baseline_dir}")
    print(f"Generated {sweep_count} sweep JSON files in {args.output_dir}")
    print_common_summary()
    print(
        "\nFixed storage sweep workflow:\n"
        f"  1. Pre-generate baseline profiles: python -m simulation_worker.RegenerateAssets --network --profile --folder {baseline_dir}\n"
        f"  2. Run the storage sweep using JSONs from: {args.output_dir}\n"
        "  3. Each sweep run keeps the same randomized load/PV assignment and only updates storage."
    )


if __name__ == "__main__":
    main()
