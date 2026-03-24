#!/usr/bin/env python3
"""
Generate simulation queue JSON files for all Solar × Storage × Seed combinations.

Creates 192 JSON config files (3 solar levels × 4 storage levels × 16 seeds)
with standardized parameters (breaker=4, balance=true, fork=[30,80]).

Usage:
    python generate_queue_files.py                        # writes to results/duilio_debug/01_params/
    python generate_queue_files.py --output-dir some/dir  # writes to custom directory
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
    "load_power_range_per_branch_kW": [5, 5],
    "initial_capacity_range": [50.0, 50.0],
}

# --- Standardized parameters (user chose option 3: breaker=4, balance=true, fork=[30,80]) ---

STANDARDIZED_PARAMS = {
    "fork_length_range": [30, 80],
    "breaker_limit_per_phase_kW": 4,
    "balance_phase_loading": True,
}


def generate_config(solar_kw, storage_kwh, seed):
    """Generate a single simulation config dict."""
    name = f"Solar-{solar_kw}-Storage-{storage_kwh}-seed{seed}.json"
    base_name = name.replace(".json", "")

    config = {
        "random_seed": seed,
        "name": name,
        "output_file": f"{base_name}.csv",
        "power_profile_id": f"Solar-{solar_kw}-Storage-{storage_kwh}",
        "solar_power_range_per_branch_kW": SOLAR_CONFIGS[solar_kw],
        "storage_capacity_range_per_branch_kWh": STORAGE_CONFIGS[storage_kwh],
    }
    config.update(FIXED_PARAMS)
    config.update(STANDARDIZED_PARAMS)
    return name, config


def main():
    parser = argparse.ArgumentParser(description="Generate simulation queue JSON files")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results", "duilio_debug", "01_params"),
        help="Directory to write JSON files to (default: results/duilio_debug/01_params/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    count = 0
    for solar_kw in sorted(SOLAR_CONFIGS.keys()):
        for storage_kwh in sorted(STORAGE_CONFIGS.keys()):
            for seed in SEEDS:
                name, config = generate_config(solar_kw, storage_kwh, seed)
                filepath = os.path.join(args.output_dir, name)
                with open(filepath, "w") as f:
                    json.dump(config, f, indent=4)
                count += 1

    print(f"Generated {count} JSON files in {args.output_dir}")

    # Summary
    print(f"\nMatrix: {len(SOLAR_CONFIGS)} solar × {len(STORAGE_CONFIGS)} storage × {len(SEEDS)} seeds")
    print(f"Solar levels: {sorted(SOLAR_CONFIGS.keys())} kW")
    print(f"Storage levels: {sorted(STORAGE_CONFIGS.keys())} kWh")
    print(f"Seeds: {SEEDS}")
    print(f"\nStandardized params: breaker={STANDARDIZED_PARAMS['breaker_limit_per_phase_kW']}kW, "
          f"balance={STANDARDIZED_PARAMS['balance_phase_loading']}, "
          f"fork={STANDARDIZED_PARAMS['fork_length_range']}")


if __name__ == "__main__":
    main()
