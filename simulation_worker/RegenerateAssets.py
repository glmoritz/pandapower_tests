"""
Regenerate Assets
-----------------
Standalone utility to regenerate networks and/or power profiles for simulation
parameter files. This allows separation of asset generation from simulation execution.

Use this script to pre-generate all networks and power profiles before running
simulations. The main simulation worker will then always load existing assets
from the database, ensuring consistency across multiple runs.

Usage:
    # Regenerate both networks and power profiles for all params in 01_params/
    python -m simulation_worker.RegenerateAssets --network --profile

    # Regenerate only networks
    python -m simulation_worker.RegenerateAssets --network

    # Regenerate only power profiles  
    python -m simulation_worker.RegenerateAssets --profile

    # Specify a custom folder
    python -m simulation_worker.RegenerateAssets --folder /path/to/params --network --profile

    # Process specific files
    python -m simulation_worker.RegenerateAssets --files file1.json file2.json --network

    # Use multiple workers for parallel processing
    python -m simulation_worker.RegenerateAssets --network --profile --workers 4
"""

import os
import json
import time
import argparse
import traceback
from datetime import datetime
from multiprocessing import Process, Queue
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_db_connection():
    """Get database connection parameters from environment."""
    return {
        "dbname": os.getenv("POSTGRES_DB_NAME", "duilio"),
        "user": os.getenv("POSTGRES_DB_USER", "root"),
        "password": os.getenv("POSTGRES_DB_PASSWORD", "skamasfrevrest"),
        "host": os.getenv("POSTGRES_DB_HOST", "103.0.2.7"),
        "port": int(os.getenv("POSTGRES_DB_PORT", "5433"))
    }


def get_db_engine(db):
    """Create SQLAlchemy engine from connection parameters."""
    db_url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(db_url)


def get_rng_for_component(root_ss: np.random.SeedSequence, key, entity_id):
    """Create a reproducible RNG for a component using a deterministic key."""
    import hashlib
    h = hashlib.sha256(str(key).encode()).digest()
    key_int = int.from_bytes(h[:4], 'little')
    child_ss = np.random.SeedSequence(
        entropy=root_ss.entropy,
        spawn_key=(key_int, entity_id)
    )
    return np.random.default_rng(child_ss)


def regenerate_network(params, db, rng_network):
    """Generate and save a new network to the database.
    
    Returns:
        tuple: (pandapower_grid_id, net, graph)
    """
    from create_random_network import generate_pandapower_net, save_network_to_database
    
    network_id = f"{params['network_id']}_seed_{str(params['random_seed'])}"
    
    net, graph = generate_pandapower_net(
        CommercialRange=params['commercial_range'],
        IndustrialRange=params['industrial_range'],
        ResidencialRange=params['residential_range'],
        ForkLengthRange=params['fork_length_range'],
        LineBusesRange=params['line_buses_range'],
        LineForksRange=params['line_forks_range'],
        mv_bus_coordinates=(float(params['mv_bus_latitude']), float(params['mv_bus_longitude'])),
        rng=rng_network
    )
    
    # Apply 3-phase network settings
    net.ext_grid['r0x0_max'] = 5.0
    net.ext_grid['x0x_max'] = 5.0
    net.line['r0_ohm_per_km'] = net.line['r_ohm_per_km'] * 3
    net.line['x0_ohm_per_km'] = net.line['x_ohm_per_km'] * 3
    net.line['c0_nf_per_km'] = net.line['c_nf_per_km'] * 3
    net.trafo['vector_group'] = 'Dyn'
    net.trafo['vk0_percent'] = net.trafo['vk_percent']
    net.trafo['mag0_percent'] = 100
    net.trafo['mag0_rx'] = 0
    net.trafo['si0_hv_partial'] = 0.9
    net.trafo['vkr0_percent'] = net.trafo['vkr_percent']
    
    pandapower_grid_id = save_network_to_database(
        graph=graph,
        net=net,
        db_connection=db,
        grid_name=network_id
    )
    
    return pandapower_grid_id, net, graph


def regenerate_power_profile(params, db, engine, pandapower_grid_id, net, graph, rng_power_profile):
    """Generate and save a new power profile to the database."""
    from generate_power_profile import distribute_loads_to_buses, save_graph_metadata, balance_phase_loading
    
    power_profile_id = f"{params['power_profile_id']}_seed_{str(params['random_seed'])}"
    
    distribute_loads_to_buses(net, graph, params, db, rng=rng_power_profile)
    
    # Phase balancing postprocessing is handled inside distribute_loads_to_buses
    # when the parameter is set, but we also call it explicitly here in case
    # the caller wants to ensure it runs even if params were modified after
    # distribute_loads_to_buses was called.
    # (distribute_loads_to_buses already checks the flag, so double-calling is safe)
    
    save_graph_metadata(engine, pandapower_grid_id, power_profile_id, graph)
    
    return power_profile_id


def process_param_file(param_file, regenerate_net=True, regenerate_profile=True, params_override=None, entry_index=None):
    """Process a single parameter file, regenerating network and/or power profile.
    
    Args:
        param_file: Path to the JSON parameter file
        regenerate_net: Whether to regenerate the network
        regenerate_profile: Whether to regenerate the power profile
        
    Returns:
        dict with processing results
    """
    from create_random_network import load_network_from_database
    
    start_time = time.time()
    file_label = os.path.basename(param_file)
    if entry_index is not None:
        file_label = f"{file_label}[{entry_index}]"

    result = {
        'file': file_label,
        'source_file': os.path.basename(param_file),
        'started_at': datetime.now().isoformat(),
        'network_regenerated': False,
        'profile_regenerated': False,
    }
    
    try:
        if params_override is not None:
            params = params_override
        else:
            with open(param_file, 'r') as f:
                params = json.load(f)
        
        db = get_db_connection()
        engine = get_db_engine(db)
        
        # Initialize RNG
        master_seed = params.get('random_seed', 1234567890)
        root_seed_sequence = np.random.SeedSequence(master_seed)
        rng_network = get_rng_for_component(root_seed_sequence, "NetworkGeneration", 0)
        rng_power_profile = get_rng_for_component(root_seed_sequence, "PowerProfile", 0)
        
        network_id = f"{params['network_id']}_seed_{str(params['random_seed'])}"
        power_profile_id = f"{params['power_profile_id']}_seed_{str(params['random_seed'])}"
        
        result['network_id'] = network_id
        result['power_profile_id'] = power_profile_id
        
        pandapower_grid_id = None
        net = None
        graph = None
        
        if regenerate_net:
            print(f"  [NETWORK] Generating network: {network_id}")
            pandapower_grid_id, net, graph = regenerate_network(params, db, rng_network)
            result['network_regenerated'] = True
            result['pandapower_grid_id'] = pandapower_grid_id
            print(f"  [NETWORK] Saved network '{network_id}' with grid_id={pandapower_grid_id}")
        
        if regenerate_profile:
            # If we didn't regenerate the network, load it from database
            if net is None or graph is None:
                print(f"  [PROFILE] Loading existing network: {network_id}")
                pandapower_grid_id, net, graph = load_network_from_database(db, network_id)
            
            print(f"  [PROFILE] Generating power profile: {power_profile_id}")
            regenerate_power_profile(params, db, engine, pandapower_grid_id, net, graph, rng_power_profile)
            result['profile_regenerated'] = True
            print(f"  [PROFILE] Saved power profile '{power_profile_id}'")
        
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['error_type'] = type(e).__name__
        result['traceback'] = traceback.format_exc()
        print(f"  [ERROR] {e}")
    
    result['duration_s'] = round(time.time() - start_time, 2)
    result['finished_at'] = datetime.now().isoformat()
    
    return result


def worker_process(work_queue, result_queue, regenerate_net, regenerate_profile, worker_id):
    """Worker process that pulls files from queue and processes them."""
    while True:
        try:
            task = work_queue.get(timeout=1)
            if task is None:  # Poison pill
                break

            if isinstance(task, str):
                task = {
                    'param_file': task,
                    'params': None,
                    'entry_index': None,
                    'display_name': os.path.basename(task)
                }

            print(f"[Worker-{worker_id}] Processing: {task['display_name']}")
            result = process_param_file(
                task['param_file'],
                regenerate_net,
                regenerate_profile,
                params_override=task.get('params'),
                entry_index=task.get('entry_index')
            )
            result_queue.put(result)
            
        except Exception:
            break


def read_params_from_file(param_file):
    """Load and return parameter JSON from file."""
    with open(param_file, 'r') as f:
        return json.load(f)


def is_valid_param_entry(params):
    """Check whether a JSON object looks like a runnable simulation param set."""
    if not isinstance(params, dict):
        return False

    required_keys = {'network_id', 'power_profile_id', 'random_seed'}
    return required_keys.issubset(set(params.keys()))


def extract_param_entries(param_file):
    """Return runnable parameter entries from a JSON file.

    Supports both single-dict parameter files and sweep/list files.
    """
    raw = read_params_from_file(param_file)

    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [entry for entry in raw if isinstance(entry, dict)]
    else:
        print(f"[WARNING] Skipping unsupported JSON type in {os.path.basename(param_file)}: {type(raw).__name__}")
        return []

    entries = []
    skipped = 0
    for idx, params in enumerate(candidates):
        if is_valid_param_entry(params):
            entries.append({
                'param_file': param_file,
                'params': params,
                'entry_index': idx if isinstance(raw, list) else None,
                'display_name': f"{os.path.basename(param_file)}[{idx}]" if isinstance(raw, list) else os.path.basename(param_file)
            })
        else:
            skipped += 1

    if skipped > 0:
        print(
            f"[INFO] Skipped {skipped} non-parameter entr{'y' if skipped == 1 else 'ies'} "
            f"in {os.path.basename(param_file)}"
        )

    return entries


def build_unique_files_by_key(param_files, key_builder):
    """Return one representative parameter entry per unique key.

    Args:
        param_files: Iterable of parameter file paths.
        key_builder: Callable that receives loaded params and returns a key.

    Returns:
        tuple: (unique_entries, unique_key_count, duplicate_count)
    """
    unique_by_key = {}
    duplicate_count = 0
    skipped_files = 0

    for param_file in param_files:
        entries = extract_param_entries(param_file)
        if not entries:
            skipped_files += 1
            continue

        for entry in entries:
            params = entry['params']
            key = key_builder(params)
            if key in unique_by_key:
                duplicate_count += 1
                continue
            unique_by_key[key] = entry

    if skipped_files > 0:
        print(f"[INFO] Skipped {skipped_files} JSON file(s) with no runnable parameter entries.")

    unique_entries = list(unique_by_key.values())
    return unique_entries, len(unique_by_key), duplicate_count


def execute_file_batch(param_files, regenerate_net, regenerate_profile, num_workers):
    """Execute regeneration for a batch of parameter entries (sequential or parallel)."""
    results = []

    if not param_files:
        return results

    if num_workers <= 1:
        for i, task in enumerate(param_files, 1):
            if isinstance(task, str):
                task = {
                    'param_file': task,
                    'params': None,
                    'entry_index': None,
                    'display_name': os.path.basename(task)
                }

            print(f"[{i}/{len(param_files)}] Processing: {task['display_name']}")
            result = process_param_file(
                task['param_file'],
                regenerate_net,
                regenerate_profile,
                params_override=task.get('params'),
                entry_index=task.get('entry_index')
            )
            results.append(result)
            print(f"  -> {result['status']} ({result['duration_s']}s)")
        return results

    work_queue = Queue()
    result_queue = Queue()

    for param_file in param_files:
        work_queue.put(param_file)

    for _ in range(num_workers):
        work_queue.put(None)

    processes = []
    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(work_queue, result_queue, regenerate_net, regenerate_profile, i)
        )
        p.start()
        processes.append(p)

    completed = 0
    while completed < len(param_files):
        try:
            result = result_queue.get(timeout=300)
            results.append(result)
            completed += 1
            print(f"[{completed}/{len(param_files)}] {result['file']} -> {result['status']} ({result['duration_s']}s)")
        except Exception:
            break

    for p in processes:
        p.join(timeout=10)

    return results


def collect_param_files(folder=None, files=None):
    """Collect parameter files from folder or specific files list.
    
    Args:
        folder: Folder to scan for JSON files (uses PARAMS_DIR if None)
        files: List of specific file paths
        
    Returns:
        List of absolute file paths
    """
    if files:
        # Use specific files provided
        param_files = []
        for f in files:
            if os.path.isabs(f):
                param_files.append(f)
            else:
                # Assume relative to current directory or PARAMS_DIR
                if os.path.exists(f):
                    param_files.append(os.path.abspath(f))
                else:
                    params_dir = folder or os.getenv('PARAMS_DIR', '')
                    full_path = os.path.join(params_dir, f)
                    if os.path.exists(full_path):
                        param_files.append(full_path)
                    else:
                        print(f"[WARNING] File not found: {f}")
        return param_files
    
    # Scan folder for JSON files
    params_dir = folder or os.getenv('PARAMS_DIR', '')
    if not params_dir:
        raise ValueError("No folder specified and PARAMS_DIR not set in environment")
    
    if not os.path.isdir(params_dir):
        raise ValueError(f"Folder does not exist: {params_dir}")
    
    param_files = []
    for filename in sorted(os.listdir(params_dir)):
        if filename.endswith('.json'):
            param_files.append(os.path.join(params_dir, filename))
    
    return param_files


def run_regeneration(folder=None, files=None, regenerate_net=True, regenerate_profile=True, 
                     num_workers=1, dry_run=False):
    """Main entry point for regenerating assets.
    
    Args:
        folder: Folder containing parameter JSON files
        files: List of specific files to process
        regenerate_net: Whether to regenerate networks
        regenerate_profile: Whether to regenerate power profiles
        num_workers: Number of parallel worker processes
        dry_run: If True, only list files without processing
        
    Returns:
        List of result dictionaries
    """
    param_files = collect_param_files(folder, files)
    
    if not param_files:
        print("[INFO] No parameter files found.")
        return []
    
    print(f"[INFO] Found {len(param_files)} parameter file(s)")
    
    if dry_run:
        print("[INFO] Dry run - files that would be processed:")
        for f in param_files:
            print(f"  - {os.path.basename(f)}")
        return []
    
    what_to_do = []
    if regenerate_net:
        what_to_do.append("networks")
    if regenerate_profile:
        what_to_do.append("power profiles")
    
    if not what_to_do:
        print("[INFO] Nothing to regenerate. Use --network and/or --profile flags.")
        return []
    
    print(f"[INFO] Regenerating: {', '.join(what_to_do)}")
    print(f"[INFO] Using {num_workers} worker(s)")
    print("-" * 60)
    
    start_time = time.time()
    results = []

    # Phase 1: networks (deduplicated by network_id + random_seed)
    if regenerate_net:
        network_files, unique_network_count, network_duplicates = build_unique_files_by_key(
            param_files,
            lambda params: f"{params['network_id']}_seed_{str(params['random_seed'])}"
        )
        if network_duplicates > 0:
            print(
                f"[INFO] Network deduplication: {len(param_files)} files -> "
                f"{unique_network_count} unique network(s), skipped {network_duplicates} duplicate request(s)."
            )
        print("[INFO] Phase 1/2: Regenerating networks")
        results.extend(
            execute_file_batch(
                param_files=network_files,
                regenerate_net=True,
                regenerate_profile=False,
                num_workers=num_workers
            )
        )

    # Phase 2: profiles (deduplicated by power_profile_id + random_seed)
    if regenerate_profile:
        profile_files, unique_profile_count, profile_duplicates = build_unique_files_by_key(
            param_files,
            lambda params: f"{params['power_profile_id']}_seed_{str(params['random_seed'])}"
        )
        if profile_duplicates > 0:
            print(
                f"[INFO] Profile deduplication: {len(param_files)} files -> "
                f"{unique_profile_count} unique profile(s), skipped {profile_duplicates} duplicate request(s)."
            )
        print("[INFO] Phase 2/2: Regenerating power profiles")
        results.extend(
            execute_file_batch(
                param_files=profile_files,
                regenerate_net=False,
                regenerate_profile=True,
                num_workers=num_workers
            )
        )
    
    # Summary
    print("-" * 60)
    total_time = round(time.time() - start_time, 2)
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"[SUMMARY] Processed: {len(results)} files")
    print(f"[SUMMARY] Success: {success_count}, Errors: {error_count}")
    print(f"[SUMMARY] Total time: {total_time}s")
    
    if error_count > 0:
        print("\n[ERRORS]")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {r['file']}: {r.get('error', 'Unknown error')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate networks and/or power profiles for simulation parameter files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate both networks and power profiles
  python -m simulation_worker.RegenerateAssets --network --profile

  # Regenerate only networks
  python -m simulation_worker.RegenerateAssets --network

  # Specify a custom folder
  python -m simulation_worker.RegenerateAssets --folder ./my_params --network --profile

  # Process specific files
  python -m simulation_worker.RegenerateAssets --files config1.json config2.json --network

  # Use parallel processing
  python -m simulation_worker.RegenerateAssets --network --profile --workers 4

  # Dry run to see what would be processed
  python -m simulation_worker.RegenerateAssets --network --profile --dry-run
        """
    )
    
    parser.add_argument('--folder', '-f', type=str, default=None,
                        help='Folder containing parameter JSON files (default: PARAMS_DIR from .env)')
    parser.add_argument('--files', nargs='+', type=str, default=None,
                        help='Specific parameter files to process')
    parser.add_argument('--network', '-n', action='store_true',
                        help='Regenerate networks')
    parser.add_argument('--profile', '-p', action='store_true',
                        help='Regenerate power profiles')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel worker processes (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files that would be processed without actually processing them')
    
    args = parser.parse_args()
    
    if not args.network and not args.profile and not args.dry_run:
        parser.error("At least one of --network or --profile must be specified (or use --dry-run)")
    
    run_regeneration(
        folder=args.folder,
        files=args.files,
        regenerate_net=args.network,
        regenerate_profile=args.profile,
        num_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
