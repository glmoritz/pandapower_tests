"""
Simulation Worker
-----------------
Scans a directory for parameter JSON files, dispatches them to
duilio_3ph_evaluate.run_simulation(), and saves result JSONs.

Supports parallel execution with configurable number of worker processes.
Uses atomic os.rename() as a filesystem-level lock so multiple workers
(even across machines sharing an NFS mount) never pick the same file.

Usage:
    python -m simulation_worker.SimulationWorker --workers 4
"""

import os
import json
import time
import random
import traceback
import argparse
from datetime import datetime
from multiprocessing import Process
from dotenv import load_dotenv

load_dotenv()


def find_and_lock_param_file():
    """Atomically claim one param file by moving it to the running dir.

    Returns:
        dict with loaded params + metadata keys (base_name, param_file,
        result_file, results_dir), or None if no files remain.
    """
    PARAMS_DIR = os.getenv('PARAMS_DIR')
    RUNNING_DIR = os.getenv('RUNNING_DIR')
    RESULTS_DIR = os.getenv('RESULTS_DIR')
    FINISHED_DIR = os.getenv('FINISHED_DIR')
    FINISHED_SUCCESS_DIR = os.path.join(FINISHED_DIR, '01_success')
    FINISHED_EXCEPTION_DIR = os.path.join(FINISHED_DIR, '02_exception')

    # Create dirs if they don't exist yet
    os.makedirs(RUNNING_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FINISHED_DIR, exist_ok=True)
    os.makedirs(FINISHED_SUCCESS_DIR, exist_ok=True)
    os.makedirs(FINISHED_EXCEPTION_DIR, exist_ok=True)

    for filename in sorted(os.listdir(PARAMS_DIR)):
        if not filename.endswith('.json'):
            continue
        src = os.path.join(PARAMS_DIR, filename)
        dst = os.path.join(RUNNING_DIR, filename)
        try:
            # Atomic rename acts as a distributed lock
            os.rename(src, dst)
            print(f"[INFO] Locked file: {filename}")

            with open(dst, 'r') as f:
                params = json.load(f)

            base_name = filename.replace('.json', '')
            params['base_name'] = base_name
            params['param_file'] = dst
            params['result_file'] = os.path.join(RESULTS_DIR, filename)
            params['results_dir'] = RESULTS_DIR
            params['finished_success_dir'] = FINISHED_SUCCESS_DIR
            params['finished_exception_dir'] = FINISHED_EXCEPTION_DIR
            # Use base_name as output_file to avoid collisions in parallel runs
            params['output_file'] = f"{base_name}.csv"

            return params
        except (FileNotFoundError, OSError):
            continue  # Already taken by another worker
    return None


def worker(worker_id):
    """Worker loop: pick param files and run simulations until none remain."""
    from duilio_3ph_evaluate import run_simulation

    while True:
        params = find_and_lock_param_file()
        if params is None:
            print(f"[Worker-{worker_id}] No more param files. Done.")
            break

        base_name = params['base_name']
        print(f"[Worker-{worker_id}] Starting: {base_name}")

        delay_base_s = max(float(params.get(
            'simulation_start_delay_s',
            os.getenv('SIMULATION_START_DELAY_S', '0.5')
        )), 0.0)
        delay_jitter_s = max(float(params.get(
            'simulation_start_delay_jitter_s',
            os.getenv('SIMULATION_START_DELAY_JITTER_S', '0.5')
        )), 0.0)
        worker_skew_s = max(float(params.get(
            'simulation_start_worker_skew_s',
            os.getenv('SIMULATION_START_WORKER_SKEW_S', '0.15')
        )), 0.0)

        start_delay_s = delay_base_s + (worker_id * worker_skew_s)
        if delay_jitter_s > 0:
            start_delay_s += random.uniform(0.0, delay_jitter_s)

        if start_delay_s > 0:
            print(f"[Worker-{worker_id}] Delaying start by {start_delay_s:.3f}s: {base_name}")
            time.sleep(start_delay_s)

        try:
            result = run_simulation(params)
        except Exception as e:
            # Catch any exception not handled inside run_simulation (setup errors)
            result = {
                'name': params.get('name', ''),
                'base_name': base_name,
                'status': 'setup_error',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'finished_at': datetime.now().isoformat(),
            }

        # Write result JSON
        with open(params['result_file'], 'w') as f:
            json.dump(result, f, indent=2)

        # Move param file to success/exception finished dir
        try:
            success_dir = params.get('finished_success_dir')
            exception_dir = params.get('finished_exception_dir')

            if not success_dir or not exception_dir:
                base_finished_dir = os.getenv('FINISHED_DIR', '')
                success_dir = os.path.join(base_finished_dir, '01_success')
                exception_dir = os.path.join(base_finished_dir, '02_exception')

            os.makedirs(success_dir, exist_ok=True)
            os.makedirs(exception_dir, exist_ok=True)

            target_dir = success_dir if result.get('status') == 'success' else exception_dir
            finished_path = os.path.join(target_dir, os.path.basename(params['param_file']))
            os.rename(params['param_file'], finished_path)
        except OSError:
            pass

        status = result.get('status', 'unknown')
        duration = result.get('duration_s', '?')
        print(f"[Worker-{worker_id}] Finished: {base_name} -> {status} ({duration}s)")


def run_all(num_workers=1):
    """Launch worker processes to consume all parameter files."""
    print(f"[INFO] Starting {num_workers} worker(s)")

    if num_workers <= 1:
        worker(0)
    else:
        processes = []
        for i in range(num_workers):
            p = Process(target=worker, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print("[INFO] All workers finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulation workers')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel worker processes (default: 1)')
    args = parser.parse_args()
    run_all(num_workers=args.workers)
