from dotenv import load_dotenv

import os
import json
import time
import shutil

PARAMS_DIR = None
RUNNING_DIR = None
RESULTS_DIR = None

def find_and_lock_param_file():
    """Atomically move one param file to the running dir."""
    load_dotenv()
    PARAMS_DIR = os.getenv('PARAMS_DIR')
    RESULTS_DIR = os.getenv('RESULTS_DIR')
    RUNNING_DIR = os.getenv('RESULTS_DIR')

    for filename in os.listdir(PARAMS_DIR):
        if not filename.endswith('.json'):
            continue
        src = os.path.join(PARAMS_DIR, filename)
        dst = os.path.join(RUNNING_DIR, filename)
        try:
            # Atomic move (acts as a lock)
            os.rename(src, dst)
            shutil.copy2(dst, src) #debug: copy the file to the source directory
            
            print(f"[INFO] Locked file: {filename}")

            base_name = os.path.basename(dst).replace('.json', '')
            result_file = os.path.join(RESULTS_DIR, f"{base_name}_result.json")
            params = None            
            with open(dst, 'r') as param_file:
                params = json.load(param_file)
            

            params['base_name'] = base_name
            params['result_file'] = result_file
            params['results_dir'] = RESULTS_DIR

            return params
        except FileNotFoundError:
            continue  # Already taken
        except OSError:
            continue  # Race condition or permission denied
    return None

def run_simulation(params):
    """Dummy simulation: replace with your actual logic."""
    result = {"input": params, "output": sum(params.get("values", []))}
    time.sleep(1)  # Simulate computation
    return result

def get_next_simulation(simulation_name: str):
    load_dotenv()

    
    RUNNING_DIR = os.getenv('RUNNING_DIR')
    RESULTS_DIR = os.getenv('RESULTS_DIR')

    while True:
        param_file = find_and_lock_param_file()
        if not param_file:
            print("[INFO] No more param files. Exiting.")
            break

        with open(param_file, 'r') as f:
            params = json.load(f)

        result = run_simulation(params)

        # Write result
        base_name = os.path.basename(param_file).replace('.json', '')
        result_file = os.path.join(RESULTS_DIR, f"{base_name}_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Optionally clean up
        os.remove(param_file)
        print(f"[INFO] Finished and saved: {result_file}")

if __name__ == '__main__':
    main()


