import pandas as pd
import mosaik_api_v3 as mosaik_api
from sqlalchemy import create_engine, text
import json
import random
import threading
import queue
import re

META = {
    'type': 'event-based',
    'models': {
        'PostgresWriterModel': {
            'public': True,
            'any_inputs': True,
            'params': [
                'buff_size', 'attrs',
                'db_connection',
                'simulation_params', 'output_csv', 'write_to_db'
            ],
            'attrs': [],
        },
    },
}

def parse_variable_name(var_name):
    """Parse variable name and return structured data"""
    # Get the rightmost part after any dot
    if '.' in var_name:
        parts = var_name.split('.')        
        element_part = parts[-1]   # e.g. "HouseholdProducer_0-EnergyConsumption[MWh]"
    else:        
        element_part = var_name
      
    
    # Extract element and output parts
    # This handles patterns like:
    # - HouseholdProducer_0-EnergyConsumption[MWh]
    # - Bus-1-P_a[MW]
    element_parts = element_part.split('-')
    
    # The output with unit is always the last part
    output_with_unit = element_parts[-1]
    
    # The element info is in the earlier parts
    element_info = '-'.join(element_parts[:-1]) if len(element_parts) > 1 else ""
    
    # Extract element type and index
    element_type = None
    element_index = None
    
    # Try to extract element info with different patterns
    if element_info:
        # Try pattern like "HouseholdProducer_0" or "Bus-1"
        element_match = re.search(r'([A-Za-z]+)(?:_|-)?(\d+)?', element_info)
        if element_match:
            element_type = element_match.group(1)
            if element_match.group(2):
                element_index = int(element_match.group(2))
    
    # Extract unit if present
    unit = None
    if '[' in output_with_unit and ']' in output_with_unit:
        unit_match = re.search(r'\[([^\]]+)\]', output_with_unit)
        if unit_match:
            unit = unit_match.group(1)
    
    
    return {
        'element': element_info,  # e.g., "HouseholdProducer_0" or "Bus-1"
        'element_type': element_type.lower() if element_type else None,  # e.g., "householdproducer"
        'element_index': element_index,  # e.g., 0 or 1        
        'output': output_with_unit,  # e.g., "EnergyConsumption[MWh]"
        'unit': unit,  # e.g., "MWh"        
    }

class PostgresWriterModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = None
        self.time_resolution = None
        self.date_format = None
        self.start_date = None
        self.output_file = None
        self.df = None
        self.buff_size = None
        self.attrs = None
        self.nan_representation = None

        # DB related
        self.engine = None
        self.conn = None
        self.write_to_db = False
        self.simulation_id = None
        self.variable_map = {}  # maps var name -> variable_id
        self.step_index = 0  # Track simulation step index
        self.db_params = None  # Store DB connection params for refresh thread

        self.output_csv = True
        
        # Debug
        self.debug = False

        # Threading
        self.flush_queue = queue.Queue()
        self.flush_thread = None
        self.stop_event = threading.Event()

    def init(self, sid, time_resolution, start_date,
             date_format='%Y-%m-%d %H:%M:%S',
             output_file='results.csv',
             nan_rep='NaN',
             # DB params
             db_connection=None,
             write_to_db=False,
             simulation_params=None,
             output_csv=True):

        self.time_resolution = time_resolution
        self.date_format = date_format
        self.start_date = pd.to_datetime(start_date, format=date_format)
        self.output_file = output_file
        self.nan_representation = nan_rep
        self.df = pd.DataFrame([])
        self.write_to_db = write_to_db
        self.output_csv = output_csv

        # DB connection if enabled
        if self.write_to_db and db_connection:
            self.db_params = db_connection  # Store for later use in refresh thread
            db_url = f"postgresql://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}:{db_connection['port']}/{db_connection['dbname']}"
            self.engine = create_engine(db_url)
            self.conn = self.engine.connect()

            # Insert simulation metadata
            sim_params_json = json.dumps(simulation_params) if simulation_params else '{}'
            query = text("INSERT INTO building_power.simulation_outputs(parameters) VALUES (:params) RETURNING simulation_output_id;")
            result = self.conn.execute(query, {"params": sim_params_json})
            self.simulation_id = result.fetchone()[0]
            self.conn.commit()

        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()

        return self.meta

    def create(self, num, model, buff_size=500, attrs=None):
        if num > 1 or self.eid is not None:
            raise RuntimeError('Can only create one instance of CSVWriter.')
        if attrs:
            self.attrs = ['date']
            self.attrs.extend(attrs)
        self.buff_size = buff_size
        self.eid = 'PostgresWriter'

        return [{'eid': self.eid, 'type': model}]

    def _register_variables(self, var_names):
        """Ensure variables exist in DB and return their IDs in batch."""
        import time
        start_time = time.time()
        
        # First check which variables we already know
        new_vars = [v for v in var_names if v not in self.variable_map]
        
        if self.debug:
            print(f"[_register_variables] Total vars: {len(var_names)}, New vars to register: {len(new_vars)}, Cached vars: {len(var_names) - len(new_vars)}")
        
        if not new_vars:
            return [self.variable_map[v] for v in var_names]
        
        # Insert variables one by one (batch insert with RETURNING doesn't work well in SQLAlchemy)
        insert_start = time.time()
        for i, var_name in enumerate(new_vars):
            extra_info = parse_variable_name(var_name)
            query = text("""
                INSERT INTO building_power.variable(
                    simulation_output_id, 
                    variable_name, 
                    unit, 
                    extra_info
                ) VALUES (:sim_id, :var_name, :unit, :extra_info)
                RETURNING variable_id, variable_name;
            """)
            
            result = self.conn.execute(query, {
                "sim_id": self.simulation_id,
                "var_name": var_name,
                "unit": extra_info.get('unit'),
                "extra_info": json.dumps(extra_info)
            })
            
            var_id, var_name_ret = result.fetchone()
            self.variable_map[var_name_ret] = var_id
            
            if self.debug and (i + 1) % 100 == 0:
                print(f"  [_register_variables] Inserted {i + 1}/{len(new_vars)} variables...")
        
        if self.debug:
            print(f"  [_register_variables] Insert time: {time.time() - insert_start:.2f}s")
        
        commit_start = time.time()
        self.conn.commit()
        
        if self.debug:
            print(f"  [_register_variables] Commit time: {time.time() - commit_start:.2f}s")
            print(f"[_register_variables] Total time: {time.time() - start_time:.2f}s, Cache now has {len(self.variable_map)} variables")
        
        return [self.variable_map[v] for v in var_names]

    def _flush_to_db(self, df):
        """Insert DataFrame rows into timeseries hypertable using batch processing."""
        import time
        start_time = time.time()
        
        # Extract step_index from DataFrame if present
        if 'step_index' in df.columns:
            step_index = df['step_index'].iloc[0] if len(df) > 0 else None
            # Remove step_index from columns to process
            df = df.drop(columns=['step_index'])
        else:
            step_index = None
        
        # Get all unique variable names in this batch
        var_names = list(df.columns)
        if self.debug:
            print(f"[_flush_to_db] Starting flush for {len(df)} rows x {len(var_names)} variables = {len(df) * len(var_names)} potential records (step_index={step_index})")
        
        # Batch register variables and get their IDs
        reg_start = time.time()
        var_ids = self._register_variables(var_names)
        var_id_map = dict(zip(var_names, var_ids))
        if self.debug:
            print(f"  [_flush_to_db] Variable registration took {time.time() - reg_start:.2f}s")
        
        # Prepare timeseries records
        prep_start = time.time()
        records = []
        for ts, row in df.iterrows():
            for col, value in row.items():
                if pd.isna(value):
                    continue
                records.append({
                    "variable_id": var_id_map[col],
                    "ts": ts,
                    "quantity": float(value),
                    "step_index": step_index
                })
        
        if self.debug:
            print(f"  [_flush_to_db] Prepared {len(records)} non-null records in {time.time() - prep_start:.2f}s")
        
        if records:
            # Insert records using bulk insert for better performance
            insert_start = time.time()
            
            if self.debug:
                print(f"  [_flush_to_db] Bulk inserting {len(records)} records...")
            
            # Get raw DBAPI connection for execute_values
            raw_conn = self.conn.connection
            cursor = raw_conn.cursor()
            
            # Convert records to tuples for psycopg2 execute_values
            values = [(r['variable_id'], r['ts'], r['quantity'], int(r['step_index'])) for r in records]
            
            # Use execute_values for maximum performance (batched multi-row INSERT)
            from psycopg2.extras import execute_values
            execute_values(
                cursor,
                "INSERT INTO building_power.output_timeseries (variable_id, ts, quantity, step_index) VALUES %s",
                values,
                page_size=1000  # Insert 1000 rows at a time
            )
            cursor.close()
            
            # IMPORTANT: Commit on the raw connection since we used the raw cursor
            raw_conn.commit()
            
            if self.debug:
                print(f"  [_flush_to_db] Bulk insert time: {time.time() - insert_start:.2f}s ({len(records)/max(0.001, time.time() - insert_start):.0f} records/sec)")
        
        commit_start = time.time()
        self.conn.commit()
        if self.debug:
            print(f"  [_flush_to_db] Commit time: {time.time() - commit_start:.2f}s")
            print(f"[_flush_to_db] Total flush time: {time.time() - start_time:.2f}s")
            print()

    def _flush_worker(self):
        """Background thread that flushes queued DataFrames to DB."""
        first_flush = True
        flush_count = 0
        import time
        
        while not self.stop_event.is_set() or not self.flush_queue.empty():
            try:
                df = self.flush_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if df is None:
                # Signal to exit
                if self.debug:
                    print(f"[_flush_worker] Received exit signal. Total flushes processed: {flush_count}")
                break

            flush_count += 1
            worker_start = time.time()
            if self.debug:
                print(f"\n[_flush_worker] Flush #{flush_count} - Queue size: {self.flush_queue.qsize()}")

            if self.output_csv:
                csv_start = time.time()
                df.to_csv(self.output_file, mode='w' if first_flush else 'a', header=first_flush,
                          date_format=self.date_format, na_rep=self.nan_representation)
                if self.debug:
                    print(f"  [_flush_worker] CSV write time: {time.time() - csv_start:.2f}s")
                first_flush = False

            if self.write_to_db:
                self._flush_to_db(df)

            if self.debug:
                print(f"[_flush_worker] Flush #{flush_count} total time: {time.time() - worker_start:.2f}s")
            self.flush_queue.task_done()

    def step(self, time, inputs, max_advance):
        import time as time_module
        step_start = time_module.time()
        
        current_date = (self.start_date
                        + pd.Timedelta(time * self.time_resolution, unit='seconds'))

        data_dict = {'date': current_date}
        for attr, values in inputs.get(self.eid, {}).items():
            for src, value in values.items():
                data_dict[f'{src}-{attr}'] = [value]
        
        num_inputs = len(inputs.get(self.eid, {}))
        num_columns = len(data_dict) - 1  # Exclude 'date' column
        
        if self.attrs:
            df_data = pd.DataFrame(data_dict, columns=self.attrs)
            df_data.set_index('date', inplace=True)
        else:
            self.attrs = list(data_dict.keys())
            df_data = pd.DataFrame(data_dict, columns=self.attrs)
            df_data.set_index('date', inplace=True)

        if time == 0:
            if self.debug:
                print(f"[step] t={time}: First step - storing initial dataframe with {num_columns} columns (step_index={self.step_index})")
            self.df = df_data            
        elif time > 0:
            queue_before = self.flush_queue.qsize()
            # Add step_index as a column to the DataFrame
            df_with_step = df_data.copy()
            df_with_step['step_index'] = self.step_index
            self.flush_queue.put(df_with_step)
            queue_after = self.flush_queue.qsize()
            if self.debug:
                print(f"[step] t={time}: Added batch to queue (columns={num_columns}, queue: {queue_before}->{queue_after}, step_index={self.step_index})")
            self.df = pd.concat([self.df, df_data])
        
        # Increment step index for next call
        self.step_index += 1
        
        step_time = time_module.time() - step_start
        if self.debug and (time % 10 == 0 or step_time > 0.1):  # Print every 10 steps or if step takes >100ms
            print(f"[step] t={time}: Step completed in {step_time:.3f}s (inputs={num_inputs}, columns={num_columns})")
                        
        return None

    def _refresh_continuous_aggregate(self):
        """Background thread to refresh continuous aggregate after simulation ends."""
        import time
        import psycopg2
        
        try:
            if self.debug:
                print(f"[_refresh_continuous_aggregate] Starting continuous aggregate refresh...")
            
            # Create a completely fresh psycopg2 connection with autocommit
            raw_conn = psycopg2.connect(
                host=self.db_params['host'],
                port=self.db_params['port'],
                dbname=self.db_params['dbname'],
                user=self.db_params['user'],
                password=self.db_params['password']
            )
            raw_conn.autocommit = True
            
            refresh_start = time.time()
            
            # Get min and max timestamps from this simulation
            cursor = raw_conn.cursor()
            cursor.execute("""
                SELECT 
                    MIN(ts) as min_ts,
                    MAX(ts) as max_ts
                FROM building_power.output_timeseries ot
                JOIN building_power.variable v ON v.variable_id = ot.variable_id
                WHERE v.simulation_output_id = %s
            """, (self.simulation_id,))
            
            result = cursor.fetchone()
            min_ts, max_ts = result[0], result[1]
            
            if min_ts is None or max_ts is None:
                if self.debug:
                    print(f"[_refresh_continuous_aggregate] No data found for simulation {self.simulation_id}, skipping refresh")
                cursor.close()
                raw_conn.close()
                return
            
            if self.debug:
                print(f"[_refresh_continuous_aggregate] Refreshing from {min_ts} to {max_ts}")
            
            # Call refresh_continuous_aggregate
            cursor.execute(f"""
                CALL refresh_continuous_aggregate(
                    'building_power.power_15min_by_variable',
                    '{min_ts}'::timestamptz,
                    '{max_ts}'::timestamptz
                )
            """)
            
            if self.debug:
                print(f"[_refresh_continuous_aggregate] Refresh completed in {time.time() - refresh_start:.2f}s")
            
            cursor.close()
            raw_conn.close()
                
        except Exception as e:
            print(f"[_refresh_continuous_aggregate] Error during refresh: {e}")
            import traceback
            traceback.print_exc()

    def finalize(self):
        import time
        finalize_start = time.time()
        queue_size = self.flush_queue.qsize()
        if self.debug:
            print(f"\n[finalize] Starting finalization. Queue size: {queue_size} batches waiting to be flushed")

        # Signal flush thread to stop
        self.stop_event.set()
        self.flush_queue.put(None)  # Sentinel
        
        if self.debug:
            print(f"[finalize] Waiting for flush thread to complete {queue_size} remaining batches...")
        self.flush_thread.join()
        if self.debug:
            print(f"[finalize] Flush thread completed all {queue_size} batches in {time.time() - finalize_start:.2f}s")

        # Start continuous aggregate refresh in background thread (daemon=True allows program to exit)
        if self.write_to_db and self.engine:
            refresh_thread = threading.Thread(target=self._refresh_continuous_aggregate, daemon=False)
            refresh_thread.start()
            if self.debug:
                print(f"[finalize] Continuous aggregate refresh started in background thread (non-daemon, will complete even if program tries to exit)")

        if self.write_to_db and self.conn:
            close_start = time.time()
            self.conn.close()
            if self.debug:
                print(f"[finalize] Connection closed in {time.time() - close_start:.2f}s")
        
        if self.debug:
            print(f"[finalize] Total finalization time: {time.time() - finalize_start:.2f}s")


if __name__ == '__main__':
    mosaik_api.start_simulation(PostgresWriterModel())
