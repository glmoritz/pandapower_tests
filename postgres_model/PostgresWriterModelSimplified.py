"""
PostgresWriterModel with Bucket Aggregation
--------------------------------------------
This is the simplified version that:
1. Aggregates data into fixed-size buckets using time-weighted averaging
2. Writes to a simple flat table (no TimescaleDB continuous aggregates)
3. Uses bulk COPY for maximum insert performance

The key difference from the original: data is homogenized in Python before 
being written to the database, eliminating the need for complex time-weighted
queries later.
"""

import pandas as pd
import mosaik_api_v3 as mosaik_api
from sqlalchemy import create_engine, text
import json
import threading
import queue
import re
from datetime import datetime
from io import StringIO

from .BucketAggregator import BucketAggregator

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
    """Parse variable name and return structured data."""
    # Get the rightmost part after any dot
    if '.' in var_name:
        parts = var_name.split('.')        
        element_part = parts[-1]
    else:        
        element_part = var_name
    
    # Extract element and output parts
    element_parts = element_part.split('-')
    
    # The output with unit is always the last part
    output_with_unit = element_parts[-1]
    
    # The element info is in the earlier parts
    element_info = '-'.join(element_parts[:-1]) if len(element_parts) > 1 else ""
    
    # Extract element type and index
    element_type = None
    element_index = None
    
    if element_info:
        match = re.search(r'([A-Za-z]+)[_-](.+)', element_info)
        if match:
            element_type = match.group(1)
            element_index = match.group(2)
        else:
            element_type = element_info
            element_index = None
    
    # Extract unit if present
    unit = None
    if '[' in output_with_unit and ']' in output_with_unit:
        unit_match = re.search(r'\[([^\]]+)\]', output_with_unit)
        if unit_match:
            unit = unit_match.group(1)
    
    return {
        'element': element_info,
        'element_type': element_type.lower() if element_type else None,
        'element_index': element_index,
        'output': output_with_unit,
        'unit': unit,        
    }


class PostgresWriterModelSimplified(mosaik_api.Simulator):
    """
    Simplified PostgresWriter that pre-aggregates data into fixed time buckets.
    """
    
    def __init__(self):
        super().__init__(META)
        self.eid = None
        self.time_resolution = None
        self.date_format = None
        self.start_date = None
        self.output_file = None
        self.buff_size = None
        self.attrs = None
        self.nan_representation = None

        # DB related
        self.engine = None
        self.conn = None
        self.write_to_db = False
        self.simulation_id = None
        self.db_params = None
        
        # Variable ID cache (normalized schema)
        # Key: (element_type, element_index, variable_name) -> variable_id
        self.variable_ids = {}
        
        # Bucket aggregator
        self.bucket_aggregator = None
        self.bucket_size_s = 900  # 15 minutes default

        self.output_csv = True
        self.debug = False

        # Threading
        self.flush_queue = queue.Queue()
        self.flush_thread = None
        self.stop_event = threading.Event()
        
        # CSV tracking
        self.csv_first_flush = True

    def init(self, sid, time_resolution, start_date,
             date_format='%Y-%m-%d %H:%M:%S',
             output_file='results.csv',
             nan_rep='NaN',
             db_connection=None,
             write_to_db=False,
             simulation_params=None,
             output_csv=True,
             bucket_size_s=900):  # New parameter

        self.time_resolution = time_resolution
        self.date_format = date_format
        self.start_date = pd.to_datetime(start_date, format=date_format)
        self.output_file = output_file
        self.nan_representation = nan_rep
        self.write_to_db = write_to_db
        self.output_csv = output_csv
        self.bucket_size_s = bucket_size_s
        
        # Initialize bucket aggregator
        self.bucket_aggregator = BucketAggregator(
            bucket_size_s=bucket_size_s,
            start_datetime=self.start_date
        )

        # DB connection if enabled
        if self.write_to_db and db_connection:
            self.db_params = db_connection
            db_url = f"postgresql://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}:{db_connection['port']}/{db_connection['dbname']}"
            self.engine = create_engine(db_url)
            self.conn = self.engine.connect()

            # Ensure simulation_data table exists
            self._ensure_tables_exist()

            # Insert simulation metadata
            sim_params_json = json.dumps(simulation_params) if simulation_params else '{}'
            query = text("""
                INSERT INTO building_power.simulation_outputs(parameters, bucket_size_seconds) 
                VALUES (:params, :bucket_size) 
                RETURNING simulation_output_id;
            """)
            result = self.conn.execute(query, {
                "params": sim_params_json,
                "bucket_size": bucket_size_s
            })
            self.simulation_id = result.fetchone()[0]
            self.conn.commit()
            
            if self.debug:
                print(f"[init] Created simulation_output_id={self.simulation_id} with bucket_size={bucket_size_s}s")

        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()

        return self.meta

    def create(self, num, model, buff_size=500, attrs=None):
        if num > 1 or self.eid is not None:
            raise RuntimeError('Can only create one instance of PostgresWriterModel.')
        if attrs:
            self.attrs = ['date']
            self.attrs.extend(attrs)
        self.buff_size = buff_size
        self.eid = 'PostgresWriter'

        return [{'eid': self.eid, 'type': model}]

    def _ensure_tables_exist(self):
        """Create normalized schema tables if they don't exist."""
        # Add bucket_size_seconds column to simulation_outputs if not exists
        self.conn.execute(text("""
            ALTER TABLE building_power.simulation_outputs 
            ADD COLUMN IF NOT EXISTS bucket_size_seconds INT DEFAULT 900;
        """))
        
        # Create simulation_variable table (variable definitions)
        # element_index uses empty string instead of NULL for simpler unique constraint
        self.conn.execute(text("""
            CREATE TABLE IF NOT EXISTS building_power.simulation_variable (
                variable_id SERIAL PRIMARY KEY,
                simulation_id INT NOT NULL,
                element_type TEXT NOT NULL,
                element_index TEXT NOT NULL DEFAULT '',
                variable_name TEXT NOT NULL,
                unit TEXT,
                UNIQUE (simulation_id, element_type, element_index, variable_name)
            );
        """))
        
        # Create simulation_timeseries table (just variable_id + bucket + value)
        self.conn.execute(text("""
            CREATE TABLE IF NOT EXISTS building_power.simulation_timeseries (
                variable_id INT NOT NULL,
                bucket TIMESTAMPTZ NOT NULL,
                value FLOAT8 NOT NULL,
                PRIMARY KEY (variable_id, bucket)
            );
        """))
        
        # Create indexes
        self.conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_simulation_variable_sim 
                ON building_power.simulation_variable(simulation_id);
        """))
        self.conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_simulation_variable_element 
                ON building_power.simulation_variable(simulation_id, element_type, element_index);
        """))
        self.conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_simulation_timeseries_bucket 
                ON building_power.simulation_timeseries(bucket);
        """))
        
        self.conn.commit()

    def set_pandapower_grid_id(self, grid_id):
        """Update the simulation record with the pandapower grid ID."""
        if not self.write_to_db or not self.db_params or self.simulation_id is None:
            return
        
        db_url = f"postgresql://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
        engine = create_engine(db_url)
        with engine.connect() as conn:
            query = text("""UPDATE building_power.simulation_outputs
                            SET pandapower_grid_id = :grid_id
                            WHERE simulation_output_id = :sim_id;""")
            conn.execute(query, {"grid_id": grid_id, "sim_id": self.simulation_id})
            conn.commit()

    def set_simulation_finished(self, converged):
        """Update the simulation record with final status."""
        if not self.write_to_db or not self.db_params or self.simulation_id is None:
            return
        
        db_url = f"postgresql://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
        engine = create_engine(db_url)
        with engine.connect() as conn:
            query = text("""UPDATE building_power.simulation_outputs
                            SET finished_at = NOW(),
                                converged = :converged
                            WHERE simulation_output_id = :sim_id;""")
            conn.execute(query, {            
                "sim_id": self.simulation_id,
                "converged": converged
            })
            conn.commit()

    def _get_or_create_variable_id(self, cursor, element_type, element_index, variable_name, unit):
        """
        Get or create a variable_id for the given variable.
        Uses in-memory cache to avoid repeated DB lookups.
        """
        # Normalize element_index: use empty string instead of None for consistency
        element_index = element_index or ''
        
        # Create cache key
        key = (element_type, element_index, variable_name)
        
        if key in self.variable_ids:
            return self.variable_ids[key]
        
        # Insert or get existing (using ON CONFLICT on the unique index)
        cursor.execute("""
            INSERT INTO building_power.simulation_variable 
                (simulation_id, element_type, element_index, variable_name, unit)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (simulation_id, element_type, element_index, variable_name)
            DO UPDATE SET unit = EXCLUDED.unit
            RETURNING variable_id;
        """, (self.simulation_id, element_type, element_index, variable_name, unit))
        
        variable_id = cursor.fetchone()[0]
        self.variable_ids[key] = variable_id
        return variable_id

    def _flush_to_db(self, records):
        """
        Insert aggregated records using normalized schema.
        
        Args:
            records: List of dicts with bucket, element_type, element_index, variable_name, value, unit
        """
        if not records:
            return
            
        import time
        start_time = time.time()
        
        raw_conn = self.conn.connection
        cursor = raw_conn.cursor()
        cursor.execute("SET search_path TO building_power, public")
        
        # Step 1: Ensure all variables exist and get their IDs
        # Group records by variable to minimize DB calls
        variable_keys = set()
        for r in records:
            key = (r['element_type'], r['element_index'] or '', r['variable_name'], r['unit'])
            variable_keys.add(key)
        
        # Get/create variable IDs for any new variables
        for element_type, element_index, variable_name, unit in variable_keys:
            self._get_or_create_variable_id(
                cursor, element_type, element_index, variable_name, unit
            )
        
        # Step 2: Build timeseries data for COPY (just variable_id, bucket, value)
        output = StringIO()
        for r in records:
            key = (r['element_type'], r['element_index'] or '', r['variable_name'])
            variable_id = self.variable_ids[key]
            line = f"{variable_id}\t{r['bucket'].isoformat()}\t{r['value']}\n"
            output.write(line)
        
        output.seek(0)
        
        # Use COPY for fast bulk insert
        cursor.copy_from(
            output,
            'simulation_timeseries',
            columns=('variable_id', 'bucket', 'value')
        )
        
        cursor.close()
        raw_conn.commit()
        
        if self.debug:
            print(f"  [_flush_to_db] Inserted {len(records)} records in {time.time() - start_time:.3f}s ({len(records)/max(0.001, time.time() - start_time):.0f} records/sec)")

    def _flush_to_csv(self, records):
        """Write records to CSV file."""
        if not records:
            return
            
        df = pd.DataFrame(records)
        df.to_csv(
            self.output_file, 
            mode='w' if self.csv_first_flush else 'a', 
            header=self.csv_first_flush,
            index=False
        )
        self.csv_first_flush = False

    def _flush_worker(self):
        """Background thread that flushes completed buckets to DB/CSV."""
        flush_count = 0
        import time
        
        while not self.stop_event.is_set() or not self.flush_queue.empty():
            try:
                records = self.flush_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if records is None:
                # Signal to exit
                if self.debug:
                    print(f"[_flush_worker] Received exit signal. Total flushes: {flush_count}")
                break

            flush_count += 1
            
            if self.debug:
                print(f"\n[_flush_worker] Flush #{flush_count} - {len(records)} records")

            if self.output_csv:
                self._flush_to_csv(records)

            if self.write_to_db:
                self._flush_to_db(records)

            self.flush_queue.task_done()

    def step(self, time, inputs, max_advance):
        """
        Process inputs and add to bucket aggregator.
        
        When buckets complete, they're queued for background DB insertion.
        """
        import time as time_module
        step_start = time_module.time()
        
        # Calculate actual simulation time in seconds
        time_s = time * self.time_resolution
        
        # Process all inputs
        for attr, values in inputs.get(self.eid, {}).items():
            for src, value in values.items():
                # Parse the source name to extract element info
                var_info = parse_variable_name(f'{src}-{attr}')
                
                # Add to bucket aggregator
                self.bucket_aggregator.add_value(
                    time_s=time_s,
                    element_type=var_info['element_type'] or 'unknown',
                    element_index=var_info['element_index'],
                    variable_name=var_info['output'],
                    value=value,
                    unit=var_info['unit']
                )
        
        # Check for completed buckets and queue them for flush
        completed_records = self.bucket_aggregator.get_completed_buckets()
        if completed_records:
            self.flush_queue.put(completed_records)
            if self.debug:
                print(f"[step] t={time}: Queued {len(completed_records)} records from completed bucket(s)")
        
        step_time = time_module.time() - step_start
        if self.debug and (time % 100 == 0 or step_time > 0.1):
            print(f"[step] t={time}: Step completed in {step_time:.3f}s")
                        
        return None

    def finalize(self):
        """Flush all remaining data and close connections."""
        import time
        finalize_start = time.time()
        
        if self.debug:
            print(f"\n[finalize] Starting finalization...")

        # Flush all remaining buckets from aggregator
        final_records = self.bucket_aggregator.flush_all()
        if final_records:
            self.flush_queue.put(final_records)
            if self.debug:
                print(f"[finalize] Flushed {len(final_records)} final records")

        # Signal flush thread to stop
        self.stop_event.set()
        self.flush_queue.put(None)  # Sentinel
        
        if self.debug:
            print(f"[finalize] Waiting for flush thread to complete...")
        self.flush_thread.join()
        
        if self.debug:
            print(f"[finalize] Flush thread completed")

        if self.write_to_db and self.conn:
            self.conn.close()
            
        if self.debug:
            print(f"[finalize] Total finalization time: {time.time() - finalize_start:.2f}s")


if __name__ == '__main__':
    mosaik_api.start_simulation(PostgresWriterModelSimplified())
