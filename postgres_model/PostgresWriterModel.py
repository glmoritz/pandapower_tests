import pandas as pd
import mosaik_api_v3 as mosaik_api
import psycopg2
from psycopg2.extras import execute_values
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
        self.conn = None
        self.cur = None
        self.write_to_db = False
        self.simulation_id = None
        self.variable_map = {}  # maps var name -> variable_id

        self.output_csv = True

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
            self.conn = psycopg2.connect(**db_connection)
            self.cur = self.conn.cursor()

            # Insert simulation metadata
            sim_params_json = json.dumps(simulation_params) if simulation_params else '{}'
            self.cur.execute(
                "INSERT INTO building_power.simulation_outputs(parameters) VALUES (%s) RETURNING simulation_output_id;",
                (sim_params_json,)
            )
            self.simulation_id = self.cur.fetchone()[0]
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
        # First check which variables we already know
        new_vars = [v for v in var_names if v not in self.variable_map]
        
        if not new_vars:
            return [self.variable_map[v] for v in var_names]
        
        # Prepare batch data
        var_data = []
        for var_name in new_vars:
            
            extra_info = parse_variable_name(var_name)            
            var_data.append((self.simulation_id, var_name, extra_info.get('unit'), json.dumps(extra_info)))

        # Solution 1: Disable pagination in execute_values
        execute_values(
            self.cur,
            """INSERT INTO building_power.variable(
                simulation_output_id, 
                variable_name, 
                unit, 
                extra_info
            ) VALUES %s RETURNING variable_id, variable_name;""",
            var_data,
            page_size=len(var_data)  # Set page_size to total records to get all results
        )
        
        # Fetch results and update cache
        results = self.cur.fetchall()
        for var_id, var_name in results:
            self.variable_map[var_name] = var_id
        
        self.conn.commit()
        
        # For any vars that weren't inserted (due to concurrent inserts), get their existing IDs
        missing_vars = set(new_vars) - set(self.variable_map.keys())
        if missing_vars:
            self.cur.execute("""
                SELECT variable_id, variable_name 
                FROM building_power.variable
                WHERE variable_name = ANY(%s)
            """, (list(missing_vars),))
            
            for var_id, var_name in self.cur.fetchall():
                self.variable_map[var_name] = var_id
        
        return [self.variable_map[v] for v in var_names]

    def _flush_to_db(self, df):
        """Insert DataFrame rows into timeseries hypertable using batch processing."""
        # Get all unique variable names in this batch
        var_names = list(df.columns)
        
        # Batch register variables and get their IDs
        var_ids = self._register_variables(var_names)
        var_id_map = dict(zip(var_names, var_ids))
        
        # Prepare timeseries records
        records = []
        for ts, row in df.iterrows():
            for col, value in row.items():
                if pd.isna(value):
                    continue
                records.append((var_id_map[col], ts, float(value)))
        
        if records:
            execute_values(
                self.cur,
                "INSERT INTO building_power.output_timeseries (variable_id, ts, quantity) VALUES %s;",
                records
            )
        
        self.conn.commit()

    def _flush_worker(self):
        """Background thread that flushes queued DataFrames to DB."""
        first_flush = True        
        while not self.stop_event.is_set() or not self.flush_queue.empty():
            try:
                df = self.flush_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if df is None:
                # Signal to exit
                break

            if self.output_csv:
                df.to_csv(self.output_file, mode='w' if first_flush else 'a', header=first_flush,
                          date_format=self.date_format, na_rep=self.nan_representation)
                first_flush = False

            if self.write_to_db:
                self._flush_to_db(df)

            self.flush_queue.task_done()

    def step(self, time, inputs, max_advance):
        current_date = (self.start_date
                        + pd.Timedelta(time * self.time_resolution, unit='seconds'))

        data_dict = {'date': current_date}
        for attr, values in inputs.get(self.eid, {}).items():
            for src, value in values.items():
                data_dict[f'{src}-{attr}'] = [value]
        if self.attrs:
            df_data = pd.DataFrame(data_dict, columns=self.attrs)
            df_data.set_index('date', inplace=True)
        else:
            self.attrs = list(data_dict.keys())
            df_data = pd.DataFrame(data_dict, columns=self.attrs)
            df_data.set_index('date', inplace=True)

        if time == 0:
            self.df = df_data            
        elif time > 0:
            self.flush_queue.put(df_data.copy())
            self.df = pd.concat([self.df, df_data])                       
            
                        

        return None

    def finalize(self):      

        # Signal flush thread to stop
        self.stop_event.set()
        self.flush_queue.put(None)  # Sentinel
        self.flush_thread.join()

        if self.write_to_db and self.conn:
            self.cur.close()
            self.conn.close()


if __name__ == '__main__':
    mosaik_api.start_simulation(PostgresWriterModel())
