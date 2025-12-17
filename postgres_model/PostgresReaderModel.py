"""
Power Consumption Model Simulator for Mosaik
--------------------------------------------
This simulator integrates with the Mosaik co-simulation framework to provide 
time-based power consumption data from a PostgreSQL database (via psycopg2).
It supports retrieving time series for multiple entities based on SQL queries,
returning the last valid measurement at or before the current simulation time.

Main features:
- Time-based simulation model (Mosaik API v3)
- Fetches data from PostgreSQL and stores it in pandas DataFrames
- Uses `.asof()` to obtain the most recent value for each attribute
- Configurable simulation start/end times, time resolution, and time step
- Designed for use with pre-defined SQL queries and dynamic entity creation

SQL Query Requirements:
------------------------
The `create()` method expects a `query` parameter (string) for each entity to 
be created. This SQL query must:
1. Return a time series with a column named `bucket` (timestamp) which will be
   used as the DataFrame index for time alignment.
2. Return one or more numeric columns matching the attributes defined in 
   `self.attrs` (e.g., `power_kw`).
3. Be self-contained — the simulator will execute it directly without adding 
   filters.

Example query:
--------------
    interval_str = '15 minutes'
    bldg_ids = '101, 102, 103'
    sql_query = f\"\"\"
        SELECT time_bucket(INTERVAL '{interval_str}', sample_time) AS bucket,
               SUM(electricity_total_energy_consumption) / 0.25 AS power_kw
        FROM building_power.building_power
        WHERE bldg_id IN ({bldg_ids})
          AND sample_time >= '2025-01-01 00:00:00'
          AND sample_time < '2025-02-01 00:00:00'
          AND electricity_total_energy_consumption IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    \"\"\"

Example usage in Mosaik scenario script:
----------------------------------------
    sim_config = {
        'PostgresReaderModel': {
            'cmd': 'python power_consumption_sim.py',
        }
    }

    world = mosaik.World(sim_config)

    db_params = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'my_db',
        'user': 'postgres',
        'password': 'secret',
    }

    power_sim = world.start(
        'PostgresReaderModel',
        time_resolution=900,      # in seconds (15 minutes)
        time_step=900,
        sim_start='2025-01-01 00:00:00',
        sim_end='2025-01-02 00:00:00',
        db_connection=db_params,
        sql_rows=['power_kw'],
        date_format='%Y-%m-%d %H:%M:%S',
    )

    power_entities = power_sim.PowerConsumption(
        num=1,
        model='PowerConsumption',
        query=[sql_query]
    )

    # Connect to another simulator or collector
    # world.connect(...)

Note:
- The `bucket` column in the query **must** be a timestamp and unique per row.
- The simulator raises an error if a requested attribute is missing from the query result.
- Requires psycopg2, pandas, and mosaik_api_v3.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

import mosaik_api_v3 as mosaik_api
from typing_extensions import override

from mosaik_api_v3.types import (
        CreateResult,
        EntityId,
        InputData,
        Meta,
        ModelDescription,
        OutputData,
        OutputRequest,
        Time,
)

DATE_FORMAT = 'YYYY-MM-DD HH:mm:ss'

SENTINEL = object()

# META: Meta = {
#     "api_version": "3.0",
#     "type": "time-based",
#     "models": {
#         "SolarIrradiation": {
#             "attrs": [],
#             "public": True,
#             "params": ['building_ids'],
#             "trigger": [],
#             "persistent": ['P[MW]'],
#             "non-trigger": [],
#             "non-persistent": [],
#         }
#     },
# }

class PostgresReaderModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__({'models': {}})
        self.start_date = None
        self.end_date = None
        self.current_date = None
        self.attrs = None
        self.sid = None
        self.eid = None
        self.eid_prefix = 'PostgresReader'
        self.eids = []        
        self.type = None
        self.time_res = None 
        self.entities = {}
        self.conn = None
        self.start_date = None
        self.results = []
        self.model_name = 'PostgresReader'
    
    @override    
    def init(self, sid, time_resolution, time_step, sim_start, sim_end, db_connection, sql_rows, date_format=None, type="time-based"):
        self.type = type        
        if self.type != "time-based":
            print("This simulator type is always time-based")

        self.sid = sid        
        self.time_step = time_step        
        self.time_res = pd.Timedelta(float(time_resolution), unit='seconds')
        self.start_date = pd.to_datetime(sim_start, format=date_format).tz_localize('UTC')
        self.end_date = pd.to_datetime(sim_end, format=date_format).tz_localize('UTC')
        self.current_date = self.start_date        
        self.attrs = sql_rows

        self.meta['type'] = self.type

        self.meta['models'][self.model_name] = {
            'public': True,
            'params': ['query', 'Index'],
            'attrs': self.attrs,
        } 
        # Create SQLAlchemy engine
        db_url = f"postgresql://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}:{db_connection['port']}/{db_connection['dbname']}"
        self.engine = create_engine(db_url)
        self.conn = self.engine.connect()
        return self.meta

    @override
    def create(self, num, model, **params):
        if model != self.eid_prefix:
            raise ValueError('Invalid model "%s"' % model)

        # Ensure all params are lists of size num or a single item if num = 1
        for key, value in params.items():
            if num == 1 and not isinstance(value, list):
                params[key] = [value]  # Wrap single item in a list
            elif not isinstance(value, list) or len(value) != num:
                raise ValueError(f'Parameter "{key}" must be a list of size {num}')
        
        entities = []
        for i in range(len(params['query'])):
            #execute query to get data
            query = text(params['query'][i])
            result = self.conn.execute(query)
            rows = result.fetchall()
            columns = result.keys()
            # Create DataFrame
            if params.get('Index'):
                eid = f'{self.eid_prefix}_{params['Index'][i]}' 
            else:
                eid = f'{self.eid_prefix}_{len(self.entities)+i}'
            df = pd.DataFrame(rows, columns=columns)
            df.set_index('bucket', inplace=True)  # Optional: make time index
            self.entities[eid] = {
                'df': df,
                'current_time': 0
            }
            entities.append({'eid': eid, 'type': model})
        return entities
        

    def step(self, time, inputs, max_advance):
        results = {}
        self.current_date = self.start_date + timedelta(seconds=time)
        return time + self.time_step

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid not in self.entities:
                raise ValueError(f'Unknown entity ID "{eid}"')
            data[eid] = {}
            for attr in attrs:
                if attr in self.entities[eid]['df'].columns:
                    if not self.entities[eid]['df'].empty:
                        val = self.entities[eid]['df'][attr].asof(self.current_date)
                        data[eid][attr] = val if pd.notna(val) else 0.0
                    else:
                        data[eid][attr] = 0.0             
                else:
                    raise ValueError(f'Unknown attribute "{attr}" for entity "{eid}"')
        return data

def main():
    return mosaik_api.start_simulation(PostgresReaderModel(), 'postgres-reader-model simulator')


if __name__ == '__main__':
    main()
