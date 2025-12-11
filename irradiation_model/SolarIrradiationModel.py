from matplotlib.pylab import SeedSequence
import pandas as pd
import pvlib
import numpy as np
from datetime import datetime, timedelta

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

META: Meta = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "SolarIrradiation": {
            "public": True,
            "params": ['latitude', 'longitude', 'master_seed_sequence'],  
            "trigger": [],
            "persistent": ['DNI[W/m2]', 'cloudiness', 'cloud_state'],
            "non-trigger": [],
            "non-persistent": [],
        }
    },
}

class SolarIrradiationModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.start_date = None
        self.attrs = None
        self.sid = None
        self.eid = None
        self.eid_prefix = 'SolarIrradiation'
        self.eids = []        
        self.type = None
        self.time_res = None 
        self.entities = {}
        self.results = {}
        self.master_seed_sequence = None  # Will be set during init

        self.start_date = None
        self.cloud_states = [0, 1, 2]
        self.cloudiness_values = [0.0, 0.4, 0.8]
        self.transition_matrix = [
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8]
        ]
        self.results = []
    
    @override    
    def init(self, sid, time_resolution, time_step, sim_start, date_format=None, type="time-based", master_seed_sequence=None):
        self.type = type
        self.time_step = time_step
        if self.type != "time-based":
            print("This simulator type is always time-based")
        self.sid = sid
        self.time_res = pd.Timedelta(time_resolution, unit='seconds')
        self.start_date = pd.to_datetime(sim_start, format=date_format)
        
        # Initialize RNG with provided seed for reproducibility
        if master_seed_sequence is not None:
            self.master_seed_sequence = master_seed_sequence            
        else:
            self.master_seed_sequence = np.random.SeedSequence() 
            print("WARNING: SolarIrradiationModel initialized without seed - results will not be reproducible!")        
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
        for i in range(num):
            eid = f'{self.eid_prefix}_{len(self.entities)+i}'
            #this will guarantee independent RNG streams per entity
            key_int = abs(hash(eid)) % (2**32)
            entity_ss = SeedSequence(
                entropy=self.master_seed_sequence.entropy,
                spawn_key=(key_int,)
            )
            entity_rng = np.random.default_rng(entity_ss)
            self.entities[eid] = {
                'location': pvlib.location.Location(params['latitude'][i], params['longitude'][i]),                
                'entity_seed_sequence': entity_ss,
                'rng': entity_rng,  # Independent RNG per entity
                'cloud_state': entity_rng.choice(self.cloud_states),  # Use entity-specific RNG
                'time': 0
            }
            entities.append({'eid': eid, 'type': model})
        return entities

    def step(self, time, inputs, max_advance):
        
        results = {}

        for eid, entity in self.entities.items():            
            last_step = self.results[eid]['time_elapsed'] if (eid in self.results and 'time_elapsed' in self.results[eid]) else 0
            current_step_size = time - last_step

            dt = self.start_date + timedelta(seconds=time)

            # Determine how many base steps have passed            
            n_steps = max(1, int(current_step_size // self.time_step))

            # Raise transition matrix to power of n_steps
            P_n = np.linalg.matrix_power(self.transition_matrix, n_steps)

            location = entity['location']
            times = pd.date_range(start=dt, periods=1, freq=f'{str(self.time_step)}s', tz='UTC')
            cs = location.get_clearsky(times)
            ghi_clear = cs['ghi'].iloc[0]

            # Markov cloudiness model with n-step transition
            current_state = entity['cloud_state']
            next_state = self.entities[eid]['rng'].choice(self.cloud_states, p=P_n[current_state])  
            entity['cloud_state'] = next_state
            cloudiness = self.cloudiness_values[next_state]

            # Attenuated irradiance
            irradiance = ghi_clear * (1 - cloudiness)

            # Store or return results (if needed)
            results[eid] = {
                'DNI[W/m2]': irradiance,
                'cloud_state': next_state,
                'timestamp': dt,                
                'time_elapsed': time,                
            }

        self.results = results
        return time+self.time_step 

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid not in self.results:
                raise ValueError('Unknown entity ID "%s"' % eid)
            data[eid] = {}
            for attr in attrs:
                if isinstance(self.results[eid][attr], np.floating) or isinstance(self.results[eid][attr], np.integer):
                    data[eid][attr] = self.results[eid][attr].tolist()
                else:
                    data[eid][attr] = str(self.results[eid][attr])  

        return data


def main():
    return mosaik_api.start_simulation(SolarIrradiationModel(), 'solar-irradiation-model simulator')


if __name__ == '__main__':
    main()
