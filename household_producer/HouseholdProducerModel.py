import pandas as pd
import pvlib
import numpy as np
from datetime import datetime, timedelta
import math

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
    "type": "hybrid",
    "models": {
        "HouseholdProducer": {
            "public": True,            
            "params": ['SolarPeakPower_MW', 'StorageCapacity_MWh', 'InitialSOC_percent', 'MaxChargePower_MW', 'MaxDischargePower_MW','InverterType', 'Index'],
            "trigger": ['Irradiance[W/m2]', 'PowerConsumption[MW]', 'GridExportLimit[MW]', 'GridDemand[MW]', 'PowerFactorDemand'],
            "persistent": ['P_a_load[MW]',
                           'P_b_load[MW]',
                           'P_c_load[MW]',
                           'Q_a_load[MVar]',
                           'Q_b_load[MVar]',
                           'Q_c_load[MVar]', 
                           'SOC[MWh]',
                             'SOC[percent]', 
                             'Curtailment[MW]',
                             'CurtailmentEnergy[MWh]',
                             'EnergyExported[MWh]', 
                             'EnergyImported[MWh]', 
                             'PVEnergyGeneration[MWh]',
                             'PVPowerGeneration[MW]',
                             'EnergyConsumption[MWh]',
                             'BatteryEnergyStored[MWh]',
                             'BatteryEnergyConsumed[MWh]',
                             'BatteryPower[MW]'
                             ]
                             ,

            "non-trigger": [],
            "non-persistent": [],
        }
    },
}      


class HouseholdProducerModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.start_date = None
        self.attrs = None
        self.sid = None
        self.eid = None
        self.eid_prefix = 'HouseholdProducer'
        self.eids = []        
        self.type = None
        self.time_res = None 
        self.entities = {}
        self.results = {}
        self.start_date = None
        
    
    @override    
    def init(self, sid, time_resolution, time_step, sim_start, date_format=None, type="hybrid"):
        self.type = type
        self.time_step = time_step
        if self.type != "hybrid":
            print("This simulator type is always hybrid")
            self.type = 'hybrid'
        
        self.sid = sid
        self.time_res = pd.Timedelta(time_resolution, unit='seconds')
        self.start_date = pd.to_datetime(sim_start, format=date_format)
        

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
            eid = f'{self.eid_prefix}_{params.get("Index")[i]}'
            if eid in self.entities:
                raise ValueError(f'Entity ID "{eid}" already exists.')  
            self.entities[eid] = {
                'SOC[MWh]': (params['InitialSOC_percent'][i]/100.0)*params['StorageCapacity_MWh'][i],         
                'SolarPeakPower_MW': params['SolarPeakPower_MW'][i],
                'StorageCapacity_MWh': params['StorageCapacity_MWh'][i],                
                'MaxChargePower_MW': params['MaxChargePower_MW'][i],
                'MaxDischargePower_MW': params['MaxDischargePower_MW'][i],       
                'InverterType': params.get('InverterType', ['3ph'])[i],
                'last_pv_generation_power_mw': 0.0,
                'last_load_power_mw': 0.0,
                'charge_efficiency': 0.95, #tesla powerwall 3
                'discharge_efficiency': 0.90, #tesla powerwall 3
                'last_load_power_mw': 0.0,                
                'time': 0
            }
            entities.append({'eid': eid, 'type': model})
             
        return entities
    
    
    def step(self, time, inputs, max_advance):
        """
        Hybrid step:
            * time  ........ current simulation time  [s]
            * inputs........ nested dict from mosaik
            * max_advance .. upper bound set by mosaik on how far we may jump
        Returns:
            (next_time, next_max_advance)
        """
        results = {}

        for eid, entity in self.entities.items():
            # -----------------------------------------------------------------
            # 1. STATIC PARAMETERS (stored at creation)
            # -----------------------------------------------------------------
            peak_mw      = entity['SolarPeakPower_MW']      # PV peak
            cap_mwh      = entity['StorageCapacity_MWh']     # battery size
            max_chg_mw   = entity['MaxChargePower_MW']      # charge limit
            max_dch_mw   = entity['MaxDischargePower_MW']   # discharge limit

            last_results = self.results.get(eid, {})

            # -----------------------------------------------------------------
            # 2. INPUTS — use last known value if not provided this step
            # -----------------------------------------------------------------
            irr_wm2 = list(inputs.get(eid, {}).get('Irradiance[W/m2]', {}).values()) or [entity.get('irr_wm2', 0.0)]
            irr_wm2 = irr_wm2[0]
            load_mw = list(inputs.get(eid, {}).get('PowerConsumption[MW]', {}).values()) or [entity.get('load_mw', 0.0)]
            load_mw = load_mw[0]

            # Optional grid signals (defaults)
            export_cap_mw = list(inputs.get(eid, {}).get('GridExportLimit[MW]', {}).values()) or [float('inf')]
            export_cap_mw = export_cap_mw[0]
            grid_req_mw   = list(inputs.get(eid, {}).get('GridDemand[MW]', {}).values()) or [0.0]
            grid_req_mw   = max(0.0, grid_req_mw[0])  # negative requests ignored

            # -----------------------------------------------------------------
            # 3. TIME KEEPING
            # -----------------------------------------------------------------
            last_time      = entity.get('last_time', 0)          # [s]
            step_seconds   = max(1, time - last_time)            # avoid zero
            step_hours     = step_seconds / 3600.0
            entity['last_time'] = time

            # -----------------------------------------------------------------
            # 4. PV GENERATION
            # -----------------------------------------------------------------
            pv_mw = (irr_wm2 / 1000.0) * peak_mw                 # simple linear

            # -----------------------------------------------------------------
            # 5. BATTERY & GRID LOGIC
            # -----------------------------------------------------------------
           
                                            
            soc_mwh = entity.get('SOC[MWh]', 0.0)                          # current SOC
            surplus_mw = pv_mw - load_mw                         # +: excess PV

            # ---- CURTAILMENT helpers
            pv_curtail_mw      = 0.0
            charge_power_mw    = 0.0            
            export_to_grid_mw  = 0.0 

            last_charge_power_mw = entity.get('last_charge_power_mw', 0.0)                                 
            soc_mwh += (last_charge_power_mw) * step_hours * (entity['charge_efficiency'] if last_charge_power_mw > 0.0 else 1.0/entity['discharge_efficiency'])  # charging efficiency
            soc_percent = 100.0 * soc_mwh / cap_mwh if cap_mwh > 0 else 0.0            
            minimum_battery_operation_time = 10.0

            # ---- CASE A: PV surplus ------------------------------------------------
            if surplus_mw > 0:
                # 1) grid may explicitly demand power - this is my first priority
                extra_for_grid = 0.0

                if grid_req_mw > 0:
                    # grid wants power, so we try to meet the request
                    if surplus_mw >= grid_req_mw:
                        # enough surplus to meet request
                        extra_for_grid = grid_req_mw
                        surplus_mw -= extra_for_grid
                    else:
                        # not enough surplus, so we export all we have
                        extra_for_grid = surplus_mw
                        surplus_mw = 0.0
                    grid_req_mw -= extra_for_grid  # reduce request

                if grid_req_mw > 0:                    
                    # try to discharge to meet request                                         
                    # Only discharge if battery has enough energy to sustain output for at least minimum_battery_operation_time seconds                    
                    potential_discharge_mw = min(grid_req_mw, max_dch_mw)
                    min_required_soc_mwh = potential_discharge_mw * (minimum_battery_operation_time/3600.0) / entity['discharge_efficiency']                    
                    if soc_mwh > min_required_soc_mwh:
                        charge_power_mw = -1 * potential_discharge_mw
                        grid_req_mw += charge_power_mw  # reduce request (charge_power_mw is negative)

                # 2) if still surplus, we can charge the battery                
                if surplus_mw > 0:
                    # Only charge if we have enough surplus to sustain charging for minimum_battery_operation_time seconds
                    if soc_mwh < cap_mwh:
                        potential_charge_mw = min(surplus_mw, max_chg_mw)
                        # Check if we have enough surplus to maintain this charge rate for minimum_battery_operation_time
                        if potential_charge_mw > 0 and potential_charge_mw * (minimum_battery_operation_time/3600.0) <= surplus_mw * step_hours:
                            charge_power_mw = potential_charge_mw
                            surplus_mw -= charge_power_mw
                
                # 3) export whatever remains, limited by export_cap
                extra_for_grid += surplus_mw
                
                export_to_grid_mw = min(extra_for_grid, export_cap_mw)
                pv_curtail_mw = max(extra_for_grid - export_cap_mw, 0) 

            # ---- CASE B: PV deficit ------------------------------------------------
            else:
                deficit_mw = -surplus_mw  # positive number
                # 1) discharge battery first
                # but Only discharge if battery has enough energy to sustain output for at least minimum_battery_operation_time seconds
                potential_discharge_mw = min(deficit_mw, max_dch_mw)
                min_required_soc_mwh = potential_discharge_mw * (minimum_battery_operation_time/3600.0) / entity['discharge_efficiency']
                
                if soc_mwh > min_required_soc_mwh:
                    charge_power_mw = -1*potential_discharge_mw
                    deficit_mw += charge_power_mw  # charge_power_mw is negative, so it reduces the deficit

                # 2) remaining deficit is imported from grid *only if battery empty*
                export_to_grid_mw = -1*deficit_mw  # may be 0 if fully covered. negative means grid import

            # -----------------------------------------------------------------
            # 7. NEXT INTERNAL EVENT  (battery full / empty)
            # -----------------------------------------------------------------
            next_time     = time + self.time_step                # default
            next_max_adv  = self.time_step                       # default bound
            
            # How fast is the battery (dis)charging right now?                        
            if charge_power_mw > 0 and soc_mwh < cap_mwh:
                #charging
                secs_to_full = 3600.0 * (cap_mwh - soc_mwh) / (charge_power_mw * entity['charge_efficiency'])  # charging efficiency
                next_max_adv = int(min(next_max_adv, secs_to_full))
            elif charge_power_mw < 0 and soc_mwh > 0:
                #discharging
                secs_to_empty = 3600.0 * soc_mwh / (-charge_power_mw / entity['discharge_efficiency'])  # discharging efficiency
                next_max_adv = int(min(next_max_adv, secs_to_empty))

            next_max_adv = max(1, next_max_adv)  # never < 1 s

         
        
            inv_type = entity['InverterType']

            power_factor = (inputs.get(eid, {}).get('PowerFactorDemand', {}).values() or [1.0])[0]

            #distribute the power to the phases
            if power_factor == 0:
                raise ValueError("PowerFactorDemand cannot be zero.")
            else:   
                q_per_p = math.tan(math.acos(power_factor))  # Q/P ratio

                # Determine number of active phases
                if inv_type == '1ph':
                    phases = ['a']
                elif inv_type == '2ph':
                    phases = ['a', 'b']
                elif inv_type == '3ph':
                    phases = ['a', 'b', 'c']
                else:
                    raise ValueError("Invalid InverterType. Must be '1ph', '2ph', or '3ph'.")

            n = len(phases)
            p_per_phase = -1*export_to_grid_mw / n
            q_per_phase = p_per_phase * q_per_p            

            
            step_results = {}

            for phase in ['a', 'b', 'c']:
                step_results[f'P_{phase}_load[MW]'] = p_per_phase if phase in phases else 0.0
                step_results[f'Q_{phase}_load[MVar]'] = q_per_phase if phase in phases else 0.0       

            #before we store the results, we update the energy values with the last step values
            grid_exported_mwh   = last_results.get('EnergyExported[MWh]', 0.0)
            grid_imported_mwh   = last_results.get('EnergyImported[MWh]', 0.0)
            battery_energy_storage_mwh = last_results.get('BatteryEnergyStored[MWh]', 0.0)
            battery_energy_consumption_mwh = last_results.get('BatteryEnergyConsumed[MWh]', 0.0)

            for phase in ['a', 'b', 'c']:
                if step_results[f'P_{phase}_load[MW]'] < 0:
                    grid_imported_mwh += ((-1)*step_results[f'P_{phase}_load[MW]'] if phase in phases else 0.0) * step_hours
                else:
                    grid_exported_mwh += (step_results[f'P_{phase}_load[MW]'] if phase in phases else 0.0) * step_hours
                    

            #update the energy values with the current step values (trapezoidal integration)
            cultailment_energy_mwh   = last_results.get('CurtailmentEnergy[MWh]', 0.0) + entity.get('last_curtailment_power_mw', 0.0) * step_hours
            pv_generation_mwh   = last_results.get('PVEnergyGeneration[MWh]', 0.0) + entity.get('last_pv_generation_power_mw', 0.0) * step_hours
            energy_consumption_mwh   = last_results.get('EnergyConsumption[MWh]', 0.0) + entity.get('last_load_power_mw', 0.0) * step_hours

            if entity.get('last_charge_power_mw', 0.0) > 0:
                battery_energy_storage_mwh = last_results.get('BatteryEnergyStored[MWh]', 0.0) + (entity.get('last_charge_power_mw', 0.0)) * step_hours
            elif last_charge_power_mw < 0:
                battery_energy_consumption_mwh = last_results.get('BatteryEnergyConsumed[MWh]', 0.0) + (-1)*entity.get('last_charge_power_mw', 0.0) * step_hours

            # -----------------------------------------------------------------
            # 8. STORE EVERYTHING WE NEED NEXT STEP
            # -----------------------------------------------------------------
            entity.update({                
                'irr_wm2': irr_wm2,                
                'last_charge_power_mw': charge_power_mw,
                'last_curtailment_power_mw': pv_curtail_mw,
                'last_pv_generation_power_mw': pv_mw,
                'last_load_power_mw': load_mw,
                'SOC[MWh]': soc_mwh
            })            



            step_results.update({                
                'SOC[MWh]':        soc_mwh,
                'SOC[percent]':          soc_percent,
                'Curtailment[MW]': pv_curtail_mw,                
                'EnergyExported[MWh]': grid_exported_mwh,
                'EnergyImported[MWh]': grid_imported_mwh, 
                'CurtailmentEnergy[MWh]':cultailment_energy_mwh,
                'PVEnergyGeneration[MWh]': pv_generation_mwh,
                'EnergyConsumption[MWh]': energy_consumption_mwh,
                'BatteryEnergyStored[MWh]': battery_energy_storage_mwh,
                'BatteryEnergyConsumed[MWh]': battery_energy_consumption_mwh,
                'PVPowerGeneration[MW]': pv_mw,
                'BatteryPower[MW]': charge_power_mw
            })         
            
            # Keep a snapshot for get_data()
            self.results[eid] = {}
            self.results[eid].update(step_results)
        return time+next_max_adv


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
    return mosaik_api.start_simulation(HouseholdProducerModel(), 'household-producer-model simulator')


if __name__ == '__main__':
    main()
