import pandas as pd
import pvlib
import numpy as np
from datetime import datetime, timedelta
import math
import os
from numbers import Number

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
            "params": ['SolarPeakPower_MW', 'StorageCapacity_MWh', 'InitialSOC_percent',
                       'MaxChargePower_MW', 'MaxDischargePower_MW', 'InverterType',
                       'InstallationType', 'BreakerLimit_MW', 'Index'],
            "trigger": ['Irradiance[W/m2]', 'PowerConsumption[MW]', 'GridExportLimit[MW]',
                         'GridDemand[MW]', 'PowerFactorDemand'],
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
                           'BatteryPower[MW]',
                           'Surplus_a[MW]',
                           'Surplus_b[MW]',
                           'Surplus_c[MW]',
                           'BreakerOverload_a[MW]',
                           'BreakerOverload_b[MW]',
                           'BreakerOverload_c[MW]',
                           'BreakerOverload[MW]',
                           ],

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
        self.enable_balance_breakpoint = False
        self.balance_tolerance_mw = 1e-6
        
    
    @override    
    def init(self, sid, time_resolution, time_step, sim_start, date_format=None, type="hybrid",
             enable_balance_breakpoint=False, balance_tolerance_mw=1e-6):
        self.type = type
        self.time_step = time_step
        if self.type != "hybrid":
            print("This simulator type is always hybrid")
            self.type = 'hybrid'
        
        self.sid = sid
        self.time_res = pd.Timedelta(time_resolution, unit='seconds')
        self.start_date = pd.to_datetime(sim_start, format=date_format)
        env_flag = os.getenv('HOUSEHOLD_BALANCE_BREAKPOINT', '')
        #self.enable_balance_breakpoint = bool(enable_balance_breakpoint) or env_flag.lower() in ('1', 'true', 'yes', 'on')
        self.enable_balance_breakpoint = True
        self.balance_tolerance_mw = float(balance_tolerance_mw)
        

        return self.meta

    def _trigger_balance_breakpoint(self, *, entity_key, time_s, values_by_var, balance_mw):
        if not self.enable_balance_breakpoint:
            return

        debug_payload = {
            'entity': entity_key,
            'time_s': time_s,
            'balance_mw': balance_mw,
            'tolerance_mw': self.balance_tolerance_mw,
            'P_a_load[MW]': values_by_var.get('P_a_load[MW]'),
            'P_b_load[MW]': values_by_var.get('P_b_load[MW]'),
            'P_c_load[MW]': values_by_var.get('P_c_load[MW]'),
            'PowerConsumption[MW]': values_by_var.get('PowerConsumption[MW]'),
            'BreakerOverload[MW]': values_by_var.get('BreakerOverload[MW]'),
            'BreakerOverload_a[MW]': values_by_var.get('BreakerOverload_a[MW]'),
            'BreakerOverload_b[MW]': values_by_var.get('BreakerOverload_b[MW]'),
            'BreakerOverload_c[MW]': values_by_var.get('BreakerOverload_c[MW]'),
            'PVPowerGeneration[MW]': values_by_var.get('PVPowerGeneration[MW]'),
            'BatteryPower[MW]': values_by_var.get('BatteryPower[MW]'),
        }
        print(
            "[BALANCE-ERROR] Household identity violation "
            f"entity={entity_key} t={time_s}s "
            f"balance={balance_mw:.12f} MW tol={self.balance_tolerance_mw} MW"
        )
        print(f"[BALANCE-ERROR] payload={debug_payload}")
        breakpoint()

    def _compute_balance_mw(self, values_by_var):
        p_sum = (
            float(values_by_var.get('P_a_load[MW]', 0.0))
            + float(values_by_var.get('P_b_load[MW]', 0.0))
            + float(values_by_var.get('P_c_load[MW]', 0.0))
        )
        overload_total = float(values_by_var.get('BreakerOverload[MW]', 0.0))
        if overload_total == 0.0:
            overload_total = (
                float(values_by_var.get('BreakerOverload_a[MW]', 0.0))
                + float(values_by_var.get('BreakerOverload_b[MW]', 0.0))
                + float(values_by_var.get('BreakerOverload_c[MW]', 0.0))
            )

        pv = float(values_by_var.get('PVPowerGeneration[MW]', 0.0))
        battery = float(values_by_var.get('BatteryPower[MW]', 0.0))
        power_consumption = float(values_by_var.get('PowerConsumption[MW]', 0.0))

        return p_sum + overload_total + pv - battery - power_consumption

    def _check_balance_identity(self, *, entity_key, time_s, values_by_var):
        required = (
            'PVPowerGeneration[MW]',
            'BatteryPower[MW]',
        )
        if not all(var in values_by_var for var in required):
            return

        balance_mw = self._compute_balance_mw(values_by_var)
        if not math.isfinite(balance_mw) or abs(balance_mw) > self.balance_tolerance_mw:
            self._trigger_balance_breakpoint(
                entity_key=entity_key,
                time_s=time_s,
                values_by_var=values_by_var,
                balance_mw=balance_mw,
            )

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
                'BreakerLimit_MW': params.get('BreakerLimit_MW', [float('inf')])[i],
                'InverterType': params.get('InverterType', ['3ph'])[i],
                'InstallationType': params.get('InstallationType', ['3ph'])[i],
                'rng': np.random.default_rng(abs(hash(f'{self.eid_prefix}_{params.get("Index")[i]}')) % (2**32)),
                # Per-hour load imbalance state (initialised to equal weights; refreshed every 3600 s)
                'load_imbalance_weights': None,  # will be set at first step
                'last_imbalance_update_time': -3600,  # force resample at t=0
                'last_pv_generation_power_mw': 0.0,
                'last_load_power_mw': 0.0,
                'last_charge_power_mw': 0.0,
                'last_curtailment_power_mw': 0.0,
                'charge_efficiency': 0.95, #tesla powerwall 3
                'discharge_efficiency': 0.90, #tesla powerwall 3
                'time': 0
            }
            entities.append({'eid': eid, 'type': model})
             
        return entities
    
    
    def step(self, time, inputs, max_advance):
        """
        Hybrid step with per-phase power-flow awareness.

        Physical model -- Normal Grid-Tied (Parallel) Inverter:
          - The grid acts as a stiff voltage source on each phase (grid-forming).
          - The inverter operates in grid-following mode as a controlled current
            source, injecting power into the home's AC bus in parallel with the
            grid.
          - The inverter splits its AC output EQUALLY across its phases --
            it does NOT compensate for load unbalance.  If the house draws
            3kW on Phase A, 1kW on Phase B, and 0kW on Phase C, and the
            inverter outputs 3kW total, it injects 1kW per phase.
          - Each installation phase has an independent breaker that limits how
            much power the house can draw from the grid on that phase.
          - The inverter prioritises sending surplus energy to the battery.
            Only surplus that the battery cannot absorb (full or max-charge-rate
            exceeded) is injected to the AC bus and potentially exported.

        Per-phase flow:
          1. Distribute house load across installation phases (hourly
             imbalance weights).
          2. Compute PV output per inverter phase (equal split).
          3. Compute per-phase surplus = PV_per_phase - load_per_phase.
          4. Battery charge / discharge based on total net surplus
             (the inverter is one unit with one DC bus).
          5. Inverter AC output per phase = (PV - battery_charge) / n_inv.
          6. Net grid per phase = load - inverter_output  (pos -> import).
          7. Apply per-phase breaker limit on grid import.
        """
        for eid, entity in self.entities.items():
            # if eid == 'HouseholdProducer_bus4' and time == 900:                
            #     breakpoint()
            # =============================================================
            # 1. STATIC PARAMETERS
            # =============================================================
            peak_mw    = entity['SolarPeakPower_MW']
            cap_mwh    = entity['StorageCapacity_MWh']
            max_chg_mw = entity['MaxChargePower_MW']
            max_dch_mw = entity['MaxDischargePower_MW']
            breaker_mw = entity['BreakerLimit_MW']  # per-phase grid-draw limit

            last_results = self.results.get(eid, {})

            # =============================================================
            # 2. PHASE CONFIGURATION
            # =============================================================
            inv_type   = entity['InverterType']
            inst_type  = entity['InstallationType']
            entity_rng = entity['rng']

            _phase_count = {'1ph': 1, '2ph': 2, '3ph': 3}
            if _phase_count.get(inv_type, 0) > _phase_count.get(inst_type, 0):
                raise ValueError(
                    f"InverterType '{inv_type}' has more phases than "
                    f"InstallationType '{inst_type}'. "
                    f"An inverter cannot have more phases than the installation."
                )

            n_inst = _phase_count[inst_type]
            n_inv  = _phase_count[inv_type]

            all_logical     = ['a', 'b', 'c']
            load_phases     = all_logical[:n_inst]
            inverter_phases = all_logical[:n_inv]

            # =============================================================
            # 3. INPUTS -- use last known value if not provided this step
            # =============================================================
            irr_wm2 = list(inputs.get(eid, {}).get('Irradiance[W/m2]', {}).values()) or [entity.get('irr_wm2', 0.0)]
            irr_wm2 = irr_wm2[0]
            load_mw = list(inputs.get(eid, {}).get('PowerConsumption[MW]', {}).values()) or [entity.get('last_load_power_mw', 0.0)]
            load_mw = load_mw[0]

            export_cap_mw = list(inputs.get(eid, {}).get('GridExportLimit[MW]', {}).values()) or [float('inf')]
            export_cap_mw = export_cap_mw[0]
            grid_req_mw = list(inputs.get(eid, {}).get('GridDemand[MW]', {}).values()) or [0.0]
            grid_req_mw = max(0.0, grid_req_mw[0])

            power_factor = list(inputs.get(eid, {}).get('PowerFactorDemand', {}).values()) or [1.0]
            power_factor = power_factor[0]
            if power_factor == 0:
                raise ValueError("PowerFactorDemand cannot be zero.")
            q_per_p = math.tan(math.acos(power_factor))

            # =============================================================
            # 4. TIME KEEPING
            # =============================================================
            last_time    = entity.get('last_time', 0)
            step_seconds = max(1, time - last_time)
            step_hours   = step_seconds / 3600.0
            entity['last_time'] = time

            # =============================================================
            # 5. LOAD IMBALANCE -- re-sampled once per hour
            #    The house load is distributed unevenly across installation
            #    phases.  A new set of per-phase weights is drawn every
            #    3600 s of sim time.
            # =============================================================
            last_imbalance_update = entity.get('last_imbalance_update_time', -3600)
            if (time - last_imbalance_update) >= 3600:
                if n_inst == 1:
                    imbalance_weights = np.array([1.0])
                else:
                    raw = entity_rng.uniform(0.5, 1.5, size=n_inst)
                    imbalance_weights = raw / raw.sum()
                entity['load_imbalance_weights'] = imbalance_weights
                entity['last_imbalance_update_time'] = time
            else:
                imbalance_weights = entity['load_imbalance_weights']

            # =============================================================
            # 6. PER-PHASE LOAD DISTRIBUTION
            # =============================================================
            load_per_phase = {}
            for idx, phase in enumerate(load_phases):
                load_per_phase[phase] = load_mw * imbalance_weights[idx]
            for phase in all_logical:
                load_per_phase.setdefault(phase, 0.0)

            # =============================================================
            # 7. PV GENERATION
            # =============================================================
            pv_mw = (irr_wm2 / 1000.0) * peak_mw
            pv_per_phase_mw = pv_mw / n_inv if n_inv > 0 else 0.0

            # =============================================================
            # 8. PER-PHASE SURPLUS (before battery)
            #    Positive -> PV exceeds load on this phase
            #    Negative -> load exceeds PV on this phase
            # =============================================================
            surplus_per_phase = {}
            for phase in all_logical:
                pv_on_phase = pv_per_phase_mw if phase in inverter_phases else 0.0
                surplus_per_phase[phase] = pv_on_phase - load_per_phase[phase]

            # =============================================================
            # 9. SOC UPDATE (integrate last step's battery decision)
            # =============================================================
            soc_mwh = entity.get('SOC[MWh]', 0.0)
            last_charge_power_mw = entity.get('last_charge_power_mw', 0.0)

            eff = (entity['charge_efficiency'] if last_charge_power_mw > 0.0
                   else 1.0 / entity['discharge_efficiency'])
            soc_mwh += last_charge_power_mw * step_hours * eff
            soc_mwh = max(0.0, min(cap_mwh, soc_mwh))
            soc_percent = 100.0 * soc_mwh / cap_mwh if cap_mwh > 0 else 0.0

            minimum_battery_operation_time = 10.0  # seconds

            # =============================================================
            # 10. BATTERY & GRID LOGIC
            #
            # The inverter is a single unit with one DC bus (PV + battery).
            # It observes the per-phase surplus but makes the battery
            # decision based on the TOTAL net surplus across all phases.
            #
            # Priority when surplus exists:
            #   1) Satisfy explicit grid demand (GridDemand)
            #   2) Charge battery -- absorb as much surplus as possible
            #   3) Export remaining surplus to grid
            #   4) Curtail if grid export cap exceeded
            #
            # The inverter only injects surplus to the AC bus when the
            # battery cannot absorb it (full capacity or max charge rate
            # exceeded).
            # =============================================================
            surplus_mw      = pv_mw - load_mw   # total net surplus
            pv_curtail_mw   = 0.0
            charge_power_mw = 0.0
            export_to_grid_mw = 0.0

            # ---- CASE A: Net PV surplus -----------------------------------
            if surplus_mw > 0:
                extra_for_grid = 0.0

                # A1) Grid explicitly demands power -- first priority
                if grid_req_mw > 0:
                    deliverable    = min(surplus_mw, grid_req_mw)
                    extra_for_grid = deliverable
                    surplus_mw    -= deliverable
                    grid_req_mw   -= deliverable

                # A1b) Still unmet grid demand -> discharge battery
                if grid_req_mw > 0:
                    potential_dch = min(grid_req_mw, max_dch_mw)
                    min_soc = (potential_dch
                               * (minimum_battery_operation_time / 3600.0)
                               / entity['discharge_efficiency'])
                    if soc_mwh > min_soc:
                        charge_power_mw = -potential_dch
                        grid_req_mw    += charge_power_mw   # negative

                # A2) Remaining surplus -> charge battery first
                #     Only surplus that the battery CANNOT absorb (full
                #     capacity or max charge rate) is injected to AC bus.
                if surplus_mw > 0 and soc_mwh < cap_mwh:
                    potential_chg = min(surplus_mw, max_chg_mw)
                    if (potential_chg > 0
                            and potential_chg * (minimum_battery_operation_time / 3600.0)
                                <= surplus_mw * step_hours):
                        charge_power_mw = potential_chg
                        surplus_mw     -= potential_chg

                # A3) Whatever surplus the battery could not absorb -> grid
                extra_for_grid   += surplus_mw
                export_to_grid_mw = min(extra_for_grid, export_cap_mw)
                pv_curtail_mw     = max(extra_for_grid - export_cap_mw, 0.0)

            # ---- CASE B: Net PV deficit -----------------------------------
            else:
                deficit_mw    = -surplus_mw
                potential_dch = min(deficit_mw, max_dch_mw)
                min_soc = (potential_dch
                           * (minimum_battery_operation_time / 3600.0)
                           / entity['discharge_efficiency'])
                if soc_mwh > min_soc:
                    charge_power_mw = -potential_dch
                    deficit_mw     += charge_power_mw   # negative

                export_to_grid_mw = -deficit_mw   # negative -> grid import

            # =============================================================
            # 11. NEXT INTERNAL EVENT (battery full / empty)
            # =============================================================
            next_max_adv = self.time_step

            if charge_power_mw > 0 and soc_mwh < cap_mwh:
                secs_to_full = (3600.0 * (cap_mwh - soc_mwh)
                                / (charge_power_mw * entity['charge_efficiency']))
                next_max_adv = int(min(next_max_adv, secs_to_full))
            elif charge_power_mw < 0 and soc_mwh > 0:
                secs_to_empty = (3600.0 * soc_mwh
                                 / (-charge_power_mw / entity['discharge_efficiency']))
                next_max_adv = int(min(next_max_adv, secs_to_empty))

            next_max_adv = max(1, next_max_adv)

            # =============================================================
            # 12. PER-PHASE GRID EXCHANGE
            #
            # Inverter AC output per phase (equal split -- the inverter does
            # NOT compensate for load unbalance):
            #   inverter_ac_phase = (PV_total - battery_charge) / n_inv
            #
            # Net grid power per phase (positive -> import, negative -> export):
            #   grid_phase = load_phase - inverter_ac_phase
            #
            # The per-phase breaker caps how much the house can DRAW from
            # the grid on each phase.
            #
            # Cross-check: sum(net_phase)
            #   = load_mw - inverter_net_mw
            #   = load_mw - (pv_mw - charge_power_mw)
            #   = -export_to_grid_mw  (same total energy balance)
            # =============================================================
            inverter_net_mw       = pv_mw - charge_power_mw - pv_curtail_mw
            inverter_per_phase_mw = inverter_net_mw / n_inv if n_inv > 0 else 0.0

            step_results = {}
            total_breaker_overload = 0.0

            for phase in all_logical:
                load_p = load_per_phase[phase]
                inv_p  = inverter_per_phase_mw if phase in inverter_phases else 0.0

                # Net power at the grid connection (positive = import)
                net_p = load_p - inv_p

                # ---- Per-phase breaker limit (import side only) ----
                breaker_overload = 0.0
                if net_p > breaker_mw:
                    breaker_overload        = net_p - breaker_mw
                    net_p                   = breaker_mw
                    total_breaker_overload += breaker_overload

                step_results[f'P_{phase}_load[MW]']          = net_p
                step_results[f'Q_{phase}_load[MVar]']        = net_p * q_per_p
                step_results[f'Surplus_{phase}[MW]']         = surplus_per_phase[phase]
                step_results[f'BreakerOverload_{phase}[MW]'] = breaker_overload

            step_results['BreakerOverload[MW]'] = total_breaker_overload

            # =============================================================
            # 13. ENERGY ACCOUNTING
            # =============================================================
            grid_exported_mwh = last_results.get('EnergyExported[MWh]', 0.0)
            grid_imported_mwh = last_results.get('EnergyImported[MWh]', 0.0)
            battery_energy_storage_mwh     = last_results.get('BatteryEnergyStored[MWh]', 0.0)
            battery_energy_consumption_mwh = last_results.get('BatteryEnergyConsumed[MWh]', 0.0)

            for phase in all_logical:
                p_phase = step_results[f'P_{phase}_load[MW]']
                if p_phase < 0:
                    grid_exported_mwh += (-p_phase) * step_hours
                else:
                    grid_imported_mwh += p_phase * step_hours

            curtailment_energy_mwh = (
                last_results.get('CurtailmentEnergy[MWh]', 0.0)
                + entity.get('last_curtailment_power_mw', 0.0) * step_hours
            )
            pv_generation_mwh = (
                last_results.get('PVEnergyGeneration[MWh]', 0.0)
                + entity.get('last_pv_generation_power_mw', 0.0) * step_hours
            )
            energy_consumption_mwh = (
                last_results.get('EnergyConsumption[MWh]', 0.0)
                + entity.get('last_load_power_mw', 0.0) * step_hours
            )

            if entity.get('last_charge_power_mw', 0.0) > 0:
                battery_energy_storage_mwh = (
                    last_results.get('BatteryEnergyStored[MWh]', 0.0)
                    + entity.get('last_charge_power_mw', 0.0) * step_hours
                )
            elif last_charge_power_mw < 0:
                battery_energy_consumption_mwh = (
                    last_results.get('BatteryEnergyConsumed[MWh]', 0.0)
                    + (-entity.get('last_charge_power_mw', 0.0)) * step_hours
                )

            # =============================================================
            # 14. STORE STATE FOR NEXT STEP
            # =============================================================
            entity.update({                
                'irr_wm2': irr_wm2,                
                'last_charge_power_mw': charge_power_mw,
                'last_curtailment_power_mw': pv_curtail_mw,
                'last_pv_generation_power_mw': pv_mw,
                'last_load_power_mw': load_mw,
                'SOC[MWh]': soc_mwh,
            })

            step_results.update({                
                'SOC[MWh]':                  soc_mwh,
                'SOC[percent]':              soc_percent,
                'Curtailment[MW]':           pv_curtail_mw,                
                'EnergyExported[MWh]':       grid_exported_mwh,
                'EnergyImported[MWh]':       grid_imported_mwh, 
                'CurtailmentEnergy[MWh]':    curtailment_energy_mwh,
                'PVEnergyGeneration[MWh]':   pv_generation_mwh,
                'EnergyConsumption[MWh]':    energy_consumption_mwh,
                'BatteryEnergyStored[MWh]':  battery_energy_storage_mwh,
                'BatteryEnergyConsumed[MWh]': battery_energy_consumption_mwh,
                'PVPowerGeneration[MW]':     pv_mw,
                'BatteryPower[MW]':          charge_power_mw,
            })

            if self.enable_balance_breakpoint:
                values_by_var = dict(step_results)
                values_by_var['PowerConsumption[MW]'] = load_mw
                self._check_balance_identity(
                    entity_key=eid,
                    time_s=time,
                    values_by_var=values_by_var,
                )
            
            # Keep a snapshot for get_data()
            self.results[eid] = {}
            self.results[eid].update(step_results)

        return time + next_max_adv


    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid not in self.results:
                raise ValueError('Unknown entity ID "%s"' % eid)
            data[eid] = {}
            for attr in attrs:
                value = self.results[eid][attr]
                if isinstance(value, np.floating) or isinstance(value, np.integer):
                    data[eid][attr] = value.item()
                elif isinstance(value, Number):
                    data[eid][attr] = value
                else:
                    data[eid][attr] = value

        return data


def main():
    return mosaik_api.start_simulation(HouseholdProducerModel(), 'household-producer-model simulator')


if __name__ == '__main__':
    main()
