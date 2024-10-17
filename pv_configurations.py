"""
@author: Fernando Penaherrera @UOL/OFFIS

Legacy collection of scenarios and configurations.
Useful for tests and for stablishing configuration dictionaries.
"""
from enum import Enum

# Data for Albedos. From the PV Lib Site
SURFACE_ALBEDOS = {
    "urban": 0.18,
    "grass": 0.20,
    "fresh grass": 0.26,
    "soil": 0.17,
    "sand": 0.40,
    "snow": 0.65,
    "fresh snow": 0.75,
    "asphalt": 0.12,
    "concrete": 0.30,
    "aluminum": 0.85,
    "copper": 0.74,
    "fresh steel": 0.35,
    "dirty steel": 0.08,
    "sea": 0.06,
}


class Scenarios(Enum):
    """Description of the scenarios for the PV Systems"""

    BASE = 1
    HOUSE = 2
    BUILDING = 3
    SIMPLE = 4


def generate_configurations(scenario=Scenarios.BASE):
    """Generates the configurations dictionaries to be passed in the PVSystem class.
    Refers to the class Scenarios for easy interpretation

    Args:
        scenario (Scenario.enum, optional): Scenario for the configurations.Defaults to Scenarios.BASE.

    Returns:
        dict: Dictionary with configuration parameters.
    """
    configuration = dict()

    # Generic Parameters

    # geometrical parameters
    configuration["latitude"] = 53.14
    configuration["longitude"] = 8.20
    configuration["elevation"] = 6
    configuration["inclination"] = 0
    configuration["surface_azimuth"] = 180
    configuration["albedo"] = SURFACE_ALBEDOS["urban"]
    configuration["timezone"] = "Europe/Berlin"
    configuration["p_dc"] = None
    configuration["p_ac"] = None
    configuration["inverter_eff"] = None

    # Technical parameters

    if scenario == Scenarios.SIMPLE:
        configuration["p_dc"] = 1000
        configuration["p_ac"] = 1000
        configuration["inverter_eff"] = 1

        return configuration

    configuration["modules_library"] = "SandiaMod"
    configuration["inverters_library"] = "cecinverter"
    configuration["module"] = "Canadian_Solar_CS5P_220M___2009_"
    configuration["inverter"] = "ABB__MICRO_0_25_I_OUTD_US_208__208V_"
    configuration["temperature_model_parameters"] = {"sapm": "open_rack_glass_glass"}
    configuration["array"] = [1, 1]  # modules per string, number of strings

    if scenario == Scenarios.BASE:
        configuration["inclination"] = 10

    if scenario == Scenarios.HOUSE:
        """
        Braunschweig location
        8 Panels, 210W per panel, 2kW Inverter
        Module: I = 5.415    V = 40.05
        Inverter: Max VDC, 416    Max IDC: 7.943536
        """

        configuration["latitude"] = 53.23
        configuration["longitude"] = 10.48
        configuration["elevation"] = 75
        configuration["albedo"] = 0.20

        configuration["module"] = "SunPower SPR-210-WHT [ 2006]"
        configuration["inverter"] = "ABB: UNO-2.0-I-OUTD-S-US [208V]"

        # 8 modules connected in parallel
        configuration["array"] = [8, 1]

    if scenario == Scenarios.BUILDING:
        """
        Berlin Location
        120 Panels, 20 Modules per string, 6 strings
        210 W per panel, 50kW Inverter
        SunPower 128-Cell Module [2009 (E)]   I = 5.49    V = 72.9    P = 400
        Inverter: Max VDC, 480    Max IDC: 142.850053
        """

        # Berlin Building
        configuration["latitude"] = 52.52
        configuration["longitude"] = 13.4
        configuration["albedo"] = 0.30
        configuration["elevation"] = 34 + 15
        configuration["inclination"]

        # 20 Strings of 6
        configuration["module"] = "SunPower SPR-210-WHT [ 2006]"
        configuration["inverter"] = "Power-One: PVI-CENTRAL-50-US [208V]"
        configuration["array"] = [20, 6]

    return configuration


if __name__ == "__main__":

    config = generate_configurations(Scenarios.BASE)
    config_simple = generate_configurations(scenario=Scenarios.SIMPLE)
