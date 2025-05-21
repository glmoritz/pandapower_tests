import os
import sys

irradiation_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if irradiation_module_path not in sys.path:
    sys.path.insert(0, irradiation_module_path)


import SolarIrradiationModel

__all__ = ["SolarIrradiationModel"]
