import os
import sys

producer_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if producer_module_path not in sys.path:
    sys.path.insert(0, producer_module_path)


import HouseholdProducerModel as HouseholdProducerModel

__all__ = ["HouseholdProducerModel"]
