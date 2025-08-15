import os
import sys

postgres_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if postgres_module_path not in sys.path:
    sys.path.insert(0, postgres_module_path)


import PostgresReaderModel, PostgresWriterModel

__all__ = ["PostgresReaderModel", "PostgresWriterModel"]
