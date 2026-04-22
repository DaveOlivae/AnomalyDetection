"""
Global variables for the project.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# data dirs
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# tep files
FAULT_FREE_TRAIN = RAW_DATA_DIR / "TEP_FaultFree_Training.csv"
FAULTY_TRAIN = RAW_DATA_DIR / "TEP_Faulty_Training.csv"
FAULT_FREE_TEST = RAW_DATA_DIR / "TEP_FaultFree_Testing.csv"
FAULTY_TEST = RAW_DATA_DIR / "TEP_Faulty_Testing.csv"
VARIABLE_NAMES = RAW_DATA_DIR / "variable_names.json"

# processed files path
FAULT_FREE_TRAIN = RAW_DATA_DIR / "TEP_FaultFree_Training.csv"
FAULTY_TRAIN = RAW_DATA_DIR / "TEP_Faulty_Training.csv"
FAULT_FREE_TEST = RAW_DATA_DIR / "TEP_FaultFree_Testing.csv"
FAULTY_TEST = RAW_DATA_DIR / "TEP_Faulty_Testing.csv"

