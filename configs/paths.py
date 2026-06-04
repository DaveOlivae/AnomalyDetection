from pathlib import Path

# root dir
ROOT_DIR = Path(__file__).resolve().parents[1]

# source code dir
SRC_DIR = ROOT_DIR / "src"

# config dir
CONFIG_DIR = ROOT_DIR / "configs"

# data dirs
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

# output dirs
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

# tep files
FAULT_FREE_TRAIN = RAW_DATA_DIR / "TEP_FaultFree_Training.csv"
FAULTY_TRAIN = RAW_DATA_DIR / "TEP_Faulty_Training.csv"
FAULT_FREE_TEST = RAW_DATA_DIR / "TEP_FaultFree_Testing.csv"
FAULTY_TEST = RAW_DATA_DIR / "TEP_Faulty_Testing.csv"
VARIABLE_NAMES = RAW_DATA_DIR / "variable_names.json"

# processed files path
PROCESSED_FF_TRAIN = PROCESSED_DATA_DIR / "TEP_FaultFree_Training_Proc.csv"
PROCESSED_FA_TRAIN = PROCESSED_DATA_DIR / "TEP_Faulty_Training_Proc.csv"
PROCESSED_FF_TEST = PROCESSED_DATA_DIR / "TEP_FaultFree_Testing_Proc.csv"
PROCESSED_FA_TEST = PROCESSED_DATA_DIR / "TEP_Faulty_Testing_Proc.csv"


def ensure_project_dirs():
    """
    Checks to ensure all project directories exist, and creates them if they don't.
    """

    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FINAL_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
