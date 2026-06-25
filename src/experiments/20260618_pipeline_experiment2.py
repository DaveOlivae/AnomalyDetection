import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from configs.logger import setup_logger
from configs.tep_config import TEPConfig, TEPDatasetPaths
from configs.paths import create_output_dirs
from src.data_handling.data_loader import load_binary_trainval_test
# ========== CONFIGS ==========

EXPERIMENT_NAME = "20260618_pipeline_experiment2"

output_paths = create_output_dirs(EXPERIMENT_NAME)

logger = setup_logger(output_paths["logs_dir"] / "experiment.log")

paths = TEPDatasetPaths()
config = TEPConfig()

config.n_ff_train = 120
config.n_fa_train = 6
config.n_ff_test = 60
config.n_fa_test = 3

rng = np.random.default_rng(config.random_state)

train_df, val_df, test_df, feature_columns = load_binary_trainval_test(paths, config)

# ========= MODEL TRAINING ==========

X_train = train_df.drop(columns=config.meta_cols + ["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=config.meta_cols + ["label"])
y_val = val_df["label"]

logger.info("X_train Shape: %s", X_train.shape)
logger.info("y_train Shape: %s", y_train.shape)

rnd_forest = RandomForestClassifier(random_state=config.random_state)

logger.info("Training Random Forest...")

rnd_forest.fit(X_train, y_train)

logger.info("Making predictions...")

predictions = rnd_forest.predict(X_val)

print(classification_report(y_val, predictions))
