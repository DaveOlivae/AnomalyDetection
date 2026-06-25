import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from configs.logger import setup_logger
from configs.tep_config import TEPConfig, TEPDatasetPaths
from configs.paths import create_output_dirs
from src.data_handling.data_loader import choose_simulations, load_selected_simulations, split_by_simulation

# ========== CONFIGS ==========

EXPERIMENT_NAME = "20260618_pipeline_experiment"

output_paths = create_output_dirs(EXPERIMENT_NAME)

logger = setup_logger(output_paths["logs_dir"] / "experiment.log")

paths = TEPDatasetPaths()
config = TEPConfig()

config.n_ff_train = 120
config.n_fa_train = 6
config.n_ff_test = 60
config.n_fa_test = 3

rng = np.random.default_rng(config.random_state)

# ========== PICKING SIMULATIONS ==========

logger.info("Picking simulations...")

sims_ff_train = choose_simulations(paths.ff_train, config.n_ff_train, rng)
sims_fa_train = choose_simulations(paths.fa_train, config.n_fa_train, rng)
#sims_ff_test = choose_simulations(paths.ff_test, config.n_ff_test, rng)
#sims_fa_test = choose_simulations(paths.fa_test, config.n_fa_test, rng)

# ========== LOADING DATA ==========

logger.info("Loading the selected simulations...")
ff_train = load_selected_simulations(path=paths.ff_train, sim_ids=sims_ff_train, chunksize=config.chunksize)
fa_train = load_selected_simulations(path=paths.fa_train, sim_ids=sims_fa_train, chunksize=config.chunksize)

logger.info("Done! FF_Train Shape: %s", ff_train.shape)
logger.info("Done! FA_Train Shape: %s", fa_train.shape)

trainval_df = pd.concat([ff_train, fa_train], ignore_index=True)

train_df, val_df = split_by_simulation(trainval_df, config.val_ratio, rng, label_col="faultNumber")

# ========= MODEL TRAINING ==========

X_train = train_df.drop(columns=config.meta_cols)
y_train = train_df["faultNumber"]

X_val = val_df.drop(columns=config.meta_cols)
y_val = val_df["faultNumber"]

logger.info("X_train Shape: %s", X_train.shape)
logger.info("y_train Shape: %s", y_train.shape)

rnd_forest = RandomForestClassifier(random_state=config.random_state)

logger.info("Training Random Forest...")

rnd_forest.fit(X_train, y_train)

logger.info("Making predictions...")

predictions = rnd_forest.predict(X_val)

print(classification_report(y_val, predictions))
