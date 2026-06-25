"""
Standard Random Forest Implementation with Sliding windows
"""


import pandas as pd
#import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from configs.logger import setup_logger
from configs.paths import create_output_dirs
from configs.tep_config import TEPWindowConfig, TEPDatasetPaths
#from configs.mlflow_config import setup_mlflow
from src.data_handling.data_loader import load_binary_trainval_test
from src.data_handling.create_windows import build_windows
from src.modeling.evaluation import evaluate_binary_classifier
from src.modeling.persistence import save_joblib


# ==================== CONFIGS =============================


EXPERIMENT_NAME = "20260622_teste_random_forest"

output_paths = create_output_dirs(EXPERIMENT_NAME)

logger = setup_logger(output_paths["logs_dir"] / "experiment.log")

paths = TEPDatasetPaths()
config = TEPWindowConfig()

params = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": config.random_state,
}


# ==================== DATA LOADING =========================

# normal data
train_df, val_df, test_df, feature_columns = load_binary_trainval_test(paths, config)

# creating windows

X_train, y_train = build_windows(train_df, feature_columns, config.window_size, config.stride, config.mode)

X_val, y_val = build_windows(val_df, feature_columns, config.window_size, config.stride, config.mode)

X_test, y_test = build_windows(test_df, feature_columns, config.window_size, config.stride, config.mode)

# ==================== TRAINING ============================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(**params))
])

logger.info("Training Random Forest...")

model.fit(X_train, y_train)


# =================== EVALUATION ===========================


logger.info("Evaluating Random Forest...")

val_metrics = evaluate_binary_classifier(
    model,
    X_val,
    y_val,
    "RF Validation",
    output_paths["validation_reports_dir"],
)

test_metrics = evaluate_binary_classifier(
    model,
    X_test,
    y_test,
    "RF Test",
    output_paths["test_reports_dir"],
)

pd.DataFrame({"validation": val_metrics, "test": test_metrics}).T.to_csv(
    output_paths["reports_dir"] / "metrics.csv"
)

model_path = save_joblib(model, output_paths["models_dir"] / "Random_Forest.joblib")
logger.info("Saved model: %s", model_path)
logger.info("Experiment finished.")
