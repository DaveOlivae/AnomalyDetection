"""
Standard Random Forest Implementation with Sliding windows and mlflow
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from configs.logger import setup_logger
from configs.paths import create_output_dirs
from configs.tep_config import TEPWindowConfig, TEPDatasetPaths
from configs.mlflow_config import setup_mlflow
from src.data_handling.data_loader import load_binary_trainval_test
from src.data_handling.create_windows import build_windows
from src.modeling.evaluation import evaluate_binary_classifier
from src.modeling.persistence import save_joblib


# ================= CONFIGS =======================

EXPERIMENT_NAME = "20260622_teste_random_forest_mlflow"

output_paths = create_output_dirs(EXPERIMENT_NAME)
logger = setup_logger(output_paths["logs_dir"] / "experiment.log")

paths = TEPDatasetPaths()
config = TEPWindowConfig()

setup_mlflow(EXPERIMENT_NAME)

params = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": config.random_state,
}


with mlflow.start_run(run_name="random_forest_windows"):

    # ================= DATA LOADING ======================

    # logging parameters to mlflow
    mlflow.log_params(params)

    mlflow.log_param("window_size", config.window_size)
    mlflow.log_param("stride", config.stride)
    mlflow.log_param("window_mode", config.mode)
    mlflow.log_param("random_state", config.random_state)

    # loads the data
    train_df, val_df, test_df, feature_columns = load_binary_trainval_test(paths, config)

    mlflow.log_param("n_feature_columns_original", len(feature_columns))
    mlflow.log_param("train_df_rows", len(train_df))
    mlflow.log_param("val_df_rows", len(val_df))
    mlflow.log_param("test_df_rows", len(test_df))

    # building the windows

    X_train, y_train = build_windows(
        train_df,
        feature_columns,
        config.window_size,
        config.stride,
        config.mode,
    )

    X_val, y_val = build_windows(
        val_df,
        feature_columns,
        config.window_size,
        config.stride,
        config.mode,
    )

    X_test, y_test = build_windows(
        test_df,
        feature_columns,
        config.window_size,
        config.stride,
        config.mode,
    )

    mlflow.log_param("X_train_shape", str(X_train.shape))
    mlflow.log_param("X_val_shape", str(X_val.shape))
    mlflow.log_param("X_test_shape", str(X_test.shape))

    mlflow.log_metric("train_windows_normal", int((y_train == 0).sum()))
    mlflow.log_metric("train_windows_fault", int((y_train == 1).sum()))
    mlflow.log_metric("val_windows_normal", int((y_val == 0).sum()))
    mlflow.log_metric("val_windows_fault", int((y_val == 1).sum()))
    mlflow.log_metric("test_windows_normal", int((y_test == 0).sum()))
    mlflow.log_metric("test_windows_fault", int((y_test == 1).sum()))

    # ================== TRAINING MODEL ========================

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**params)),
    ])

    logger.info("Training Random Forest...")
    model.fit(X_train, y_train)


    # ================= EVALUATING MODEL =======================


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

    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    metrics_path = output_paths["reports_dir"] / "metrics.csv"
    pd.DataFrame({"validation": val_metrics, "test": test_metrics}).T.to_csv(metrics_path)

    model_path = save_joblib(
        model,
        output_paths["models_dir"] / "Random_Forest.joblib",
    )

    mlflow.log_artifact(metrics_path, artifact_path="reports")
    mlflow.log_artifact(output_paths["logs_dir"] / "experiment.log", artifact_path="logs")

    mlflow.log_artifacts(output_paths["validation_reports_dir"], artifact_path="validation_reports")
    mlflow.log_artifacts(output_paths["test_reports_dir"], artifact_path="test_reports")
    mlflow.log_artifact(model_path, artifact_path="models")

    mlflow.sklearn.log_model(model, name="model")

    logger.info("Saved model: %s", model_path)
    logger.info("Experiment finished.")