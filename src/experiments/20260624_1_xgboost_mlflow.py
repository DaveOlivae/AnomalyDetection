"""
Standard XGBoost implementation with sliding windows and MLflow.

This experiment mirrors the Random Forest baseline script, but uses a fixed
XGBoost configuration.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost não está instalado. Instale com: poetry add xgboost "
        "ou pip install xgboost"
    ) from exc

from configs.logger import setup_logger
from configs.paths import create_output_dirs
from configs.tep_config import TEPDatasetPaths, TEPWindowConfig
from configs.mlflow_config import setup_mlflow
from src.data_handling.create_windows import build_windows
from src.data_handling.data_loader import load_binary_trainval_test
from src.modeling.evaluation import evaluate_binary_classifier
from src.modeling.persistence import save_joblib


# ================= CONFIGS =======================

EXPERIMENT_NAME = "20260622_1_xgboost_mlflow"

output_paths = create_output_dirs(EXPERIMENT_NAME)
logger = setup_logger(output_paths["logs_dir"] / "experiment.log")

paths = TEPDatasetPaths()
config = TEPWindowConfig()

setup_mlflow(EXPERIMENT_NAME)

params = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": config.random_state,
    "verbosity": 0,
}


def log_metrics_safe(metrics: dict[str, Any], prefix: str) -> None:
    """Loga no MLflow somente métricas numéricas válidas."""
    for key, value in metrics.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(value):
            mlflow.log_metric(f"{prefix}_{key}", value)


def log_artifact_if_exists(path: str | Path, artifact_path: str | None = None) -> None:
    """Evita quebrar o experimento caso algum relatório não tenha sido gerado."""
    path = Path(path)
    if not path.exists():
        logger.warning("Artifact not found, skipping: %s", path)
        return

    if path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
    else:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def main() -> None:
    with mlflow.start_run(run_name="xgboost_windows"):
        # ================= DATA LOADING ======================

        logger.info("Starting experiment: %s", EXPERIMENT_NAME)
        logger.info("Loading train/validation/test datasets...")

        mlflow.log_params(params)
        mlflow.log_param("window_size", config.window_size)
        mlflow.log_param("stride", config.stride)
        mlflow.log_param("window_mode", config.mode)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("model_name", "xgboost")
        mlflow.log_param("uses_scaler", False)

        train_df, val_df, test_df, feature_columns = load_binary_trainval_test(paths, config)

        mlflow.log_param("n_feature_columns_original", len(feature_columns))
        mlflow.log_param("train_df_rows", len(train_df))
        mlflow.log_param("val_df_rows", len(val_df))
        mlflow.log_param("test_df_rows", len(test_df))

        # ================= BUILDING WINDOWS ======================

        logger.info("Building sliding windows...")

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

        # XGBoost é baseado em árvores; StandardScaler normalmente não é necessário.
        model = Pipeline([
            ("model", XGBClassifier(**params)),
        ])

        logger.info("Training XGBoost...")
        model.fit(X_train, y_train)

        # ================= EVALUATING MODEL =======================

        logger.info("Evaluating XGBoost...")

        val_metrics = evaluate_binary_classifier(
            model,
            X_val,
            y_val,
            "XGB Validation",
            output_paths["validation_reports_dir"],
        )

        test_metrics = evaluate_binary_classifier(
            model,
            X_test,
            y_test,
            "XGB Test",
            output_paths["test_reports_dir"],
        )

        log_metrics_safe(val_metrics, prefix="val")
        log_metrics_safe(test_metrics, prefix="test")

        metrics_path = output_paths["reports_dir"] / "metrics.csv"
        pd.DataFrame({"validation": val_metrics, "test": test_metrics}).T.to_csv(metrics_path)

        model_path = save_joblib(
            model,
            output_paths["models_dir"] / "XGBoost.joblib",
        )

        log_artifact_if_exists(metrics_path, artifact_path="reports")
        log_artifact_if_exists(output_paths["logs_dir"] / "experiment.log", artifact_path="logs")
        log_artifact_if_exists(output_paths["validation_reports_dir"], artifact_path="validation_reports")
        log_artifact_if_exists(output_paths["test_reports_dir"], artifact_path="test_reports")
        log_artifact_if_exists(model_path, artifact_path="models")

        try:
            mlflow.sklearn.log_model(model, name="model")
        except TypeError:
            # Compatibilidade com versões antigas do MLflow.
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Saved model: %s", model_path)
        logger.info("Experiment finished.")


if __name__ == "__main__":
    main()
