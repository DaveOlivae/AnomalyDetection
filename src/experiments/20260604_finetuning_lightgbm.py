from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import paths, settings
from configs.logger import setup_logger
from src.data_handling.make_dataset import load_window_split
from src.modeling.evaluation import evaluate_binary_classifier, save_metrics_table
from src.modeling.persistence import load_json, save_joblib, save_json


EXPERIMENT_NAME = "20260604_finetuning_lightgbm"

DATA_DIR = paths.FINAL_DATA_DIR

MODEL_NAME = "lightgbm"
N_ITER = 40
CV_SPLITS = 3
SCORING = "f1"
REFIT = True
N_JOBS = -1
USE_SCALER = True

logger = logging.getLogger(__name__)

output_paths = paths.create_output_dirs(EXPERIMENT_NAME)
LOG_PATH = output_paths["logs_dir"] / f"{EXPERIMENT_NAME}.log"



def lightgbm_param_distributions() -> dict[str, list]:
    return {
        "model__n_estimators": [200, 300, 500, 800, 1200],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "model__num_leaves": [15, 31, 63, 127, 255],
        "model__max_depth": [-1, 4, 6, 8, 10, 12],
        "model__min_child_samples": [10, 20, 30, 50, 80, 120],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        "model__min_split_gain": [0.0, 0.01, 0.05, 0.1, 0.2],
    }


def build_model(random_state: int):
    if LGBMClassifier is None:
        raise ImportError("LightGBM is not installed. Install it with: poetry add lightgbm")
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )


def build_pipeline():
    estimator = build_model(settings.RANDOM_STATE)
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def save_finetuning_config(feature_names: list[str], param_distributions: dict[str, list]) -> Path:
    config_path = save_experiment_config(feature_names)
    payload = load_json(config_path)
    payload.update(
        {
            "model": MODEL_NAME,
            "search": {
                "type": "RandomizedSearchCV",
                "n_iter": N_ITER,
                "cv_splits": CV_SPLITS,
                "scoring": SCORING,
                "refit": REFIT,
                "n_jobs": N_JOBS,
                "param_distributions": param_distributions,
            },
        }
    )
    return save_json(payload, config_path)

def save_experiment_config(feature_names: list[str]) -> Path:
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_paths["experiment_output_dir"]),
        "models_dir": str(output_paths["models_dir"]),
        "reports_dir": str(output_paths["reports_dir"]),
        "logs_dir": str(output_paths["logs_dir"]),
        "log_path": str(LOG_PATH),
        "random_state": settings.RANDOM_STATE,
        "use_scaler": USE_SCALER,
        "models": MODEL_NAME,
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    return save_json(payload, output_paths["reports_dir"] / "experiment_config.json")


def build_random_search() -> RandomizedSearchCV:
    pipeline = build_pipeline()
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=settings.RANDOM_STATE,
    )
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=lightgbm_param_distributions(),
        n_iter=N_ITER,
        scoring=SCORING,
        n_jobs=N_JOBS,
        cv=cv,
        refit=REFIT,
        random_state=settings.RANDOM_STATE,
        verbose=2,
        return_train_score=True,
    )


def main() -> None:
    
    setup_logger(LOG_PATH)

    logger.info("Starting LightGBM finetuning: %s", EXPERIMENT_NAME)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Output directory: %s", output_paths["experiment_output_dir"])
    logger.info("Log path: %s", LOG_PATH)

    X_train, y_train, feature_names = load_window_split(DATA_DIR / "train.npz")
    X_val, y_val, _ = load_window_split(DATA_DIR / "val.npz")
    X_test, y_test, _ = load_window_split(DATA_DIR / "test.npz")
    logger.info("Train: X=%s | y=%s", X_train.shape, y_train.shape)
    logger.info("Validation: X=%s | y=%s", X_val.shape, y_val.shape)
    logger.info("Test: X=%s | y=%s", X_test.shape, y_test.shape)

    param_distributions = lightgbm_param_distributions()
    config_path = save_finetuning_config(feature_names, param_distributions)
    logger.info("Saved finetuning config: %s", config_path)

    search = build_random_search()

    start = time.time()
    logger.info("Running random search for %s candidates with %s-fold CV.", N_ITER, CV_SPLITS)
    search.fit(X_train, y_train)
    search_time = time.time() - start

    cv_results_path = output_paths["reports_dir"] / "random_search_cv_results.csv"
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        cv_results_path,
        index=False,
    )
    logger.info("Saved CV results: %s", cv_results_path)

    best_payload = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_index": int(search.best_index_),
        "scoring": SCORING,
        "search_time_seconds": search_time,
    }
    best_params_path = save_json(best_payload, output_paths["reports_dir"] / "best_params.json")
    logger.info("Saved best params: %s", best_params_path)

    best_model = search.best_estimator_
    validation_metrics = evaluate_binary_classifier(
        best_model,
        X_val,
        y_val,
        MODEL_NAME,
        output_paths["reports_dir"] / "validation",
    )
    validation_metrics["search_time_seconds"] = search_time
    validation_metrics["best_cv_score"] = float(search.best_score_)

    test_metrics = evaluate_binary_classifier(
        best_model,
        X_test,
        y_test,
        MODEL_NAME,
        output_paths["reports_dir"] / "test",
    )
    test_metrics["search_time_seconds"] = search_time
    test_metrics["best_cv_score"] = float(search.best_score_)

    save_metrics_table({MODEL_NAME: validation_metrics}, output_paths["reports_dir"] / "validation_metrics.csv")
    save_metrics_table({MODEL_NAME: test_metrics}, output_paths["reports_dir"] / "test_metrics.csv")

    model_path = save_joblib(best_model, output_paths["models_dir"] / f"{MODEL_NAME}_best.joblib")
    search_path = save_joblib(search, output_paths["models_dir"] / f"{MODEL_NAME}_random_search.joblib")
    logger.info(
        "Finished finetuning in %.2f seconds | best_cv_%s=%.4f | validation_f1=%.4f | test_f1=%.4f",
        search_time,
        SCORING,
        search.best_score_,
        validation_metrics.get("f1", float("nan")),
        test_metrics.get("f1", float("nan")),
    )
    logger.info("Saved best model: %s", model_path)
    logger.info("Saved random search object: %s", search_path)


if __name__ == "__main__":
    main()
