from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import paths, settings
from configs.logger import setup_logger
from src.data_handling.make_dataset import load_window_split
from src.modeling.evaluation import evaluate_binary_classifier, save_metrics_table
from src.modeling.persistence import save_joblib, save_json


EXPERIMENT_NAME = "20260610_finetuning_xgboost_fast"

DATA_DIR = paths.FINAL_DATA_DIR

# --- Configurações calibradas pro tempo disponível ---
N_TRIALS             = 12       # mais que RF pois early stopping encurta cada trial
CV_SPLITS            = 2
SUBSAMPLE_SIZE       = 20_000
EARLY_STOPPING_ROUNDS = 30
SCORING              = "f1"
N_JOBS               = -1
USE_SCALER           = True

logger = logging.getLogger(__name__)

output_paths = paths.create_output_dirs(EXPERIMENT_NAME)
LOG_PATH = output_paths["logs_dir"] / f"{EXPERIMENT_NAME}.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pipeline(model) -> Pipeline:
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _subsample(X, y, size: int, random_state: int):
    if len(y) <= size:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=size, stratify=y, random_state=random_state,
    )
    logger.info("Subsampled for search: %d → %d samples", len(y), len(y_sub))
    return X_sub, y_sub


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(X_search, y_search, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        if XGBClassifier is None:
            raise ImportError("XGBoost não instalado. Use: poetry add xgboost")

        params = {
            "n_estimators":        trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":       trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth":           trial.suggest_int("max_depth", 3, 10),
            "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":           trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":          trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "min_child_weight":    trial.suggest_int("min_child_weight", 1, 10),
            "gamma":               trial.suggest_float("gamma", 0.0, 0.5),
            "n_jobs":              N_JOBS,
            "random_state":        settings.RANDOM_STATE,
            "eval_metric":         "logloss",
            "verbosity":           0,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        }

        scores = []
        for train_idx, val_idx in cv.split(X_search, y_search):
            X_tr, X_vl = X_search[train_idx], X_search[val_idx]
            y_tr, y_vl = y_search[train_idx], y_search[val_idx]

            if USE_SCALER:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_vl = scaler.transform(X_vl)

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            scores.append(f1_score(y_vl, model.predict(X_vl)))

        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

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
        "model": "xgboost",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "search": {
            "type": "Optuna/TPE",
            "n_trials": N_TRIALS,
            "cv_splits": CV_SPLITS,
            "subsample_size": SUBSAMPLE_SIZE,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "scoring": SCORING,
            "note": "fast run — early stopping + subsampling",
        },
    }
    return save_json(payload, output_paths["reports_dir"] / "experiment_config.json")


def save_optuna_trials(study: optuna.Study, reports_dir: Path) -> Path:
    rows = [
        {
            "trial_number": t.number,
            "value": t.value,
            "state": t.state.name,
            "duration_seconds": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else None
            ),
            **{f"param_{k}": v for k, v in t.params.items()},
        }
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    df = pd.DataFrame(rows).sort_values("value", ascending=False)
    path = reports_dir / "optuna_trials.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logger(LOG_PATH)

    logger.info("Starting FAST experiment: %s", EXPERIMENT_NAME)
    logger.info("Trials: %d | CV: %d-fold | Subsample: %d | Early stopping: %d rounds",
                N_TRIALS, CV_SPLITS, SUBSAMPLE_SIZE, EARLY_STOPPING_ROUNDS)

    X_train, y_train, feature_names = load_window_split(DATA_DIR / "train.npz")
    X_val, y_val, _                 = load_window_split(DATA_DIR / "val.npz")
    X_test, y_test, _               = load_window_split(DATA_DIR / "test.npz")
    logger.info("Train: X=%s | y=%s", X_train.shape, y_train.shape)
    logger.info("Validation: X=%s | y=%s", X_val.shape, y_val.shape)
    logger.info("Test: X=%s | y=%s", X_test.shape, y_test.shape)

    save_experiment_config(feature_names)

    X_search, y_search = _subsample(X_train, y_train, SUBSAMPLE_SIZE, settings.RANDOM_STATE)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=settings.RANDOM_STATE)
    objective = make_objective(X_search, y_search, cv)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_STATE),
    )

    start = time.time()
    logger.info("Running Optuna study...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    search_time = time.time() - start
    logger.info("Search done in %.1f min", search_time / 60)

    trials_path = save_optuna_trials(study, output_paths["reports_dir"])
    logger.info("Saved trials: %s", trials_path)

    best_params = study.best_params
    save_json(
        {
            "best_score": float(study.best_value),
            "best_params": best_params,
            "best_trial": study.best_trial.number,
            "scoring": SCORING,
            "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "search_time_seconds": search_time,
        },
        output_paths["reports_dir"] / "best_params.json",
    )

    # Remove early_stopping_rounds do retreinamento final —
    # sem eval_set ele causaria erro
    final_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}

    logger.info("Retraining on full training set (%d samples)...", len(y_train))
    best_model = XGBClassifier(
        **final_params,
        n_jobs=N_JOBS,
        random_state=settings.RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    best_pipeline = _build_pipeline(best_model)
    best_pipeline.fit(X_train, y_train)

    val_metrics = evaluate_binary_classifier(
        best_pipeline, X_val, y_val, "xgboost", output_paths["reports_dir"] / "validation",
    )
    val_metrics["search_time_seconds"] = search_time
    val_metrics["best_cv_score"] = float(study.best_value)

    test_metrics = evaluate_binary_classifier(
        best_pipeline, X_test, y_test, "xgboost", output_paths["reports_dir"] / "test",
    )
    test_metrics["search_time_seconds"] = search_time
    test_metrics["best_cv_score"] = float(study.best_value)

    save_metrics_table({"xgboost": val_metrics},  output_paths["reports_dir"] / "validation_metrics.csv")
    save_metrics_table({"xgboost": test_metrics}, output_paths["reports_dir"] / "test_metrics.csv")

    model_path = save_joblib(best_pipeline, output_paths["models_dir"] / "xgboost_best.joblib")
    study_path = save_joblib(study,         output_paths["models_dir"] / "xgboost_optuna_study.joblib")

    total_time = time.time() - start
    logger.info(
        "Finished in %.1f min | best_cv_f1=%.4f | val_f1=%.4f | test_f1=%.4f",
        total_time / 60,
        study.best_value,
        val_metrics.get("f1", float("nan")),
        test_metrics.get("f1", float("nan")),
    )
    logger.info("Saved model: %s", model_path)
    logger.info("Saved study: %s", study_path)


if __name__ == "__main__":
    main()