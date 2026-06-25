from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except ImportError:
    LGBMClassifier = None

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


EXPERIMENT_NAME = "20260611_finetuning_all_models"

DATA_DIR = paths.FINAL_DATA_DIR

MODELS_TO_RUN = ["lightgbm", "random_forest", "xgboost"]

# --- Configurações de tempo ---
N_TRIALS = 20              # 20 trials por modelo = 60 no total
CV_SPLITS = 3
SUBSAMPLE_SIZE = 50_000    # amostras usadas na busca (treino final usa tudo)
SCORING = "f1"
N_JOBS = -1
USE_SCALER = True
OPTUNA_DIRECTION = "maximize"

# --- Early stopping (LightGBM e XGBoost) ---
EARLY_STOPPING_ROUNDS = 30

logger = logging.getLogger(__name__)

output_paths = paths.create_output_dirs(EXPERIMENT_NAME)
LOG_PATH = output_paths["logs_dir"] / f"{EXPERIMENT_NAME}.log"


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def make_lightgbm_objective(X_search, y_search, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        if LGBMClassifier is None:
            raise ImportError("LightGBM não instalado. Use: poetry add lightgbm")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "n_jobs": N_JOBS,
            "random_state": settings.RANDOM_STATE,
            "verbose": -1,
        }

        scores = []
        for train_idx, val_idx in cv.split(X_search, y_search):
            X_tr, X_vl = X_search[train_idx], X_search[val_idx]
            y_tr, y_vl = y_search[train_idx], y_search[val_idx]

            if USE_SCALER:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_vl = scaler.transform(X_vl)

            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    log_evaluation(period=-1),
                ],
            )
            preds = model.predict(X_vl)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_vl, preds))

        return float(np.mean(scores))

    return objective


def make_random_forest_objective(X_search, y_search, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "n_jobs": N_JOBS,
            "random_state": settings.RANDOM_STATE,
        }

        model = RandomForestClassifier(**params)
        pipeline = _build_pipeline(model)
        scores = cross_val_score(pipeline, X_search, y_search, cv=cv, scoring=SCORING, n_jobs=1)
        return float(scores.mean())

    return objective


def make_xgboost_objective(X_search, y_search, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        if XGBClassifier is None:
            raise ImportError("XGBoost não instalado. Use: poetry add xgboost")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "n_jobs": N_JOBS,
            "random_state": settings.RANDOM_STATE,
            "eval_metric": "logloss",
            "verbosity": 0,
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
            preds = model.predict(X_vl)
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_vl, preds))

        return float(np.mean(scores))

    return objective


OBJECTIVES = {
    "lightgbm": make_lightgbm_objective,
    "random_forest": make_random_forest_objective,
    "xgboost": make_xgboost_objective,
}

MODEL_BUILDERS = {
    "lightgbm": lambda p: LGBMClassifier(**p, n_jobs=N_JOBS, random_state=settings.RANDOM_STATE, verbose=-1),
    "random_forest": lambda p: RandomForestClassifier(**p, n_jobs=N_JOBS, random_state=settings.RANDOM_STATE),
    "xgboost": lambda p: XGBClassifier(**p, n_jobs=N_JOBS, random_state=settings.RANDOM_STATE, eval_metric="logloss", verbosity=0),
}

# Parâmetros que não devem ser repassados ao construtor final
# (são usados apenas durante o fit com early stopping)
_FIT_ONLY_PARAMS = {"early_stopping_rounds"}


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
    """Retorna uma amostra estratificada de `size` exemplos."""
    if len(y) <= size:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=size,
        stratify=y,
        random_state=random_state,
    )
    logger.info("Subsampled dataset for search: %d → %d samples", len(y), len(y_sub))
    return X_sub, y_sub


def _rebuild_best_pipeline(model_name: str, best_params: dict) -> Pipeline:
    clean_params = {k: v for k, v in best_params.items() if k not in _FIT_ONLY_PARAMS}
    model = MODEL_BUILDERS[model_name](clean_params)
    return _build_pipeline(model)


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
        "models": MODELS_TO_RUN,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "search": {
            "type": "Optuna/TPE",
            "n_trials": N_TRIALS,
            "cv_splits": CV_SPLITS,
            "subsample_size": SUBSAMPLE_SIZE,
            "scoring": SCORING,
            "direction": OPTUNA_DIRECTION,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
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
                if t.datetime_complete and t.datetime_start
                else None
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
# Per-model finetuning
# ---------------------------------------------------------------------------

def run_model_finetuning(
    model_name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_names: list[str],
) -> dict:

    model_reports_dir = output_paths["reports_dir"] / model_name
    model_models_dir = output_paths["models_dir"] / model_name
    model_reports_dir.mkdir(parents=True, exist_ok=True)
    model_models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Optuna finetuning: %s", model_name.upper())
    logger.info("=" * 60)

    X_search, y_search = _subsample(X_train, y_train, SUBSAMPLE_SIZE, settings.RANDOM_STATE)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=settings.RANDOM_STATE)
    objective = OBJECTIVES[model_name](X_search, y_search, cv)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        study_name=f"{EXPERIMENT_NAME}_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_STATE),
    )

    start = time.time()
    logger.info(
        "[%s] Optuna search — %d trials | %d-fold CV | search set: %d samples",
        model_name, N_TRIALS, CV_SPLITS, len(y_search),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    search_time = time.time() - start

    trials_path = save_optuna_trials(study, model_reports_dir)
    logger.info("[%s] Saved Optuna trials: %s", model_name, trials_path)

    best_params = study.best_params
    best_payload = {
        "best_score": float(study.best_value),
        "best_params": best_params,
        "best_trial": study.best_trial.number,
        "scoring": SCORING,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "search_time_seconds": search_time,
    }
    best_params_path = save_json(best_payload, model_reports_dir / "best_params.json")
    logger.info("[%s] Saved best params: %s", model_name, best_params_path)

    # Retreina no dataset completo com os melhores params
    logger.info("[%s] Retraining on full training set (%d samples)...", model_name, len(y_train))
    best_pipeline = _rebuild_best_pipeline(model_name, best_params)
    best_pipeline.fit(X_train, y_train)

    validation_metrics = evaluate_binary_classifier(
        best_pipeline, X_val, y_val, model_name, model_reports_dir / "validation",
    )
    validation_metrics["search_time_seconds"] = search_time
    validation_metrics["best_cv_score"] = float(study.best_value)

    test_metrics = evaluate_binary_classifier(
        best_pipeline, X_test, y_test, model_name, model_reports_dir / "test",
    )
    test_metrics["search_time_seconds"] = search_time
    test_metrics["best_cv_score"] = float(study.best_value)

    save_metrics_table({model_name: validation_metrics}, model_reports_dir / "validation_metrics.csv")
    save_metrics_table({model_name: test_metrics}, model_reports_dir / "test_metrics.csv")

    model_path = save_joblib(best_pipeline, model_models_dir / f"{model_name}_best.joblib")
    study_path = save_joblib(study, model_models_dir / f"{model_name}_optuna_study.joblib")

    logger.info(
        "[%s] Done in %.2f s (%.1f min) | best_cv_%s=%.4f | val_f1=%.4f | test_f1=%.4f",
        model_name, search_time, search_time / 60, SCORING,
        study.best_value,
        validation_metrics.get("f1", float("nan")),
        test_metrics.get("f1", float("nan")),
    )
    logger.info("[%s] Saved best model: %s", model_name, model_path)
    logger.info("[%s] Saved Optuna study: %s", model_name, study_path)

    return {"model": model_name, "validation": validation_metrics, "test": test_metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logger(LOG_PATH)

    logger.info("Starting experiment: %s", EXPERIMENT_NAME)
    logger.info("Models: %s | Trials/model: %d | CV: %d | Subsample: %d", MODELS_TO_RUN, N_TRIALS, CV_SPLITS, SUBSAMPLE_SIZE)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Output directory: %s", output_paths["experiment_output_dir"])

    X_train, y_train, feature_names = load_window_split(DATA_DIR / "train.npz")
    X_val, y_val, _ = load_window_split(DATA_DIR / "val.npz")
    X_test, y_test, _ = load_window_split(DATA_DIR / "test.npz")
    logger.info("Train: X=%s | y=%s", X_train.shape, y_train.shape)
    logger.info("Validation: X=%s | y=%s", X_val.shape, y_val.shape)
    logger.info("Test: X=%s | y=%s", X_test.shape, y_test.shape)

    config_path = save_experiment_config(feature_names)
    logger.info("Saved experiment config: %s", config_path)

    all_validation_metrics: dict[str, dict] = {}
    all_test_metrics: dict[str, dict] = {}
    experiment_start = time.time()

    for model_name in MODELS_TO_RUN:
        result = run_model_finetuning(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
        )
        all_validation_metrics[model_name] = result["validation"]
        all_test_metrics[model_name] = result["test"]

    save_metrics_table(all_validation_metrics, output_paths["reports_dir"] / "all_models_validation_metrics.csv")
    save_metrics_table(all_test_metrics, output_paths["reports_dir"] / "all_models_test_metrics.csv")

    total_time = time.time() - experiment_start
    logger.info("=" * 60)
    logger.info("Experiment finished in %.2f s (%.1f min)", total_time, total_time / 60)
    logger.info("Consolidated validation: %s", output_paths["reports_dir"] / "all_models_validation_metrics.csv")
    logger.info("Consolidated test: %s", output_paths["reports_dir"] / "all_models_test_metrics.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()