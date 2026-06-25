from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
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


EXPERIMENT_NAME = "20260610_finetuning_all_models"

DATA_DIR = paths.FINAL_DATA_DIR

MODELS_TO_RUN = ["lightgbm", "random_forest", "xgboost"]
N_TRIALS = 60
CV_SPLITS = 3
SCORING = "f1"
N_JOBS = -1
USE_SCALER = True
OPTUNA_DIRECTION = "maximize"

logger = logging.getLogger(__name__)

output_paths = paths.create_output_dirs(EXPERIMENT_NAME)
LOG_PATH = output_paths["logs_dir"] / f"{EXPERIMENT_NAME}.log"


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def make_lightgbm_objective(X_train, y_train, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        if LGBMClassifier is None:
            raise ImportError("LightGBM não está instalado. Instale com: poetry add lightgbm")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
            "n_jobs": -1,
            "random_state": settings.RANDOM_STATE,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        pipeline = _build_pipeline(model)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=N_JOBS)
        return scores.mean()

    return objective


def make_random_forest_objective(X_train, y_train, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            "n_jobs": -1,
            "random_state": settings.RANDOM_STATE,
        }

        model = RandomForestClassifier(**params)
        pipeline = _build_pipeline(model)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        return scores.mean()

    return objective


def make_xgboost_objective(X_train, y_train, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        if XGBClassifier is None:
            raise ImportError("XGBoost não está instalado. Instale com: poetry add xgboost")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "n_jobs": -1,
            "random_state": settings.RANDOM_STATE,
            "eval_metric": "logloss",
            "verbosity": 0,
        }

        model = XGBClassifier(**params)
        pipeline = _build_pipeline(model)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=N_JOBS)
        return scores.mean()

    return objective


OBJECTIVES = {
    "lightgbm": make_lightgbm_objective,
    "random_forest": make_random_forest_objective,
    "xgboost": make_xgboost_objective,
}

MODEL_BUILDERS = {
    "lightgbm": lambda params: LGBMClassifier(**params, n_jobs=-1, random_state=settings.RANDOM_STATE, verbose=-1),
    "random_forest": lambda params: RandomForestClassifier(**params, n_jobs=-1, random_state=settings.RANDOM_STATE),
    "xgboost": lambda params: XGBClassifier(**params, n_jobs=-1, random_state=settings.RANDOM_STATE, eval_metric="logloss", verbosity=0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pipeline(model) -> Pipeline:
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _rebuild_best_pipeline(model_name: str, best_params: dict) -> Pipeline:
    """Reconstrói o pipeline com os melhores hiperparâmetros para avaliação final."""
    model = MODEL_BUILDERS[model_name](best_params)
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
            "type": "Optuna",
            "n_trials": N_TRIALS,
            "cv_splits": CV_SPLITS,
            "scoring": SCORING,
            "direction": OPTUNA_DIRECTION,
        },
    }
    return save_json(payload, output_paths["reports_dir"] / "experiment_config.json")


def save_optuna_trials(study: optuna.Study, reports_dir: Path) -> Path:
    """Salva todos os trials do study como CSV, ordenados por valor."""
    rows = [
        {
            "trial_number": t.number,
            "value": t.value,
            "state": t.state.name,
            "duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds()
            if t.datetime_complete and t.datetime_start
            else None,
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
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names: list[str],
) -> dict:
    """Roda o finetuning com Optuna de um único modelo e retorna as métricas."""

    model_reports_dir = output_paths["reports_dir"] / model_name
    model_models_dir = output_paths["models_dir"] / model_name
    model_reports_dir.mkdir(parents=True, exist_ok=True)
    model_models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Optuna finetuning: %s", model_name.upper())
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=settings.RANDOM_STATE)
    objective = OBJECTIVES[model_name](X_train, y_train, cv)

    # Silencia os logs internos do Optuna — o nosso logger já cobre tudo
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        study_name=f"{EXPERIMENT_NAME}_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    start = time.time()
    logger.info("[%s] Running Optuna study — %s trials | %s-fold CV", model_name, N_TRIALS, CV_SPLITS)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    search_time = time.time() - start

    # Trials
    trials_path = save_optuna_trials(study, model_reports_dir)
    logger.info("[%s] Saved Optuna trials: %s", model_name, trials_path)

    # Best params
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

    # Reconstrói e treina o pipeline final com todos os dados de treino
    logger.info("[%s] Retraining best model on full training set...", model_name)
    best_pipeline = _rebuild_best_pipeline(model_name, best_params)
    best_pipeline.fit(X_train, y_train)

    # Avaliação
    validation_metrics = evaluate_binary_classifier(
        best_pipeline,
        X_val,
        y_val,
        model_name,
        model_reports_dir / "validation",
    )
    validation_metrics["search_time_seconds"] = search_time
    validation_metrics["best_cv_score"] = float(study.best_value)

    test_metrics = evaluate_binary_classifier(
        best_pipeline,
        X_test,
        y_test,
        model_name,
        model_reports_dir / "test",
    )
    test_metrics["search_time_seconds"] = search_time
    test_metrics["best_cv_score"] = float(study.best_value)

    save_metrics_table({model_name: validation_metrics}, model_reports_dir / "validation_metrics.csv")
    save_metrics_table({model_name: test_metrics}, model_reports_dir / "test_metrics.csv")

    # Persistência
    model_path = save_joblib(best_pipeline, model_models_dir / f"{model_name}_best.joblib")
    study_path = save_joblib(study, model_models_dir / f"{model_name}_optuna_study.joblib")

    logger.info(
        "[%s] Done in %.2f s | best_cv_%s=%.4f | val_f1=%.4f | test_f1=%.4f",
        model_name,
        search_time,
        SCORING,
        study.best_value,
        validation_metrics.get("f1", float("nan")),
        test_metrics.get("f1", float("nan")),
    )
    logger.info("[%s] Saved best model: %s", model_name, model_path)
    logger.info("[%s] Saved Optuna study: %s", model_name, study_path)

    return {
        "model": model_name,
        "validation": validation_metrics,
        "test": test_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logger(LOG_PATH)

    logger.info("Starting experiment: %s", EXPERIMENT_NAME)
    logger.info("Models to run: %s", MODELS_TO_RUN)
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
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
        )
        all_validation_metrics[model_name] = result["validation"]
        all_test_metrics[model_name] = result["test"]

    # Tabelas comparativas consolidadas
    save_metrics_table(all_validation_metrics, output_paths["reports_dir"] / "all_models_validation_metrics.csv")
    save_metrics_table(all_test_metrics, output_paths["reports_dir"] / "all_models_test_metrics.csv")

    total_time = time.time() - experiment_start
    logger.info("=" * 60)
    logger.info("Experiment finished in %.2f seconds (%.1f min)", total_time, total_time / 60)
    logger.info("Consolidated validation metrics: %s", output_paths["reports_dir"] / "all_models_validation_metrics.csv")
    logger.info("Consolidated test metrics: %s", output_paths["reports_dir"] / "all_models_test_metrics.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()