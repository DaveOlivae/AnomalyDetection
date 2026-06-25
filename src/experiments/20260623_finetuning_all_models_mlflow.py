from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

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
from configs.mlflow_config import setup_mlflow
from src.data_handling.make_dataset import load_window_split
from src.modeling.evaluation import evaluate_binary_classifier, save_metrics_table
from src.modeling.persistence import save_joblib, save_json


EXPERIMENT_NAME = "20260623_finetuning_all_models_mlflow"

DATA_DIR = paths.FINAL_DATA_DIR

output_paths = paths.create_output_dirs(EXPERIMENT_NAME)

LOG_PATH = output_paths["logs_dir"] / f"{EXPERIMENT_NAME}.log"

logger = logging.getLogger(__name__)

MODELS_TO_RUN = ["lightgbm", "random_forest", "xgboost"]
N_TRIALS = 60
CV_SPLITS = 3
SCORING = "f1"
OPTUNA_DIRECTION = "maximize"

# Para modelos de árvore, scaler geralmente não ajuda e só adiciona custo.
# Se futuramente entrar SVM/LogisticRegression/MLP, pode voltar para True.
USE_SCALER = False

# Evita paralelismo duplicado: o modelo usa todos os cores e o CV roda fold a fold.
# Isso costuma ser mais estável do que CV_N_JOBS=-1 + modelo n_jobs=-1.
MODEL_N_JOBS = -1
CV_N_JOBS = 1

# MLflow
MLFLOW_ENABLED = True
MLFLOW_EXPERIMENT_NAME = EXPERIMENT_NAME
# Se quiser usar servidor/banco externo, defina a env var MLFLOW_TRACKING_URI.
# Exemplos:
#   export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
#   export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"


MLFLOW_LOG_TRIAL_RUNS = True
MLFLOW_LOG_FINAL_MODEL = True


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _require_mlflow() -> None:
    if not MLFLOW_ENABLED:
        return
    if mlflow is None:
        raise ImportError(
            "MLflow não está instalado. Instale com: poetry add mlflow "
            "ou pip install mlflow"
        )


@contextmanager
def mlflow_run(run_name: str, *, nested: bool = False, tags: dict[str, Any] | None = None) -> Iterator[Any | None]:
    """Abre uma run do MLflow quando habilitado; caso contrário, vira no-op."""
    if not MLFLOW_ENABLED:
        yield None
        return

    _require_mlflow()
    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
        yield run


def _active_mlflow_run() -> bool:
    return bool(MLFLOW_ENABLED and mlflow is not None and mlflow.active_run() is not None)


def _to_mlflow_param(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def log_params_safe(params: dict[str, Any], prefix: str = "") -> None:
    """Loga parâmetros no MLflow evitando quebrar por tipos não serializáveis."""
    if not _active_mlflow_run():
        return

    for key, value in params.items():
        mlflow.log_param(f"{prefix}{key}", _to_mlflow_param(value))


def log_metric_safe(name: str, value: Any, *, step: int | None = None) -> None:
    """Loga somente métricas numéricas válidas."""
    if not _active_mlflow_run():
        return

    if isinstance(value, (bool, str)) or value is None:
        return

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return

    if not math.isfinite(numeric_value):
        return

    mlflow.log_metric(name, numeric_value, step=step)


def log_metrics_safe(metrics: dict[str, Any], prefix: str = "") -> None:
    for key, value in metrics.items():
        log_metric_safe(f"{prefix}{key}", value)


def log_artifact_path(path: str | os.PathLike[str] | Path, artifact_path: str | None = None) -> None:
    """Loga arquivo ou diretório como artefato do MLflow se existir."""
    if not _active_mlflow_run():
        return

    try:
        artifact = Path(path)
    except TypeError:
        logger.warning(
            "MLflow artifact skipped because object is not path-like: %s",
            type(path).__name__,
        )
        return

    if not artifact.exists():
        logger.warning("MLflow artifact skipped because path does not exist: %s", artifact)
        return

    if artifact.is_dir():
        mlflow.log_artifacts(str(artifact), artifact_path=artifact_path)
    else:
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)


def log_sklearn_model(model: Pipeline, model_name: str) -> None:
    """Loga o modelo final. Compatível com versões recentes e antigas do MLflow."""
    if not (_active_mlflow_run() and MLFLOW_LOG_FINAL_MODEL):
        return

    try:
        mlflow.sklearn.log_model(sk_model=model, name=f"model_{model_name}")
    except TypeError:
        # MLflow antigo usa artifact_path em vez de name.
        mlflow.sklearn.log_model(sk_model=model, artifact_path=f"model_{model_name}")


def get_git_commit() -> str | None:
    """Tenta capturar o commit atual para reprodutibilidade."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers gerais
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


def _safe_index(data, indices):
    """Indexação compatível com numpy arrays, pandas Series/DataFrame e listas."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def run_cv_with_pruning(
    pipeline: Pipeline,
    X_train,
    y_train,
    cv: StratifiedKFold,
    trial: optuna.Trial,
) -> tuple[float, float, list[float]]:
    """
    Executa CV manualmente para permitir pruning real do Optuna.

    No script original havia MedianPruner, mas cross_val_score não reporta métricas
    intermediárias para o trial. Aqui cada fold reporta a média parcial.
    """
    scorer = get_scorer(SCORING)
    scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        fold_pipeline = clone(pipeline)

        X_fold_train = _safe_index(X_train, train_idx)
        y_fold_train = _safe_index(y_train, train_idx)
        X_fold_valid = _safe_index(X_train, valid_idx)
        y_fold_valid = _safe_index(y_train, valid_idx)

        fold_pipeline.fit(X_fold_train, y_fold_train)
        fold_score = float(scorer(fold_pipeline, X_fold_valid, y_fold_valid))
        scores.append(fold_score)

        partial_mean = float(np.mean(scores))
        log_metric_safe(f"fold_{fold_idx}_{SCORING}", fold_score, step=fold_idx)
        log_metric_safe(f"partial_mean_{SCORING}", partial_mean, step=fold_idx)

        trial.report(partial_mean, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at fold {fold_idx} with {SCORING}={partial_mean:.4f}")

    return float(np.mean(scores)), float(np.std(scores)), scores


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _evaluate_trial(
    *,
    model_name: str,
    trial: optuna.Trial,
    model,
    params: dict[str, Any],
    X_train,
    y_train,
    cv: StratifiedKFold,
) -> float:
    pipeline = _build_pipeline(model)

    if MLFLOW_LOG_TRIAL_RUNS:
        run_context = mlflow_run(
            run_name=f"{model_name}_trial_{trial.number}",
            nested=True,
            tags={
                "run_level": "trial",
                "model_name": model_name,
                "optuna_trial_number": str(trial.number),
            },
        )
    else:
        run_context = nullcontext(None)

    with run_context:
        log_params_safe(
            {
                "model_name": model_name,
                "trial_number": trial.number,
                "scoring": SCORING,
                "cv_splits": CV_SPLITS,
                "use_scaler": USE_SCALER,
                "model_n_jobs": MODEL_N_JOBS,
                "cv_n_jobs": CV_N_JOBS,
                **params,
            }
        )

        try:
            mean_score, std_score, fold_scores = run_cv_with_pruning(pipeline, X_train, y_train, cv, trial)
        except optuna.TrialPruned:
            if _active_mlflow_run():
                mlflow.set_tag("trial_state", "PRUNED")
            raise
        except Exception as exc:
            if _active_mlflow_run():
                mlflow.set_tag("trial_state", "FAIL")
                mlflow.set_tag("error", repr(exc)[:500])
            raise

        log_metric_safe(f"mean_cv_{SCORING}", mean_score)
        log_metric_safe(f"std_cv_{SCORING}", std_score)
        if _active_mlflow_run():
            mlflow.set_tag("trial_state", "COMPLETE")
            mlflow.set_tag("fold_scores", ",".join(f"{score:.6f}" for score in fold_scores))

        return mean_score


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
            "n_jobs": MODEL_N_JOBS,
            "random_state": settings.RANDOM_STATE,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        return _evaluate_trial(
            model_name="lightgbm",
            trial=trial,
            model=model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

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
            "n_jobs": MODEL_N_JOBS,
            "random_state": settings.RANDOM_STATE,
        }

        model = RandomForestClassifier(**params)
        return _evaluate_trial(
            model_name="random_forest",
            trial=trial,
            model=model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

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
            "n_jobs": MODEL_N_JOBS,
            "random_state": settings.RANDOM_STATE,
            "eval_metric": "logloss",
            "verbosity": 0,
        }

        model = XGBClassifier(**params)
        return _evaluate_trial(
            model_name="xgboost",
            trial=trial,
            model=model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

    return objective


OBJECTIVES = {
    "lightgbm": make_lightgbm_objective,
    "random_forest": make_random_forest_objective,
    "xgboost": make_xgboost_objective,
}

MODEL_BUILDERS = {
    "lightgbm": lambda params: LGBMClassifier(
        **params,
        n_jobs=MODEL_N_JOBS,
        random_state=settings.RANDOM_STATE,
        verbose=-1,
    ),
    "random_forest": lambda params: RandomForestClassifier(
        **params,
        n_jobs=MODEL_N_JOBS,
        random_state=settings.RANDOM_STATE,
    ),
    "xgboost": lambda params: XGBClassifier(
        **params,
        n_jobs=MODEL_N_JOBS,
        random_state=settings.RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    ),
}


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
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
        },
        "parallelism": {
            "model_n_jobs": MODEL_N_JOBS,
            "cv_n_jobs": CV_N_JOBS,
        },
        "mlflow": {
            "enabled": MLFLOW_ENABLED,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "tracking_uri": mlflow.get_tracking_uri() if mlflow is not None else None,
            "log_trial_runs": MLFLOW_LOG_TRIAL_RUNS,
            "log_final_model": MLFLOW_LOG_FINAL_MODEL,
        },
        "git_commit": get_git_commit(),
    }
    return save_json(payload, output_paths["reports_dir"] / "experiment_config.json")


def save_optuna_trials(study: optuna.Study, reports_dir: Path) -> Path:
    """Salva todos os trials do study como CSV, incluindo COMPLETE/PRUNED/FAIL."""
    rows = []
    for t in study.trials:
        duration = (
            (t.datetime_complete - t.datetime_start).total_seconds()
            if t.datetime_complete and t.datetime_start
            else None
        )
        rows.append(
            {
                "trial_number": t.number,
                "value": t.value,
                "state": t.state.name,
                "duration_seconds": duration,
                **{f"param_{k}": v for k, v in t.params.items()},
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "value" in df.columns:
        df = df.sort_values("value", ascending=False, na_position="last")

    path = reports_dir / "optuna_trials.csv"
    df.to_csv(path, index=False)
    return path


def count_completed_trials(study: optuna.Study) -> int:
    return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])


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

    with mlflow_run(
        run_name=f"{model_name}_optuna_search",
        nested=True,
        tags={"run_level": "model_search", "model_name": model_name},
    ):
        log_params_safe(
            {
                "model_name": model_name,
                "n_trials": N_TRIALS,
                "cv_splits": CV_SPLITS,
                "scoring": SCORING,
                "optuna_direction": OPTUNA_DIRECTION,
                "use_scaler": USE_SCALER,
                "n_features": len(feature_names),
                "model_n_jobs": MODEL_N_JOBS,
                "cv_n_jobs": CV_N_JOBS,
            },
            prefix="search__",
        )

        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=settings.RANDOM_STATE)
        objective = OBJECTIVES[model_name](X_train, y_train, cv)

        # Silencia os logs internos do Optuna — o nosso logger já cobre tudo.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction=OPTUNA_DIRECTION,
            study_name=f"{EXPERIMENT_NAME}_{model_name}",
            sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )

        start = time.time()
        logger.info("[%s] Running Optuna study — %s trials | %s-fold CV", model_name, N_TRIALS, CV_SPLITS)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
        search_time = time.time() - start

        # Trials
        trials_path = save_optuna_trials(study, model_reports_dir)
        logger.info("[%s] Saved Optuna trials: %s", model_name, trials_path)
        log_artifact_path(trials_path, artifact_path=f"reports/{model_name}")

        # Best params
        best_params = study.best_params
        best_payload = {
            "best_score": float(study.best_value),
            "best_params": best_params,
            "best_trial": study.best_trial.number,
            "scoring": SCORING,
            "n_trials_completed": count_completed_trials(study),
            "search_time_seconds": search_time,
        }
        best_params_path = save_json(best_payload, model_reports_dir / "best_params.json")
        logger.info("[%s] Saved best params: %s", model_name, best_params_path)

        log_params_safe({f"best_{k}": v for k, v in best_params.items()})
        log_metric_safe(f"best_cv_{SCORING}", study.best_value)
        log_metric_safe("search_time_seconds", search_time)
        log_metric_safe("n_trials_completed", count_completed_trials(study))
        log_artifact_path(best_params_path, artifact_path=f"reports/{model_name}")

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

        validation_metrics_path = model_reports_dir / "validation_metrics.csv"
        test_metrics_path = model_reports_dir / "test_metrics.csv"

        save_metrics_table({model_name: validation_metrics}, validation_metrics_path)
        save_metrics_table({model_name: test_metrics}, test_metrics_path)

        log_metrics_safe(validation_metrics, prefix="validation__")
        log_metrics_safe(test_metrics, prefix="test__")
        log_artifact_path(validation_metrics_path, artifact_path=f"reports/{model_name}")
        log_artifact_path(test_metrics_path, artifact_path=f"reports/{model_name}")
        log_artifact_path(model_reports_dir / "validation", artifact_path=f"reports/{model_name}/validation")
        log_artifact_path(model_reports_dir / "test", artifact_path=f"reports/{model_name}/test")

        # Persistência local + MLflow
        model_path = save_joblib(best_pipeline, model_models_dir / f"{model_name}_best.joblib")
        study_path = save_joblib(study, model_models_dir / f"{model_name}_optuna_study.joblib")
        log_artifact_path(model_path, artifact_path=f"models/{model_name}")
        log_artifact_path(study_path, artifact_path=f"models/{model_name}")
        log_sklearn_model(best_pipeline, model_name)

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
            "model_path": str(model_path),
            "study_path": str(study_path),
            "best_cv_score": float(study.best_value),
            "best_trial": study.best_trial.number,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if MLFLOW_ENABLED:
        _require_mlflow()
        setup_mlflow(MLFLOW_EXPERIMENT_NAME)
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

    with mlflow_run(
        run_name="all_models_finetuning",
        tags={
            "run_level": "experiment",
            "experiment_name": EXPERIMENT_NAME,
            "git_commit": get_git_commit() or "unknown",
        },
    ):
        log_params_safe(
            {
                "experiment_name": EXPERIMENT_NAME,
                "data_dir": DATA_DIR,
                "models_to_run": ",".join(MODELS_TO_RUN),
                "n_trials": N_TRIALS,
                "cv_splits": CV_SPLITS,
                "scoring": SCORING,
                "optuna_direction": OPTUNA_DIRECTION,
                "use_scaler": USE_SCALER,
                "random_state": settings.RANDOM_STATE,
                "n_features": len(feature_names),
                "train_rows": X_train.shape[0],
                "val_rows": X_val.shape[0],
                "test_rows": X_test.shape[0],
            },
            prefix="experiment__",
        )
        log_artifact_path(config_path, artifact_path="config")
        log_artifact_path(LOG_PATH, artifact_path="logs")

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
        validation_table_path = output_paths["reports_dir"] / "all_models_validation_metrics.csv"
        test_table_path = output_paths["reports_dir"] / "all_models_test_metrics.csv"

        save_metrics_table(all_validation_metrics, validation_table_path)
        save_metrics_table(all_test_metrics, test_table_path)

        total_time = time.time() - experiment_start
        log_metric_safe("total_time_seconds", total_time)
        log_artifact_path(validation_table_path, artifact_path="reports/consolidated")
        log_artifact_path(test_table_path, artifact_path="reports/consolidated")
        log_artifact_path(LOG_PATH, artifact_path="logs")

        if all_validation_metrics:
            best_val_model = max(
                all_validation_metrics,
                key=lambda name: all_validation_metrics[name].get("f1", float("-inf")),
            )
            log_params_safe({"best_validation_model_by_f1": best_val_model}, prefix="summary__")
            log_metric_safe(
                "best_validation_f1",
                all_validation_metrics[best_val_model].get("f1"),
            )

        if all_test_metrics:
            best_test_model = max(
                all_test_metrics,
                key=lambda name: all_test_metrics[name].get("f1", float("-inf")),
            )
            log_params_safe({"best_test_model_by_f1": best_test_model}, prefix="summary__")
            log_metric_safe("best_test_f1", all_test_metrics[best_test_model].get("f1"))

        logger.info("=" * 60)
        logger.info("Experiment finished in %.2f seconds (%.1f min)", total_time, total_time / 60)
        logger.info("Consolidated validation metrics: %s", validation_table_path)
        logger.info("Consolidated test metrics: %s", test_table_path)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
