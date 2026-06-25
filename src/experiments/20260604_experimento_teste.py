from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from configs import paths, settings
from configs.logger import setup_logger
from src.data_handling.make_dataset import load_window_split
from src.modeling.evaluation import evaluate_binary_classifier, save_metrics_table
from src.modeling.persistence import save_joblib, save_json

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


MODEL_NAMES = (
    "logistic",
    "sgd",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "mlp",
    "xgboost",
    "lightgbm",
)

EXPERIMENT_NAME = "20260604_experimento_teste"
DATA_DIR = paths.FINAL_DATA_DIR
OUTPUT_DIR = paths.OUTPUTS_DIR / EXPERIMENT_NAME
LOG_PATH = paths.LOGS_DIR / f"{EXPERIMENT_NAME}.log"


USE_SCALER = True
EXPERIMENT_MODELS = MODEL_NAMES

logger = logging.getLogger(__name__)


def build_model(name: str, random_state: int):
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Install it with: poetry add xgboost")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )

    if name == "lightgbm":
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

    models = {
        "logistic": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state),
        "sgd": SGDClassifier(loss="log_loss", random_state=random_state),
        "linear_svm": LinearSVC(max_iter=3000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            batch_size=512,
            max_iter=100,
            early_stopping=True,
            random_state=random_state,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Options: {', '.join(MODEL_NAMES)}")
    return models[name]


def create_output_dirs() -> None:
    for output_dir in [
        OUTPUT_DIR,
        paths.MODELS_DIR,
        paths.REPORTS_DIR,
        paths.REPORTS_DIR / "validation",
        paths.REPORTS_DIR / "test",
        paths.LOGS_DIR,
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)


def build_pipeline(model_name: str):
    estimator = build_model(model_name, settings.RANDOM_STATE)
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def save_experiment_config(feature_names: list[str]) -> Path:
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "models_dir": str(paths.MODELS_DIR),
        "reports_dir": str(paths.REPORTS_DIR),
        "logs_dir": str(paths.LOGS_DIR),
        "log_path": str(LOG_PATH),
        "random_state": settings.RANDOM_STATE,
        "use_scaler": USE_SCALER,
        "models": list(EXPERIMENT_MODELS),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    return save_json(payload, paths.REPORTS_DIR / "experiment_config.json")


def main() -> None:
    create_output_dirs()
    setup_logger(LOG_PATH)

    logger.info("Starting experiment: %s", EXPERIMENT_NAME)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Log path: %s", LOG_PATH)

    X_train, y_train, feature_names = load_window_split(DATA_DIR / "train.npz")
    X_val, y_val, _ = load_window_split(DATA_DIR / "val.npz")
    X_test, y_test, _ = load_window_split(DATA_DIR / "test.npz")
    logger.info("Train: X=%s | y=%s", X_train.shape, y_train.shape)
    logger.info("Validation: X=%s | y=%s", X_val.shape, y_val.shape)
    logger.info("Test: X=%s | y=%s", X_test.shape, y_test.shape)

    config_path = save_experiment_config(feature_names)
    logger.info("Saved experiment config: %s", config_path)

    validation_results = {}
    test_results = {}
    failures = {}
    for model_name in EXPERIMENT_MODELS:
        logger.info("Training model: %s", model_name)

        start = time.time()
        try:
            model = build_pipeline(model_name)
            model.fit(X_train, y_train)
            train_time = time.time() - start

            validation_metrics = evaluate_binary_classifier(
                model,
                X_val,
                y_val,
                model_name,
                paths.REPORTS_DIR / "validation",
            )
            validation_metrics["train_time_seconds"] = train_time
            validation_results[model_name] = validation_metrics

            test_metrics = evaluate_binary_classifier(
                model,
                X_test,
                y_test,
                model_name,
                paths.REPORTS_DIR / "test",
            )
            test_metrics["train_time_seconds"] = train_time
            test_results[model_name] = test_metrics

            model_path = paths.MODELS_DIR / f"{model_name}.joblib"
            save_joblib(model, model_path)
            logger.info(
                "Finished %s in %.2f seconds | validation_f1=%.4f | test_f1=%.4f | model=%s",
                model_name,
                train_time,
                validation_metrics.get("f1", float("nan")),
                test_metrics.get("f1", float("nan")),
                model_path,
            )
        except Exception as exc:
            failures[model_name] = repr(exc)
            logger.exception("Model %s failed and will be skipped.", model_name)

    if failures:
        failures_path = save_json(failures, paths.REPORTS_DIR / "failures.json")
        logger.warning("Some models failed. Details saved to %s", failures_path)

    if not validation_results:
        raise RuntimeError("No model finished successfully. Check the experiment log.")

    validation_metrics_df = save_metrics_table(
        validation_results,
        paths.REPORTS_DIR / "validation_metrics.csv",
    )
    test_metrics_df = save_metrics_table(
        test_results,
        paths.REPORTS_DIR / "test_metrics.csv",
    )

    logger.info("Validation summary:\n%s", validation_metrics_df.round(4).to_string())
    logger.info("Test summary:\n%s", test_metrics_df.round(4).to_string())
    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
