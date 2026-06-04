from __future__ import annotations

import argparse
import time
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from configs import paths
from old.tep import load_window_split
from src.modeling.evaluation import evaluate_binary_classifier, save_metrics_table
from src.modeling.persistence import save_joblib

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one or more binary time-series classifiers from saved TEP windows.")
    parser.add_argument("--data-dir", type=Path, default=paths.FINAL_DATA_DIR / "tep_binary_windows")
    parser.add_argument("--output-dir", type=Path, default=paths.MODELS_DIR / "tep_binary")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest"],
        help=f"Models: {', '.join(MODEL_NAMES)}",
    )
    parser.add_argument("--no-scaler", action="store_true", help="Disable StandardScaler pipeline step.")
    parser.add_argument("--random-state", type=int, default=paths.RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = load_window_split(args.data_dir / "train.npz")
    X_val, y_val, _ = load_window_split(args.data_dir / "val.npz")
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    results = {}
    for model_name in args.models:
        estimator = build_model(model_name, args.random_state)
        steps = []
        if not args.no_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        model = Pipeline(steps)

        print(f"\nTraining {model_name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        metrics = evaluate_binary_classifier(model, X_val, y_val, model_name, args.output_dir / "validation")
        metrics["train_time_seconds"] = train_time
        results[model_name] = metrics

        model_path = args.output_dir / f"{model_name}.joblib"
        save_joblib(model, model_path)
        print(f"Saved model: {model_path}")

    metrics_df = save_metrics_table(results, args.output_dir / "validation_metrics.csv")
    print("\nValidation summary:")
    print(metrics_df.round(4).to_string())


if __name__ == "__main__":
    main()
