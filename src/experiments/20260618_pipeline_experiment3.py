from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.logger import setup_logger
from configs.paths import create_output_dirs
from configs.tep_config import TEPConfig, TEPDatasetPaths
from src.data_handling.data_loader import (
    choose_simulations,
    infer_feature_columns,
    load_selected_simulations,
    split_by_simulation,
)
from src.modeling.persistence import save_json, save_joblib


EXPERIMENT_NAME = "20260618_pipeline_experiment3"
MODEL_NAME = "random_forest"

ALL_FAULTS = tuple(range(1, 21))
EXCLUDED_FAULTS = (5, 9, 15)
FAULTS_TO_USE = tuple(fault for fault in ALL_FAULTS if fault not in EXCLUDED_FAULTS)


def load_multiclass_faults(
    path: Path,
    n_sims: int | None,
    cfg: TEPConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sim_ids = choose_simulations(path, n_sims, rng)
    df = load_selected_simulations(
        path=path,
        sim_ids=sim_ids,
        chunksize=cfg.chunksize,
    )
    df = df[df["faultNumber"].isin(FAULTS_TO_USE)].copy()

    counts = df["faultNumber"].value_counts().sort_index()
    missing_faults = sorted(set(FAULTS_TO_USE) - set(counts.index))
    if missing_faults:
        raise ValueError(f"Missing fault classes after loading: {missing_faults}")

    if counts.nunique() != 1:
        min_count = int(counts.min())
        df = (
            df.groupby("faultNumber", group_keys=False)
            .sample(n=min_count, random_state=cfg.random_state)
            .reset_index(drop=True)
        )

    return df.reset_index(drop=True)


def save_class_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    distribution = (
        df["faultNumber"]
        .value_counts()
        .sort_index()
        .rename_axis("faultNumber")
        .reset_index(name="n_rows")
    )
    distribution.to_csv(output_path, index=False)
    return output_path


def evaluate_multiclass_classifier(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    output_dir: Path,
) -> dict[str, float]:
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
    }

    labels = list(FAULTS_TO_USE)
    target_names = [f"Fault {fault}" for fault in labels]
    report = classification_report(
        y,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    print(f"\n{split_name.upper()} REPORT")
    print(report)
    for name, value in metrics.items():
        print(f"{split_name}_{name}: {value:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = MODEL_NAME

    report_path = output_dir / f"{safe_name}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.6f}\n")

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        labels=labels,
        display_labels=labels,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {MODEL_NAME} - {split_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    return metrics


def main() -> None:
    output_paths = create_output_dirs(EXPERIMENT_NAME)
    logger = setup_logger(output_paths["logs_dir"] / "experiment.log")

    paths = TEPDatasetPaths()
    cfg = TEPConfig()
    rng = np.random.default_rng(cfg.random_state)

    logger.info("Starting multiclass fault experiment: %s", EXPERIMENT_NAME)
    logger.info("Using faults: %s | Excluded faults: %s", FAULTS_TO_USE, EXCLUDED_FAULTS)

    trainval_df = load_multiclass_faults(paths.fa_train, cfg.n_fa_train, cfg, rng)
    test_df = load_multiclass_faults(paths.fa_test, cfg.n_fa_test, cfg, rng)

    feature_columns = infer_feature_columns(trainval_df, cfg.meta_cols)

    train_df, val_df = split_by_simulation(
        trainval_df,
        cfg.val_ratio,
        rng,
        label_col="faultNumber",
    )

    save_class_distribution(train_df, output_paths["reports_dir"] / "train_class_distribution.csv")
    save_class_distribution(val_df, output_paths["reports_dir"] / "validation_class_distribution.csv")
    save_class_distribution(test_df, output_paths["reports_dir"] / "test_class_distribution.csv")

    X_train = train_df[feature_columns]
    y_train = train_df["faultNumber"]
    X_val = val_df[feature_columns]
    y_val = val_df["faultNumber"]
    X_test = test_df[feature_columns]
    y_test = test_df["faultNumber"]

    logger.info("Train: X=%s | y=%s", X_train.shape, y_train.shape)
    logger.info("Validation: X=%s | y=%s", X_val.shape, y_val.shape)
    logger.info("Test: X=%s | y=%s", X_test.shape, y_test.shape)

    save_json(
        {
            "experiment_name": EXPERIMENT_NAME,
            "model": MODEL_NAME,
            "dataset_paths": {name: str(path) for name, path in asdict(paths).items()},
            "tep_config": asdict(cfg),
            "faults": {
                "all": list(ALL_FAULTS),
                "excluded": list(EXCLUDED_FAULTS),
                "used": list(FAULTS_TO_USE),
            },
            "n_classes": len(FAULTS_TO_USE),
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
        },
        output_paths["reports_dir"] / "experiment_config.json",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    logger.info("Training Random Forest...")
    model.fit(X_train, y_train)

    logger.info("Evaluating Random Forest...")
    val_metrics = evaluate_multiclass_classifier(
        model,
        X_val,
        y_val,
        "validation",
        output_paths["validation_reports_dir"],
    )
    test_metrics = evaluate_multiclass_classifier(
        model,
        X_test,
        y_test,
        "test",
        output_paths["test_reports_dir"],
    )

    pd.DataFrame({"validation": val_metrics, "test": test_metrics}).T.to_csv(
        output_paths["reports_dir"] / "metrics.csv"
    )

    model_path = save_joblib(model, output_paths["models_dir"] / f"{MODEL_NAME}.joblib")
    logger.info("Saved model: %s", model_path)
    logger.info("Experiment finished.")


if __name__ == "__main__":
    main()
