from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)


def get_positive_scores(model, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        if len(np.unique(y_true)) < 2:
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            metrics["average_precision"] = average_precision_score(y_true, y_score)
    return metrics


def evaluate_binary_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> dict[str, float]:
    y_pred = model.predict(X)
    y_score = get_positive_scores(model, X)
    metrics = compute_binary_metrics(y, y_pred, y_score)

    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Fault"],
            digits=4,
            zero_division=0,
        )
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if output_dir is None:
        return metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    report_path = output_dir / f"{safe_name}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y,
                y_pred,
                labels=[0, 1],
                target_names=["Normal", "Fault"],
                digits=4,
                zero_division=0,
            )
        )
        f.write("\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        labels=[0, 1],
        display_labels=["Normal", "Fault"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    if y_score is not None:
        if len(np.unique(y)) == 2:
            fig, ax = plt.subplots(figsize=(7, 6))
            RocCurveDisplay.from_predictions(y, y_score, ax=ax)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_title(f"ROC Curve - {model_name}")
            fig.tight_layout()
            fig.savefig(output_dir / f"{safe_name}_roc_curve.png", dpi=180)
            plt.close(fig)

        if len(np.unique(y)) == 2:
            precision, recall, _ = precision_recall_curve(y, y_score)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(recall, precision)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve - {model_name}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{safe_name}_pr_curve.png", dpi=180)
            plt.close(fig)

    return metrics


def save_metrics_table(results: dict[str, dict[str, float]], output_path: Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T
    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False)
    df.to_csv(output_path)
    return df
