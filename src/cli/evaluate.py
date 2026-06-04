from __future__ import annotations

import argparse
from pathlib import Path

from configs import paths
from old.tep import load_window_split
from src.modeling.evaluation import evaluate_binary_classifier, save_metrics_table
from src.modeling.persistence import load_joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved binary classifier against a saved TEP window split.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=paths.FINAL_DATA_DIR / "tep_binary_windows" / "test.npz")
    parser.add_argument("--output-dir", type=Path, default=paths.REPORTS_DIR / "tep_binary")
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model_name or args.model_path.stem
    model = load_joblib(args.model_path)
    X, y, _ = load_window_split(args.data_path)
    print(f"Evaluating {model_name}: X={X.shape}")
    metrics = evaluate_binary_classifier(model, X, y, model_name, args.output_dir)
    save_metrics_table({model_name: metrics}, args.output_dir / f"{model_name}_metrics.csv")
    print(f"Saved evaluation artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
