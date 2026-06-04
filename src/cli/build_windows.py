from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from configs import paths, settings
from old.tep import TEPDatasetPaths, TEPWindowConfig, build_binary_window_splits, save_window_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable binary TEP sliding-window datasets.")
    parser.add_argument("--output-dir", type=Path, default=paths.FINAL_DATA_DIR / "tep_binary_windows")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--mode", choices=["stats", "flatten"], default="stats")
    parser.add_argument("--n-ff-train", type=int, default=120)
    parser.add_argument("--n-fa-train", type=int, default=6)
    parser.add_argument("--n-ff-test", type=int, default=60)
    parser.add_argument("--n-fa-test", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=settings.RANDOM_STATE)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--ff-train", type=Path, default=paths.FAULT_FREE_TRAIN)
    parser.add_argument("--fa-train", type=Path, default=paths.FAULTY_TRAIN)
    parser.add_argument("--ff-test", type=Path, default=paths.FAULT_FREE_TEST)
    parser.add_argument("--fa-test", type=Path, default=paths.FAULTY_TEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = TEPDatasetPaths(
        ff_train=args.ff_train,
        fa_train=args.fa_train,
        ff_test=args.ff_test,
        fa_test=args.fa_test,
    )
    window_cfg = TEPWindowConfig(
        window_size=args.window_size,
        stride=args.stride,
        mode=args.mode,
        n_ff_train=args.n_ff_train,
        n_fa_train=args.n_fa_train,
        n_ff_test=args.n_ff_test,
        n_fa_test=args.n_fa_test,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        chunksize=args.chunksize,
    )

    print("Building TEP binary window datasets...")
    dataset = build_binary_window_splits(paths, window_cfg)
    save_window_splits(dataset, args.output_dir)

    for split in ["train", "val", "test"]:
        X, y = dataset[split]
        labels, values = np.unique(y, return_counts=True)
        counts = {int(label): int(value) for label, value in zip(labels, values)}
        print(f"{split}: X={X.shape}, y={counts}")
    print(f"Saved datasets to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
