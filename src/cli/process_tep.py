from __future__ import annotations

import argparse
from pathlib import Path

from configs import paths
from src.data_handling.preprocess import preprocess


DEFAULT_PAIRS = [
    (paths.FAULT_FREE_TRAIN, paths.PROCESSED_FF_TRAIN),
    (paths.FAULTY_TRAIN, paths.PROCESSED_FA_TRAIN),
    (paths.FAULT_FREE_TEST, paths.PROCESSED_FF_TEST),
    (paths.FAULTY_TEST, paths.PROCESSED_FA_TEST),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename TEP columns and save processed CSV files.")
    parser.add_argument("--input", type=Path, help="Single input CSV to process.")
    parser.add_argument("--output", type=Path, help="Output CSV path for --input.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process the four canonical raw TEP CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths.ensure_project_dirs()

    if args.input or args.output:
        if not args.input or not args.output:
            raise SystemExit("--input and --output must be used together.")
        preprocess(args.input, args.output)
        return

    if not args.all:
        raise SystemExit("Use --all or pass --input and --output.")

    for input_path, output_path in DEFAULT_PAIRS:
        preprocess(input_path, output_path)


if __name__ == "__main__":
    main()
