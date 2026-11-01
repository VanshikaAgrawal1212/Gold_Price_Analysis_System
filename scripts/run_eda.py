"""Standalone EDA runner - step 3 of the write-up.

    py scripts/run_eda.py            # live data
    py scripts/run_eda.py --synthetic

Prints the data-quality tables to the console and writes every figure plus the
cleaned frame, so you can eyeball the dataset before any model is fitted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from configs import settings as S  # noqa: E402
from src.data_loader import build_dataset, synthetic_dataset  # noqa: E402
from src.eda import (  # noqa: E402
    clean,
    correlation_matrix,
    longest_gap,
    quality_report,
    run_eda,
    stationary_adf_test,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory analysis of the gold dataset")
    ap.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--no-cache", action="store_true", help="force a fresh download")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    raw = synthetic_dataset() if args.synthetic else build_dataset(use_cache=not args.no_cache)

    print("\n" + "=" * 78)
    print("1. RAW SHAPE / DATE RANGE")
    print("=" * 78)
    print(f"rows={len(raw)}  cols={raw.shape[1]}  "
          f"range={raw.index.min():%Y-%m-%d} -> {raw.index.max():%Y-%m-%d}")
    print(f"total missing cells: {int(raw.isna().sum().sum())}")

    print("\n" + "=" * 78)
    print("2. LONGEST CONSECUTIVE NaN RUN PER COLUMN (holiday vs. broken feed)")
    print("=" * 78)
    print(longest_gap(raw).to_string())

    print("\n" + "=" * 78)
    print("3. CLEANING")
    print("=" * 78)
    tidy = clean(raw)
    print(quality_report(tidy).to_string())

    print("\n" + "=" * 78)
    print("4. SPEARMAN CORRELATION OF LEVELS (rank-based, tolerant of outliers)")
    print("=" * 78)
    print(correlation_matrix(tidy, method="spearman").round(2).to_string())

    print("\n" + "=" * 78)
    print(f"5. CORRELATION WITH {S.TARGET_COL} - ranked")
    print("=" * 78)
    print(correlation_matrix(tidy, method="spearman")[S.TARGET_COL]
          .dropna().sort_values(ascending=False).round(3).to_string())

    adf = stationary_adf_test(tidy)
    if not adf.empty:
        print("\n" + "=" * 78)
        print("6. AUGMENTED DICKEY-FULLER (p<0.05 => stationary)")
        print("=" * 78)
        print(adf.to_string(index=False))

    print("\n" + "=" * 78)
    print("7. WRITING FIGURES")
    print("=" * 78)
    paths = run_eda(raw)
    for k, v in paths.items():
        print(f"  {k:<16} {v}")

    print("\nRead the figures in ./figures, then train:  py -m src.pipeline")


if __name__ == "__main__":
    main()
