"""Build and sanity-check the symmetric matchup feature table.

Run from the repo root (venv active):
    python scripts/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (  # noqa: E402
    FEATURE_TABLE_COLS,
    METADATA_NAMES,
    build_feature_table,
    chronological_split,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main() -> None:
    features, stats = build_feature_table()

    print("=" * 78)
    print("PIPELINE COUNTS")
    print("=" * 78)
    print(f"raw fights                     : {stats['n_fights_raw']}")
    print(f"rows after join resolution     : {stats['n_joined_rows']}"
          f"   (dropped unresolved: {stats['n_dropped_unresolved']})")
    print(f"dropped non-binary (draw/NC)   : {stats['n_dropped_nonbinary']}")
    print(f"final feature rows             : {stats['n_final']}")

    print("\n" + "=" * 78)
    print(f"FEATURE TABLE: shape = {features.shape}")
    print("=" * 78)
    print("\n--- dtypes ---")
    print(features.dtypes.to_string())

    # The table's columns should be exactly metadata + feature-table + label.
    # A stray column means something was added without deciding which side of
    # the model boundary it falls on.
    expected = set(METADATA_NAMES) | set(FEATURE_TABLE_COLS) | {"label"}
    actual = set(features.columns)
    print("\n--- column contract ---")
    print(f"unexpected columns: {sorted(actual - expected) or 'none'}")
    print(f"missing columns   : {sorted(expected - actual) or 'none'}")

    # Scope the zero-NaN requirement to what the model actually consumes.
    # The table also carries METADATA_NAMES (ids, Method, End_Round); a NaN
    # there is worth seeing but is not a modelling defect, so reporting it as
    # "PROBLEM" alongside the feature columns would cry wolf.
    model_cols = FEATURE_TABLE_COLS + ["label"]
    print("\n--- missing per feature column (post-imputation) ---")
    miss = features[model_cols].isna().sum()
    total_na = int(miss.sum())
    print(miss[miss > 0].to_string() if miss.any() else "(none)")
    print(f"\nfinal row count: {len(features)}  (expected 8400, nothing dropped)")
    print(f"NaNs across feature+label columns: {total_na}  -> "
          f"{'OK, none remain' if total_na == 0 else 'PROBLEM'}")

    meta_miss = features[METADATA_NAMES].isna().sum()
    meta_total = int(meta_miss.sum())
    print(f"NaNs across metadata columns     : {meta_total}"
          + ("" if meta_total == 0 else "  (informational, not model inputs)"))
    if meta_total:
        print(meta_miss[meta_miss > 0].to_string())

    print("\n--- missing-flag counts (how many were imputed to 0) ---")
    for col in ("reach_diff_missing", "height_diff_missing", "age_diff_missing"):
        print(f"{col:22s}: {int(features[col].sum())}")

    print("\n--- 6 sample rows ---")
    print(features.head(6).to_string())

    print("\n--- stance_matchup distribution ---")
    print(features["stance_matchup"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 78)
    print("POSITION-BIAS FIX — sanity check")
    print("=" * 78)
    a_rate = features["label"].mean()
    print(f"A win rate = {a_rate:.4f}  (was ~0.63 for Fighter_1; want ~0.50)")

    print("\n" + "=" * 78)
    print("CHRONOLOGICAL TRAIN/TEST SPLIT (most recent ~18% held out)")
    print("=" * 78)
    train, test = chronological_split(features, test_frac=0.18)
    n = len(features)
    print(f"train: {len(train):5d} rows ({len(train)/n:.1%})  "
          f"{train['Event_Date'].min().date()} -> {train['Event_Date'].max().date()}")
    print(f"test : {len(test):5d} rows ({len(test)/n:.1%})  "
          f"{test['Event_Date'].min().date()} -> {test['Event_Date'].max().date()}")
    print(f"boundary: no train fight is later than the first test fight -> "
          f"{train['Event_Date'].max() <= test['Event_Date'].min()}")
    print(f"train A-win rate: {train['label'].mean():.4f}   "
          f"test A-win rate: {test['label'].mean():.4f}")


if __name__ == "__main__":
    main()
