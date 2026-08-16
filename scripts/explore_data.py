"""Step 1 sanity check: load, join, and describe the UFC data.

Run from the repo root (with the venv active):
    python scripts/explore_data.py

Prints shape, dtypes, per-column missing counts, and a couple of sample
joined rows for each table — plus a few flags about things that will
complicate a chronological train/test split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import load_fights, load_fighters, join_fighter_bios  # noqa: E402

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def describe(name: str, df: pd.DataFrame, sample: int = 3) -> None:
    print("\n" + "=" * 78)
    print(f"{name}: shape = {df.shape}")
    print("=" * 78)

    print("\n--- dtypes ---")
    print(df.dtypes.to_string())

    print("\n--- missing values per column (only columns with any) ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("(none)")
    else:
        pct = (missing / len(df) * 100).round(1)
        print(pd.DataFrame({"missing": missing, "pct": pct}).to_string())

    print(f"\n--- {sample} sample rows ---")
    print(df.head(sample).to_string())


def flag_chronological_leakage(
    fights: pd.DataFrame, fighters: pd.DataFrame, joined: pd.DataFrame
) -> None:
    print("\n" + "#" * 78)
    print("# FLAGS for a chronological train/test split")
    print("#" * 78)

    dates = fights["Event_Date"]
    print(f"\nDate range: {dates.min().date()} -> {dates.max().date()}"
          f"  ({dates.isna().sum()} unparseable dates)")

    print(
        "\n[1] LEAKAGE — career-average bio columns are point-in-TODAY snapshots.\n"
        "    Wins/Losses/Draws, SLpM, Str_Acc, SApM, Str_Def, TD_Avg, TD_Acc,\n"
        "    TD_Def, Sub_Avg in ufc_fighters_final.csv are each fighter's CURRENT\n"
        "    career totals/averages, not their stats as-of the fight date. Joined\n"
        "    onto a 1994 fight, they already 'know' the outcome of fights that\n"
        "    happened decades later. Using them raw leaks future info into the\n"
        "    past. Real fix: rebuild these as running, as-of-fight aggregates from\n"
        "    the per-fight F1_/F2_ columns, ordered by Event_Date."
    )

    # The per-fight F1_/F2_ columns ARE point-in-time (they describe that bout),
    # so they're safe to aggregate forward — unlike the bio snapshots.
    winner = fights["Winner"].astype("string")
    is_f1 = winner == fights["Fighter_1"]
    is_f2 = winner == fights["Fighter_2"]
    non_decisive = (~is_f1 & ~is_f2).sum()
    print(
        f"\n[2] LABEL — 'Winner' is a name, not a class. It matches Fighter_1 in "
        f"{is_f1.sum()} rows, Fighter_2 in {is_f2.sum()} rows, and NEITHER in "
        f"{non_decisive} rows (draws / no-contests / name mismatches). Those "
        f"non-decisive rows need an explicit policy before you can build a binary label."
    )

    unmatched_f1 = joined["F1_bio_Fighter_Name"].isna().sum()
    unmatched_f2 = joined["F2_bio_Fighter_Name"].isna().sum()
    row_inflation = len(joined) - len(fights)
    dup_names = fighters["Fighter_Name"].duplicated().sum()
    print(
        f"\n[3] JOIN INTEGRITY — the join produced {len(joined)} rows from "
        f"{len(fights)} fights (+{row_inflation}). Cause: {dup_names} duplicate "
        f"Fighter_Name values in the profile table, so a name-based join fans a "
        f"fight out into multiple rows. Unmatched bios: F1 {unmatched_f1}, F2 "
        f"{unmatched_f2} (near-zero only because duplicates happen to cover the "
        f"gaps). Name joins are fragile (duplicates, spelling, accents) — a stable "
        f"fighter ID, or de-duping profiles before the merge, would be safer."
    )

    print(
        f"\n[4] POSITION BIAS — Fighter_1 wins {is_f1.sum()} vs Fighter_2 "
        f"{is_f2.sum()} ({is_f1.mean():.0%} of decisive fights). Whoever built the "
        f"dataset tends to list the winner first, so 'always pick Fighter_1' is a "
        f"~64% baseline and any raw F1/F2 feature encodes the answer. The matchup "
        f"features must be symmetric (diffs) and the F1/F2 assignment randomized."
    )

    print(
        "\n[5] STANCE — Stance is categorical with blanks; the stance_matchup "
        "feature must handle missing/unknown stances explicitly."
    )


def main() -> None:
    fights = load_fights()
    fighters = load_fighters()
    joined = join_fighter_bios(fights, fighters)

    describe("FIGHTS (raw)", fights)
    describe("FIGHTERS (parsed bios)", fighters)
    describe("JOINED (fights + both bios)", joined, sample=2)
    flag_chronological_leakage(fights, fighters, joined)


if __name__ == "__main__":
    main()
