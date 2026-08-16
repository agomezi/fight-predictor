"""Data loading and joining for the UFC fight predictor.

This module is deliberately plain: read the two CSVs, normalize a few of the
string-encoded biometric columns into numbers, and join each fighter's bio
onto both sides of every fight. It does NOT engineer matchup features or do
anything model-specific yet — that happens in the feature pipeline.

Filenames reflect what's actually in data/ (not the names in the build plan):
    - ufc_gold_dataset_final.csv  : one row per fight
    - ufc_fighters_final.csv      : one row per fighter profile
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve data/ relative to the repo root, regardless of the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

FIGHTS_CSV = DATA_DIR / "ufc_gold_dataset_final.csv"
FIGHTERS_CSV = DATA_DIR / "ufc_fighters_final.csv"


# ---------------------------------------------------------------------------
# String -> number parsers for the fighter bio columns
# ---------------------------------------------------------------------------
def parse_height_to_inches(value: object) -> float:
    """Convert a height like `5' 11"` to inches (71.0). Blank -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    m = re.match(r"""\s*(\d+)'\s*(\d+(?:\.\d+)?)"?""", value)
    if not m:
        return np.nan
    feet, inches = int(m.group(1)), float(m.group(2))
    return feet * 12 + inches


def parse_reach_to_inches(value: object) -> float:
    """Convert a reach like `66.0"` to inches (66.0). Blank -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)", value)
    return float(m.group(1)) if m else np.nan


def parse_weight_to_lbs(value: object) -> float:
    """Convert a weight like `155 lbs.` to pounds (155.0). Blank -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)", value)
    return float(m.group(1)) if m else np.nan


def parse_percent(value: object) -> float:
    """Convert a percentage string like `38%` to a fraction (0.38). Blank -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*%", value)
    return float(m.group(1)) / 100.0 if m else np.nan


# Career-average percentage columns stored as "38%" strings.
PERCENT_COLS = ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]


def load_fighters(path: Path = FIGHTERS_CSV) -> pd.DataFrame:
    """Load the fighter-profile CSV and normalize biometric columns to numbers."""
    df = pd.read_csv(path)
    df["Height_in"] = df["Height"].apply(parse_height_to_inches)
    df["Reach_in"] = df["Reach"].apply(parse_reach_to_inches)
    df["Weight_lbs"] = df["Weight"].apply(parse_weight_to_lbs)
    for col in PERCENT_COLS:
        df[col] = df[col].apply(parse_percent)
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    return df


def load_fights(path: Path = FIGHTS_CSV) -> pd.DataFrame:
    """Load the fight CSV and parse the event date."""
    df = pd.read_csv(path)
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    return df


def join_fighter_bios(fights: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Attach each fighter's bio to both sides of every fight.

    Fighter_1's bio columns get an ``F1_bio_`` prefix, Fighter_2's an ``F2_bio_``
    prefix, so they never collide with each other or with the existing F1_/F2_
    per-fight stat columns already in the fight table.
    """
    # Columns we want to carry over from the fighter profile.
    bio_cols = [
        "Fighter_Name", "Height_in", "Reach_in", "Weight_lbs", "Stance", "DOB",
        "Wins", "Losses", "Draws",
        "SLpM", "Str_Acc", "SApM", "Str_Def", "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
    ]
    bios = fighters[bio_cols].copy()

    merged = fights.merge(
        bios.add_prefix("F1_bio_"),
        left_on="Fighter_1", right_on="F1_bio_Fighter_Name", how="left",
    ).merge(
        bios.add_prefix("F2_bio_"),
        left_on="Fighter_2", right_on="F2_bio_Fighter_Name", how="left",
    )
    return merged


def load_joined(
    fights_path: Path = FIGHTS_CSV, fighters_path: Path = FIGHTERS_CSV
) -> pd.DataFrame:
    """Convenience: load both CSVs and return the joined fight+bios table."""
    fights = load_fights(fights_path)
    fighters = load_fighters(fighters_path)
    return join_fighter_bios(fights, fighters)
