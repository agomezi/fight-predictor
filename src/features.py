"""Feature pipeline for the UFC fight predictor.

This turns the raw fight + fighter tables into a modelling table of SYMMETRIC
matchup features. Four things have to be right before any of it is usable:

  1. Resolve the fighter join to exactly one profile per fighter per fight
     (no silent many-to-one fan-out from duplicate names).
  2. Clean the label to binary (drop draws / no-contests).
  3. Build symmetric static features: reach_diff, height_diff, age_diff,
     stance_matchup. NO performance-average columns yet — those need the
     rolling as-of-fight rebuild, which is a later step.
  4. Randomise which fighter is "A" vs "B" so the ~63% Fighter_1 position
     bias is destroyed; label and every diff are computed relative to A.

The unresolved-safe join key is Fighter_URL (a real unique id). The fights
table doesn't carry it, so for names that collide across profiles we pick the
profile whose weight best matches the bout's weight class.

The resulting table is deliberately not just the model matrix: alongside the
features it carries METADATA_NAMES -- the fight id, both resolved fighter ids,
and the Method/End_Round outcome detail. The model never sees any of it
(to_matrix selects FEATURE_NAMES), but keeping the ids means downstream work can
key a fighter by a unique id rather than a name string, and can join back to the
raw per-fight stats that rolling as-of-fight aggregates are built from.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data_loading import load_fights, load_fighters

RANDOM_SEED = 42

# Canonical division weights (lbs). Used only to disambiguate duplicate names;
# women's divisions share the same limits as their keyword.
WEIGHT_CLASS_LBS = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}
# Longest keys first so "Light Heavyweight" matches before "Heavyweight".
_CLASS_KEYS = sorted(WEIGHT_CLASS_LBS, key=len, reverse=True)


def estimate_class_lbs(weight_class: object) -> float:
    """Map a bout's weight-class string to an approximate limit in lbs.

    Returns NaN for catch-weight / open-weight / anything non-standard.
    """
    if not isinstance(weight_class, str):
        return np.nan
    for key in _CLASS_KEYS:
        if key.lower() in weight_class.lower():
            return float(WEIGHT_CLASS_LBS[key])
    return np.nan


# ---------------------------------------------------------------------------
# 1. Resolve the join to one Fighter_URL per fighter per fight
# ---------------------------------------------------------------------------
def _build_name_resolver(fighters: pd.DataFrame):
    """Return a function (name, class_lbs) -> Fighter_URL (or None).

    Unique names resolve immediately. Colliding names resolve to the profile
    whose weight is nearest the bout's class weight; if that can't be decided
    (unknown class weight, or profiles lack weight), returns None.
    """
    by_name: dict[str, pd.DataFrame] = {
        name: grp for name, grp in fighters.groupby("Fighter_Name")
    }

    def resolve(name: str, class_lbs: float) -> Optional[str]:
        grp = by_name.get(name)
        if grp is None:
            return None
        if len(grp) == 1:
            return grp.iloc[0]["Fighter_URL"]
        # Duplicate name: pick nearest weight to the bout's class.
        if np.isnan(class_lbs):
            return None
        weights = grp["Weight_lbs"]
        if weights.isna().all():
            return None
        idx = (weights - class_lbs).abs().idxmin()
        return grp.loc[idx, "Fighter_URL"]

    return resolve


def resolve_join(fights: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Attach a resolved Fighter_URL for each side, then merge bios one-to-one.

    Any fight whose fighter can't be resolved to a single profile is dropped
    (reported by the caller), so the output has exactly one row per surviving
    fight — no many-to-one inflation.
    """
    resolve = _build_name_resolver(fighters)
    fights = fights.copy()
    class_lbs = fights["Weight_Class"].apply(estimate_class_lbs)

    fights["F1_url"] = [
        resolve(n, w) for n, w in zip(fights["Fighter_1"], class_lbs)
    ]
    fights["F2_url"] = [
        resolve(n, w) for n, w in zip(fights["Fighter_2"], class_lbs)
    ]

    bio_cols = [
        "Fighter_URL", "Height_in", "Reach_in", "Weight_lbs", "Stance", "DOB",
    ]
    bios = fighters[bio_cols].copy()

    merged = fights.merge(
        bios.add_prefix("F1_bio_"),
        left_on="F1_url", right_on="F1_bio_Fighter_URL", how="left",
    ).merge(
        bios.add_prefix("F2_bio_"),
        left_on="F2_url", right_on="F2_bio_Fighter_URL", how="left",
    )
    return merged


# ---------------------------------------------------------------------------
# 2. Clean the label to binary
# ---------------------------------------------------------------------------
def clean_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fights the Winner equals Fighter_1 or Fighter_2 (drop draws/NC)."""
    winner = df["Winner"].astype("string")
    decisive = (winner == df["Fighter_1"]) | (winner == df["Fighter_2"])
    return df[decisive].copy()


# ---------------------------------------------------------------------------
# 3 + 4. Symmetric features under a randomised A/B assignment
# ---------------------------------------------------------------------------
def _stance_matchup(a: object, b: object) -> str:
    if not isinstance(a, str) or not a.strip() or not isinstance(b, str) or not b.strip():
        return "Unknown"
    return "Same" if a == b else "Opposite"


def _age_years(dob: pd.Series, event_date: pd.Series) -> pd.Series:
    return (event_date - dob).dt.days / 365.25


def build_features(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the symmetric matchup table under a random A/B swap per fight.

    For each fight a coin flip decides whether A = Fighter_1 (a_is_f1=True) or
    A = Fighter_2. The label is 'did A win', and every diff is A minus B, so no
    feature encodes the original Fighter_1/Fighter_2 ordering.

    The table also carries METADATA_NAMES: the fight and fighter ids, and the
    Method/End_Round outcome detail. None of it reaches the model -- to_matrix
    selects FEATURE_NAMES explicitly -- but it is what lets later work join back
    to the raw per-fight stats and identify a fighter by a unique id instead of
    a name string. fighter_A_url / fighter_B_url follow the same A/B swap as the
    features, so they always describe the same side as reach_diff does.
    """
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    a_is_f1 = rng.random(len(df)) < 0.5

    def pick(f1_series: pd.Series, f2_series: pd.Series) -> pd.Series:
        return f1_series.where(a_is_f1, f2_series)

    # Raw A/B side values, chosen per the coin flip.
    a_reach = pick(df["F1_bio_Reach_in"], df["F2_bio_Reach_in"])
    b_reach = pick(df["F2_bio_Reach_in"], df["F1_bio_Reach_in"])
    a_height = pick(df["F1_bio_Height_in"], df["F2_bio_Height_in"])
    b_height = pick(df["F2_bio_Height_in"], df["F1_bio_Height_in"])
    a_dob = pick(df["F1_bio_DOB"], df["F2_bio_DOB"])
    b_dob = pick(df["F2_bio_DOB"], df["F1_bio_DOB"])
    a_stance = pick(df["F1_bio_Stance"], df["F2_bio_Stance"])
    b_stance = pick(df["F2_bio_Stance"], df["F1_bio_Stance"])
    a_name = pick(df["Fighter_1"], df["Fighter_2"])

    a_age = _age_years(a_dob, df["Event_Date"])
    b_age = _age_years(b_dob, df["Event_Date"])

    # Label: did A win?
    label = (df["Winner"].astype("string") == a_name).astype(int)

    out = pd.DataFrame({
        # Identity and event metadata. Never model inputs -- carried so that
        # downstream code can join back to the raw per-fight stats and can key
        # a fighter by a unique id rather than by a name string.
        "Fight_URL": df["Fight_URL"],
        "Event_Date": df["Event_Date"],
        "Weight_Class": df["Weight_Class"],
        "fighter_A": a_name,
        "fighter_B": pick(df["Fighter_2"], df["Fighter_1"]),
        "fighter_A_url": pick(df["F1_url"], df["F2_url"]),
        "fighter_B_url": pick(df["F2_url"], df["F1_url"]),
        "a_is_f1": a_is_f1,
        # Outcome detail for the method/round head. Also never model inputs --
        # these describe HOW the fight ended, so feeding them to the winner
        # model would be target leakage.
        "Method": df["Method"],
        "End_Round": df["End_Round"],
        # Symmetric matchup features -- the model-facing set is FEATURE_NAMES.
        "reach_diff": a_reach - b_reach,
        "height_diff": a_height - b_height,
        "age_diff": a_age - b_age,
        "stance_matchup": [_stance_matchup(a, b) for a, b in zip(a_stance, b_stance)],
        "label": label,
    })

    # Missing-value policy: never drop a row. Impute each numeric diff to 0 (a
    # neutral "no known advantage") and keep a companion boolean recording that
    # the value was unknown, so the split code never has to handle NaN itself.
    # stance_matchup already carries its own "Unknown" category — left as-is.
    for col in ("reach_diff", "height_diff", "age_diff"):
        out[f"{col}_missing"] = out[col].isna()
        out[col] = out[col].fillna(0.0)
    return out


def chronological_split(features: pd.DataFrame, test_frac: float = 0.18):
    """Split by time: earliest fights train, most recent ~test_frac test.

    Sorting by Event_Date and holding out the tail (never a random split) is
    the whole point — career-average features would otherwise leak future
    information backwards. Returns (train, test), each Event_Date-sorted.

    The cut lands on a DATE boundary, not a row index. Cutting by row count
    can slice a single event's card in half, putting some of one night's
    fights in train and the rest in test — exactly the kind of contamination
    a chronological split exists to prevent. The row index only locates the
    target date; the whole date group at that boundary then moves into test,
    so no event date appears on both sides. Because the boundary moves
    earlier rather than later, the realised test fraction is >= test_frac.

    Raises:
        ValueError: if the requested split leaves either side empty — which
        happens when test_frac is degenerate, or when too few distinct event
        dates exist to place a boundary at all.
    """
    ordered = features.sort_values("Event_Date", kind="mergesort").reset_index(drop=True)
    target = int(round(len(ordered) * (1 - test_frac)))
    target = min(max(target, 1), len(ordered) - 1)

    dates = ordered["Event_Date"]
    boundary = dates.iloc[target]
    is_test = dates >= boundary

    train = ordered[~is_test].copy()
    test = ordered[is_test].copy()

    # Snapping to a date boundary can swallow one side entirely (e.g. every
    # row shares a single event date). Fail loudly rather than hand back an
    # empty frame that only breaks further downstream.
    if train.empty or test.empty:
        raise ValueError(
            f"chronological_split(test_frac={test_frac}) produced "
            f"train={len(train)}, test={len(test)}; not enough distinct "
            f"event dates ({dates.nunique()}) to place a boundary"
        )
    return train, test


# ---------------------------------------------------------------------------
# 5. Model matrix — everything the tree sees, as plain numbers
# ---------------------------------------------------------------------------
# The feature TABLE's columns, split into two groups. Note this is a different
# space from FEATURE_NAMES below, which names the columns of the encoded MATRIX
# that to_matrix produces: stance_matchup is one nominal column here and expands
# into stance_same / stance_unknown there. Conflating the two is an easy mistake
# -- indexing the table by FEATURE_NAMES raises KeyError.
#
# Kept explicit so build_features.py can assert the table's shape rather than
# infer it: a column in neither list is an accident.
METADATA_NAMES = [
    "Fight_URL",
    "Event_Date",
    "Weight_Class",
    "fighter_A",
    "fighter_B",
    "fighter_A_url",
    "fighter_B_url",
    "a_is_f1",
    "Method",
    "End_Round",
]

FEATURE_TABLE_COLS = [
    "reach_diff",
    "height_diff",
    "age_diff",
    "reach_diff_missing",
    "height_diff_missing",
    "age_diff_missing",
    "stance_matchup",
]

FEATURE_NAMES = [
    "reach_diff",
    "height_diff",
    "age_diff",
    "reach_diff_missing",
    "height_diff_missing",
    "age_diff_missing",
    "stance_same",
    "stance_unknown",
]


def to_matrix(features: pd.DataFrame):
    """Turn the feature table into (X, y) float/int numpy arrays.

    The tree only compares numbers with <=, so the one categorical column has
    to be encoded. stance_matchup is nominal with three levels, so it becomes
    two indicator columns (Same, Unknown); Opposite is the implied reference
    when both are 0. Booleans become 0.0/1.0, where a `<= 0.5` split is just
    "is this flag set".
    """
    out = pd.DataFrame(index=features.index)
    for col in ("reach_diff", "height_diff", "age_diff"):
        out[col] = features[col].astype(float)
    for col in ("reach_diff_missing", "height_diff_missing", "age_diff_missing"):
        out[col] = features[col].astype(float)
    # Opposite is the implied reference (both indicators 0), so any value that
    # isn't one of the three known levels — including NaN — would silently be
    # encoded as Opposite. Check rather than guess.
    stance = features["stance_matchup"]
    unexpected = set(stance[~stance.isin(["Same", "Opposite", "Unknown"])])
    if unexpected:
        raise ValueError(f"unexpected stance_matchup levels: {sorted(map(str, unexpected))}")

    out["stance_same"] = (features["stance_matchup"] == "Same").astype(float)
    out["stance_unknown"] = (features["stance_matchup"] == "Unknown").astype(float)

    X = out[FEATURE_NAMES].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    return X, y


def build_feature_table(seed: int = RANDOM_SEED):
    """Full pipeline. Returns (features, stats) where stats reports drop counts."""
    fights = load_fights()
    fighters = load_fighters()

    n_fights = len(fights)
    merged = resolve_join(fights, fighters)
    unresolved = merged["F1_url"].isna() | merged["F2_url"].isna()
    merged = merged[~unresolved].copy()

    labelled = clean_label(merged)
    features = build_features(labelled, seed=seed)

    stats = {
        "n_fights_raw": n_fights,
        "n_joined_rows": len(merged),  # after resolution, before label clean
        "n_dropped_unresolved": int(unresolved.sum()),
        "n_dropped_nonbinary": len(merged) - len(labelled),
        "n_final": len(features),
    }
    return features, stats
