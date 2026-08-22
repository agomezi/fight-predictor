"""One function that builds a matchup row, used by BOTH training and prediction.

THE POINT. Training builds its matrix from historical fights; predict_card.py
builds a row for a bout that has not happened. If those two paths are separate
code, they drift -- a column in a different order, an imputation rule applied in
one and not the other, an age computed from a different reference date -- and the
served probabilities are quietly wrong while every test-set metric stays clean.
That failure mode is called training/serving skew, and it is nasty precisely
because nothing goes red.

The fix is structural rather than disciplinary: there is one function, and both
paths call it. scripts/test_matchup.py proves the training path agrees, by
rebuilding real historical fights through this module and comparing against the
rows features.py produced.

Note this module does NOT re-derive the static features. It imports the same
helpers features.py uses, because a second implementation of "age on fight
night" is the very skew being prevented.

Two modes:

    build_matchup_row(..., index=None)   static features only -- v1 parity,
                                         works today
    build_matchup_row(..., index=idx)    adds rolling + Elo diffs

The static mode existing on its own is deliberate. It means a working
predict_card.py does not have to wait for history.py, and "here is my model's
calibrated pick for Saturday" is available now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loading import load_fighters
from src.features import (
    FEATURE_NAMES,
    _age_years,
    _stance_matchup,
    estimate_class_lbs,
)
from src.history import AS_OF_FEATURES, features_as_of, prior_for

# Rolling features are compared as A-minus-B differences, exactly like the
# static ones, so the symmetry that makes the label meaningful is preserved.
# support is a flag about the row, not a difference, so it is carried per side.
ROLLING_DIFF_NAMES = tuple(f"{k}_diff" for k in AS_OF_FEATURES)


class FighterBios:
    """Fighter bio lookup keyed by Fighter_URL, with name resolution.

    Keyed by url rather than name on purpose: names collide, which is why
    PR #3 threaded the resolved urls through the feature table.
    """

    def __init__(self, fighters: pd.DataFrame = None):
        self.fighters = load_fighters() if fighters is None else fighters
        self._by_url = self.fighters.set_index("Fighter_URL")
        self._by_name: dict[str, pd.DataFrame] = {
            name: grp for name, grp in self.fighters.groupby("Fighter_Name")
        }

    def resolve_name(self, name: str, weight_class=None):
        """Name -> (fighter_url, note). Mirrors features.resolve_join's rule.

        Returns (None, reason) when the name is unknown or ambiguous, so a CLI
        can report which of the two it was instead of guessing.
        """
        grp = self._by_name.get(name)
        if grp is None:
            return None, f"no fighter named {name!r}"
        if len(grp) == 1:
            return str(grp.iloc[0]["Fighter_URL"]), "unique"
        class_lbs = estimate_class_lbs(weight_class)
        if np.isnan(class_lbs):
            return None, (f"{len(grp)} fighters named {name!r}; a division is "
                          "needed to tell them apart")
        weights = grp["Weight_lbs"]
        if weights.isna().all():
            return None, f"{len(grp)} fighters named {name!r}, none with a weight"
        idx = (weights - class_lbs).abs().idxmin()
        return str(grp.loc[idx, "Fighter_URL"]), "resolved by division weight"

    def candidates(self, name: str):
        grp = self._by_name.get(name)
        return [] if grp is None else list(grp["Fighter_URL"])

    def bio(self, fighter_url: str) -> dict:
        if fighter_url not in self._by_url.index:
            raise KeyError(f"unknown fighter url: {fighter_url}")
        row = self._by_url.loc[fighter_url]
        if isinstance(row, pd.DataFrame):        # duplicate url, should not happen
            row = row.iloc[0]
        return {
            "height_in": row["Height_in"],
            "reach_in": row["Reach_in"],
            "stance": row["Stance"],
            "dob": row["DOB"],
        }


def build_matchup_row(fighter_a_url: str, fighter_b_url: str, division,
                      as_of_date, bios: FighterBios,
                      index=None, priors: dict = None,
                      elo_ratings: dict = None) -> dict:
    """The feature row for A versus B, in `division`, as known on `as_of_date`.

    Args:
        fighter_a_url, fighter_b_url: resolved fighter ids.
        division: the bout's weight class string. Used for the division prior
            and for name disambiguation upstream.
        as_of_date: the bout date. Every rolling feature is computed strictly
            before it -- for a future bout, pass today.
        bios: FighterBios instance.
        index: HistoryIndex, or None for static features only.
        priors: from history.division_priors(). Required when index is given.
        elo_ratings: {fighter_url: rating} as of this date, or None.

    Returns:
        dict of feature name -> value. Always contains FEATURE_NAMES minus the
        two stance indicators (which to_matrix derives from stance_matchup),
        i.e. the same columns features.build_features emits. With `index` it
        also contains ROLLING_DIFF_NAMES and per-side support flags.

    The A/B asymmetry is the caller's business: this builds the row for the
    sides as given. predict_card averages both orderings so the answer does not
    depend on which fighter was typed first.
    """
    a, b = bios.bio(fighter_a_url), bios.bio(fighter_b_url)
    when = pd.Timestamp(as_of_date)

    # Ages via features.py's own helper, on one-element Series, so the day-count
    # convention cannot drift from the training table's.
    a_age = _age_years(pd.Series([a["dob"]]), pd.Series([when])).iloc[0]
    b_age = _age_years(pd.Series([b["dob"]]), pd.Series([when])).iloc[0]

    diffs = {
        "reach_diff": a["reach_in"] - b["reach_in"],
        "height_diff": a["height_in"] - b["height_in"],
        "age_diff": a_age - b_age,
    }
    row: dict = {
        "Event_Date": when,
        "Weight_Class": division,
        "fighter_A_url": fighter_a_url,
        "fighter_B_url": fighter_b_url,
        "stance_matchup": _stance_matchup(a["stance"], b["stance"]),
    }
    # Same missing-value policy as build_features: impute the diff to a neutral
    # 0.0 and record that it was unknown. Diverging here is exactly the skew
    # this module exists to prevent.
    for name, value in diffs.items():
        missing = bool(pd.isna(value))
        row[name] = 0.0 if missing else float(value)
        row[f"{name}_missing"] = missing

    if index is None:
        return row

    if priors is None:
        raise ValueError("priors are required when a HistoryIndex is supplied")
    a_feat = features_as_of(index, fighter_a_url, when, weight_class=division,
                            priors=priors, elo_ratings=elo_ratings)
    b_feat = features_as_of(index, fighter_b_url, when, weight_class=division,
                            priors=priors, elo_ratings=elo_ratings)
    for key in AS_OF_FEATURES:
        av, bv = a_feat.get(key, np.nan), b_feat.get(key, np.nan)
        value = np.nan if (pd.isna(av) or pd.isna(bv)) else float(av) - float(bv)
        missing = bool(pd.isna(value))
        row[f"{key}_diff"] = 0.0 if missing else value
        row[f"{key}_diff_missing"] = missing
    row["support_A"] = a_feat.get("support", "none")
    row["support_B"] = b_feat.get("support", "none")
    return row


def feature_columns(with_rolling: bool = False) -> list:
    """The model's column list, in the one order both paths must agree on."""
    cols = list(FEATURE_NAMES)
    if with_rolling:
        cols += list(ROLLING_DIFF_NAMES)
    return cols


def build_training_matrix(features: pd.DataFrame, bios: "FighterBios",
                          index=None, priors: dict = None,
                          elo_index=None, columns=None):
    """Build (X, y, columns) for historical fights through build_matchup_row.

    The reason this lives here rather than in each script: training-matrix
    construction was duplicated in predict_card.py and test_leakage.py, and two
    copies of "how a training row is assembled" is the same skew risk that
    build_matchup_row exists to remove -- one copy gaining a column, or a
    different column order, would not fail any test.

    Elo is the part that has to be per-row. `elo_index.for_fight(Fight_URL)`
    gives each corner its rating going into THAT fight; a single global snapshot
    would be correct for the most recent fight and wrong for every earlier one.
    Passing elo_index=None leaves the elo_diff column present but inert, which
    is what the model has been doing until now.

    Args:
        features: the feature table (needs Fight_URL, both fighter urls,
            Weight_Class, Event_Date, label).
        bios: FighterBios.
        index: HistoryIndex, or None for static features only.
        priors: from history.division_priors(); required when index is given.
        elo_index: history.EloIndex, or None to leave Elo inert.
        columns: explicit column order. Defaults to feature_columns(index is
            not None).

    Returns:
        (X, y, columns) -- X float, y int, columns the list actually used.
    """
    cols = list(columns) if columns is not None else feature_columns(index is not None)
    rows = []
    for r in features.itertuples(index=False):
        elo = elo_index.for_fight(r.Fight_URL) if elo_index is not None else None
        rows.append(build_matchup_row(
            r.fighter_A_url, r.fighter_B_url, r.Weight_Class, r.Event_Date,
            bios, index=index, priors=priors, elo_ratings=elo,
        ))
    X = rows_to_matrix(rows, cols)
    y = features["label"].to_numpy(dtype=int)
    return X, y, cols


def rows_to_matrix(rows, feature_names=None):
    """Stack matchup dicts into a matrix, in an explicit column order.

    The column order comes from a named list, never from dict iteration order.
    A model fitted on one order and served another produces confident nonsense,
    and it is the single most likely way this module gets misused.
    """
    names = list(FEATURE_NAMES if feature_names is None else feature_names)
    frame = pd.DataFrame(list(rows))
    # stance_matchup expands into the two indicators to_matrix produces.
    if "stance_same" in names:
        frame["stance_same"] = (frame["stance_matchup"] == "Same").astype(float)
    if "stance_unknown" in names:
        frame["stance_unknown"] = (frame["stance_matchup"] == "Unknown").astype(float)
    for n in names:
        if n not in frame.columns:
            raise KeyError(f"matchup rows are missing feature {n!r}")
    return frame[names].to_numpy(dtype=float)
