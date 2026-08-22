"""Train/serve parity for src/matchup.py.

The claim matchup.py makes is that training and prediction build a feature row
the same way. This is the test that makes that claim falsifiable: take real
historical fights out of the training table, rebuild each one through
build_matchup_row, and require the numbers to match.

If they diverge, the served probabilities are wrong in a way no accuracy metric
would ever reveal -- so this failing is more serious than a metric regression.

Run from the repo root (venv active):
    python scripts/test_matchup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_NAMES, build_feature_table, to_matrix  # noqa: E402
from src.matchup import FighterBios, build_matchup_row, rows_to_matrix  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


features, _ = build_feature_table(seed=42)
bios = FighterBios()

print("=" * 78)
print(f"REBUILDING {len(features)} HISTORICAL FIGHTS THROUGH build_matchup_row")
print("=" * 78)

rebuilt = [
    build_matchup_row(row.fighter_A_url, row.fighter_B_url,
                      row.Weight_Class, row.Event_Date, bios)
    for row in features.itertuples(index=False)
]

COMPARE = ("reach_diff", "height_diff", "age_diff",
           "reach_diff_missing", "height_diff_missing", "age_diff_missing")

worst = {}
for col in COMPARE:
    theirs = features[col].to_numpy()
    mine = np.array([r[col] for r in rebuilt])
    if theirs.dtype == bool or mine.dtype == bool:
        ok = bool(np.array_equal(theirs.astype(bool), mine.astype(bool)))
        detail = ""
    else:
        diff = np.abs(theirs.astype(float) - mine.astype(float))
        ok = bool(np.all(diff < 1e-9))
        worst[col] = float(diff.max())
        detail = f"(max |delta| {diff.max():.2e})"
    check(f"{col} reproduced exactly", ok, detail)

check("stance_matchup reproduced exactly",
      bool(np.array_equal(features["stance_matchup"].to_numpy(),
                          np.array([r["stance_matchup"] for r in rebuilt]))))

# The whole encoded matrix, which is what the model actually consumes. Column
# ORDER is part of the contract, so this compares the matrices rather than the
# columns one at a time.
X_train, _ = to_matrix(features)
X_serve = rows_to_matrix(rebuilt, FEATURE_NAMES)
check("encoded matrices are identical",
      X_train.shape == X_serve.shape and bool(np.allclose(X_train, X_serve, atol=1e-9)),
      f"({X_train.shape} vs {X_serve.shape})")

print()
print("=" * 78)
print("SERVING-PATH EDGE CASES")
print("=" * 78)

# Pick real fighters off the feature table. Crucially, pick a pair whose bios
# are POPULATED: features.iloc[0] is a 1994 bout with no DOB on either side, so
# its age_diff is the imputed 0.0 and every check below would compare 0.0 with
# 0.0 -- passing because the value is absent, not because the property holds.
populated = features[~features["age_diff_missing"]
                     & ~features["reach_diff_missing"]].iloc[0]
url_x = str(populated["fighter_A_url"])
url_y = str(populated["fighter_B_url"])
div = populated["Weight_Class"]

# A future bout: no history needed for static features, so this must just work.
future = build_matchup_row(url_x, url_y, div, pd.Timestamp("2030-01-01"), bios)
check("a future-dated bout builds a real age_diff",
      np.isfinite(future["age_diff"]) and not future["age_diff_missing"],
      f"(age_diff {future['age_diff']:+.4f})")
# age_diff is a DIFFERENCE of two ages, so both sides advance together and the
# value is invariant to the bout date (up to leap-year rounding in the /365.25).
far = build_matchup_row(url_x, url_y, div, pd.Timestamp("2030-01-01"), bios)["age_diff"]
near = build_matchup_row(url_x, url_y, div, pd.Timestamp("2020-01-01"), bios)["age_diff"]
check("age_diff is invariant to the bout date",
      np.isclose(far, near, atol=1e-9) and abs(far) > 1e-6,
      f"(2030 {far:.6f} vs 2020 {near:.6f}; non-zero, so not vacuous)")

# Missing bios must reproduce build_features' flag-and-impute policy exactly.
# Restored using a real pair that IS missing data -- 12.1% of fights are
# missing reach, so this is a common path, not a corner.
gap = features[features["reach_diff_missing"]].iloc[0]
gap_row = build_matchup_row(str(gap["fighter_A_url"]), str(gap["fighter_B_url"]),
                            gap["Weight_Class"], gap["Event_Date"], bios)
check("a missing bio sets the missing flag", gap_row["reach_diff_missing"])
check("missing diffs are imputed to 0.0, matching build_features",
      gap_row["reach_diff"] == 0.0 and gap["reach_diff"] == 0.0)

# Swapping the sides must negate every diff. This is the symmetry the label
# depends on, and predict_card relies on it to average both orderings.
fwd = build_matchup_row(url_x, url_y, div, pd.Timestamp("2026-01-01"), bios)
rev = build_matchup_row(url_y, url_x, div, pd.Timestamp("2026-01-01"), bios)
check("swapping sides negates the numeric diffs",
      all(np.isclose(fwd[c], -rev[c]) for c in ("height_diff", "age_diff")
          if not fwd[f"{c}_missing"]))
check("swapping sides preserves stance_matchup",
      fwd["stance_matchup"] == rev["stance_matchup"],
      "(it is symmetric by construction)")
check("no NaN reaches the model matrix",
      bool(np.all(np.isfinite(rows_to_matrix([fwd], FEATURE_NAMES)))))

# Column order is a contract, not a convention.
try:
    rows_to_matrix(rebuilt, list(FEATURE_NAMES) + ["not_a_feature"])
    check("unknown feature name raises", False, "(no error)")
except KeyError:
    check("unknown feature name raises", True)

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    print("Train/serve skew: the model would be served rows it was not fitted on.")
    sys.exit(1)
print(f"Train/serve parity holds across all {len(features)} fights.")
