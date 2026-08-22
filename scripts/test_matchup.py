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

# Pass the real scheduled_rounds, because that is what the training path does
# (build_training_matrix forwards it). Calling without it exercises the SERVE
# path's default instead, which is a different question -- measured separately
# below.
rebuilt = [
    build_matchup_row(row.fighter_A_url, row.fighter_B_url,
                      row.Weight_Class, row.Event_Date, bios,
                      rounds=row.scheduled_rounds)
    for row in features.itertuples(index=False)
]

COMPARE = ("reach_diff", "height_diff", "age_diff",
           "reach_diff_missing", "height_diff_missing", "age_diff_missing",
           # Bout context must reproduce too, or the opt-in group is unsafe.
           "is_title_bout", "is_womens_bout", "is_nonstandard_weight",
           "scheduled_rounds",
           # Weight features must reproduce too, and cut_burden_diff especially:
           # it is division-dependent, so a serving path that derived it
           # differently would be wrong exactly on the cross-division matchups
           # it exists to handle.
           "weight_diff", "weight_diff_missing",
           "cut_burden_diff", "cut_burden_diff_missing")

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

# The serve-time default is a GUESS, because a future bout's Time_Format does
# not exist yet. Recording its accuracy here rather than leaving it implicit:
# the misses are overwhelmingly non-title five-rounders, i.e. main events, which
# the data carries no bout-order field to identify.
served = np.array([
    build_matchup_row(row.fighter_A_url, row.fighter_B_url,
                      row.Weight_Class, row.Event_Date, bios)["scheduled_rounds"]
    for row in features.itertuples(index=False)
])
agree = float(np.mean(served == features["scheduled_rounds"].to_numpy(dtype=float)))
check("serve-time rounds default is right at least 90% of the time",
      agree >= 0.90,
      f"({agree:.1%}; a caller who knows the format should pass rounds=)")

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

print()
print("=" * 78)
print("CROSS-DIVISION BEHAVIOUR — the reason the weight features exist")
print("=" * 78)
# predict_card "Islam Makhachev" "Jon Jones" returned ~50% because the model had
# no weight feature at all: it saw Jones 6 inches taller with 14 inches more
# reach and was blind to the 78 lb gap. These assertions pin the fix.
jones, _n = bios.resolve_name("Jon Jones")
islam, _n = bios.resolve_name("Islam Makhachev")
if jones and islam:
    when = pd.Timestamp("2026-08-23")
    rows = {d: build_matchup_row(jones, islam, d, when, bios)
            for d in ("Welterweight Bout", "Heavyweight Bout")}
    ww, hw = rows["Welterweight Bout"], rows["Heavyweight Bout"]

    check("weight_diff does NOT depend on the division",
          np.isclose(ww["weight_diff"], hw["weight_diff"]),
          f"({ww['weight_diff']:+.0f} both ways -- Jones is heavier regardless)")
    check("cut_burden_diff DOES depend on the division",
          ww["cut_burden_diff"] > 0 > hw["cut_burden_diff"],
          f"(welterweight {ww['cut_burden_diff']:+.0f}, "
          f"heavyweight {hw['cut_burden_diff']:+.0f})")
    check("the heavier fighter carries the burden at the lower weight",
          ww["cut_burden_diff"] > 50,
          "(Jones is ~78 lb over a 170 lb limit)")
    check("and the lighter fighter carries it at the higher weight",
          hw["cut_burden_diff"] < -50,
          "(Makhachev is ~95 lb under a 265 lb limit)")

    # Model-level: a forest that can see the weight columns must prefer
    # Makhachev more at welterweight than at heavyweight. Small and fast --
    # this asserts the DIRECTION of the response, not a particular probability.
    from src.forest import RandomForest
    from src.matchup import build_training_matrix, feature_columns
    cols = feature_columns(with_weight=True)
    Xw, yw, cols = build_training_matrix(features, bios, columns=cols)
    fw = RandomForest(n_trees=60, max_depth=8, min_samples_split=20,
                      min_samples_leaf=10, feature_subset="sqrt",
                      oob_score=False, random_state=42).fit(Xw, yw)

    def p_jones(div):
        f_ = build_matchup_row(jones, islam, div, when, bios)
        r_ = build_matchup_row(islam, jones, div, when, bios)
        pf = float(fw.predict_proba(rows_to_matrix([f_], cols))[0])
        pr = float(fw.predict_proba(rows_to_matrix([r_], cols))[0])
        return (pf + (1.0 - pr)) / 2.0

    p_ww, p_hw = p_jones("Welterweight Bout"), p_jones("Heavyweight Bout")
    check("the division actually changes the prediction",
          abs(p_ww - p_hw) > 0.01,
          f"(P(Jones) {p_ww:.3f} at WW vs {p_hw:.3f} at HW, moved "
          f"{abs(p_ww - p_hw):.3f}; before these features it could not move)")

    # NOT asserted: the DIRECTION of that move.
    #
    # The intuition says a weight-drained heavyweight at 170 should be
    # penalised, and in aggregate the data agrees weakly -- side A wins 47.6%
    # when it is >30 lb further from the limit against 52.8% when its opponent
    # is. But only 198 fights have |cut_burden_diff| > 30 at all, 121 of them
    # heavyweight bouts where the number means "a natural light-heavy against a
    # true heavyweight" rather than a weight cut. At the +/-78 of this matchup
    # the model is extrapolating into a region containing essentially no
    # training rows, because the UFC does not book a lightweight champion
    # against a heavyweight.
    #
    # So the model's direction there is not a learned fact and asserting it
    # would be asserting a wish. What IS testable is that the feature is wired,
    # responds to the division, and reaches the model -- which the checks above
    # do assert. The honest conclusion belongs in the README: the model can now
    # SEE weight, and at extreme cross-division gaps it is extrapolating and
    # should say so rather than answer confidently.
    print(f"       note: direction not asserted -- at |cut_burden_diff| ~ 78 "
          f"the model extrapolates")
else:
    print("[SKIP] Jones/Makhachev not resolvable in this dataset")

print()
print("=" * 78)
print("COLUMN ORDER")
print("=" * 78)
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
