"""Checks for src/evaluate.py.

Two halves.

The metric functions are checked differentially against scikit-learn, the same
way test_tree.py checks the fast split sweep against a naive reference: an
independent implementation is the only thing that catches a formula that is
self-consistently wrong.

The bootstrap and the fold generator carry the invariants that actually matter
-- pairing, ordering, and date boundaries -- as executable checks. (They still
degrade to SKIP if either function is left unimplemented, so the suite runs
either way.)

Run from the repo root (venv active):
    python scripts/test_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    log_loss as sk_log_loss,
)

from src.evaluate import (  # noqa: E402
    LOG_LOSS_EPS,
    accuracy,
    bootstrap_ci,
    brier_score,
    log_loss,
    reference_scores,
    reliability_table,
    walk_forward_folds,
)

failures = []
skipped = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def skip(name, why):
    print(f"[SKIP] {name}  ({why})")
    skipped.append(name)


rng = np.random.default_rng(42)
n = 2000
y = rng.integers(0, 2, size=n)
# Probabilities correlated with the label, kept away from 0/1 so sklearn's own
# clipping cannot be what makes the two agree.
p = np.clip(0.5 + 0.25 * (y - 0.5) * 2 + rng.normal(0, 0.15, n), 0.02, 0.98)

print("=" * 78)
print("METRICS vs scikit-learn")
print("=" * 78)

check("log_loss matches sklearn",
      np.isclose(log_loss(y, p), sk_log_loss(y, p), atol=1e-12),
      f"({log_loss(y, p):.10f} vs {sk_log_loss(y, p):.10f})")
check("brier matches sklearn",
      np.isclose(brier_score(y, p), brier_score_loss(y, p), atol=1e-12),
      f"({brier_score(y, p):.10f} vs {brier_score_loss(y, p):.10f})")
check("accuracy matches sklearn",
      np.isclose(accuracy(y, p), accuracy_score(y, (p > 0.5).astype(int))))

# The clip is load-bearing: a pure leaf really does emit 0.0 and 1.0, and one
# confidently-wrong row must stay finite rather than poisoning the whole mean.
y_pure = np.array([0, 1, 1, 0])
p_pure = np.array([1.0, 0.0, 1.0, 0.0])   # two confident and wrong
ll_pure = log_loss(y_pure, p_pure)
check("log_loss finite on confidently-wrong pure leaves",
      np.isfinite(ll_pure), f"({ll_pure:.4f})")
# The clip, not the model, sets the magnitude here: two of four rows are wrong
# at the clip, so the mean is -log(eps)/2. Loose tolerance on purpose --
# 1 - (1 - eps) is not exactly eps in floating point, and asserting that
# detail would be testing numpy's rounding rather than this formula.
check("log_loss magnitude is set by the clip",
      np.isclose(ll_pure, -np.log(LOG_LOSS_EPS) / 2.0, rtol=1e-3),
      f"({ll_pure:.4f} vs -log(eps)/2 = {-np.log(LOG_LOSS_EPS) / 2.0:.4f})")
check("brier needs no clip", np.isclose(brier_score(y_pure, p_pure), 0.5))

# The honest zero point. Anything that cannot beat these has learned nothing.
ref = reference_scores(np.array([0, 1] * 500))
check("reference log_loss is log(2)", np.isclose(ref["log_loss"], np.log(2)),
      f"({ref['log_loss']:.6f})")
check("reference brier is 0.25", np.isclose(ref["brier"], 0.25))

# Accuracy must use the STRICT threshold, matching _majority_class's tie rule.
check("accuracy threshold is strict at exactly 0.5",
      accuracy(np.array([1]), np.array([0.5])) == 0.0)

rel = reliability_table(y, p, n_bins=10)
check("reliability_table bins sum to n", sum(r["n"] for r in rel) == n)
check("reliability_table keeps p == 1.0",
      any(r["bin_hi"] == 1.0 for r in reliability_table(
          np.array([1, 0]), np.array([1.0, 1.0]))))

print()
print("=" * 78)
print("BOOTSTRAP")
print("=" * 78)
try:
    point, lo, hi = bootstrap_ci(y, p, brier_score, n_boot=500,
                                 rng=np.random.default_rng(0))
    check("point estimate is the metric on the ORIGINAL data",
          np.isclose(point, brier_score(y, p)),
          "(not the mean of the resamples)")
    check("interval brackets the point estimate", lo <= point <= hi,
          f"({lo:.4f} <= {point:.4f} <= {hi:.4f})")
    check("interval is non-degenerate", hi > lo)
    # Same seed must give the same interval, or nothing downstream reproduces.
    again = bootstrap_ci(y, p, brier_score, n_boot=500,
                         rng=np.random.default_rng(0))
    check("same rng reproduces the interval", np.allclose(again, (point, lo, hi)))
    # More data must narrow it. This is the check that catches resampling
    # fewer than n indices, which inflates the interval regardless of n.
    small = bootstrap_ci(y[:200], p[:200], brier_score, n_boot=500,
                         rng=np.random.default_rng(1))
    big = bootstrap_ci(y, p, brier_score, n_boot=500,
                       rng=np.random.default_rng(1))
    check("interval narrows as n grows",
          (big[2] - big[1]) < (small[2] - small[1]),
          f"(n=2000 width {big[2]-big[1]:.4f} < n=200 width {small[2]-small[1]:.4f})")
    # Pairing: if y and p were resampled independently the metric would drift
    # toward its value on shuffled data, which for Brier is much worse.
    shuffled = brier_score(y, rng.permutation(p))
    check("interval excludes the shuffled-pairing value",
          not (lo <= shuffled <= hi),
          f"(shuffled brier {shuffled:.4f} outside [{lo:.4f}, {hi:.4f}])")
except NotImplementedError:
    skip("bootstrap_ci checks", "not implemented yet")

print()
print("=" * 78)
print("WALK-FORWARD FOLDS")
print("=" * 78)
# 400 rows over 40 event dates, 10 fights per card -- so a row-index cut would
# land mid-card and a date-boundary cut cannot.
dates = np.repeat(np.arange("2015-01", "2018-05", dtype="datetime64[M]")[:40], 10)
try:
    folds = list(walk_forward_folds(dates, n_folds=5, min_train_frac=0.5))
    check("yields the requested number of folds", len(folds) == 5,
          f"(got {len(folds)})")
    ok_order, ok_disjoint, ok_nonempty, ok_dates = True, True, True, True
    for tr, te in folds:
        if len(tr) == 0 or len(te) == 0:
            ok_nonempty = False
            continue
        if dates[te].min() <= dates[tr].max():
            ok_order = False
        if set(np.asarray(tr).tolist()) & set(np.asarray(te).tolist()):
            ok_disjoint = False
        # No single event date may straddle the cut.
        if set(np.unique(dates[tr]).tolist()) & set(np.unique(dates[te]).tolist()):
            ok_dates = False
    check("every fold has non-empty train and test", ok_nonempty)
    check("test always strictly after train", ok_order)
    check("train and test indices are disjoint", ok_disjoint)
    check("no event date appears on both sides of a cut", ok_dates)
    check("the last fold reaches the final date",
          dates[folds[-1][1]].max() == dates.max(),
          "(otherwise the most recent fights are never tested)")
    check("first fold trains on >= min_train_frac of the rows",
          len(folds[0][0]) >= 0.5 * len(dates),
          f"({len(folds[0][0])} of {len(dates)})")
except NotImplementedError:
    skip("walk_forward_folds checks", "not implemented yet")

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
if skipped:
    print(f"All implemented checks pass. {len(skipped)} group(s) skipped: "
          f"{', '.join(skipped)}")
    print("A skipped group means its function is unimplemented in src/evaluate.py.")
else:
    print("All evaluation checks pass.")
