"""Checks for src/boosting.py.

Two halves, same pattern as the other suites.

The booster is checked differentially against scikit-learn's
GradientBoostingClassifier, the way test_evaluate.py checks the metrics: an
independent implementation is the only thing that catches an update rule that is
self-consistently wrong.

The monotonic constraints are checked as a PROPERTY, and on data deliberately
built to punish an unconstrained fit -- a feature whose association reverses in
a sparse tail, which is exactly the pathology cut_burden_diff shows at extreme
weight gaps. A constraint that only holds where the data already agrees would be
decoration.

Run from the repo root (venv active):
    python scripts/test_boosting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

from src.boosting import GradientBoosting, _sigmoid  # noqa: E402
from src.evaluate import accuracy, log_loss  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


rng = np.random.default_rng(0)
n = 1500
X = rng.normal(size=(n, 4))
z = 1.2 * X[:, 0] - 0.6 * X[:, 1]
y = (rng.random(n) < _sigmoid(z)).astype(int)

print("=" * 78)
print("SIGMOID")
print("=" * 78)
check("sigmoid(0) is 0.5", np.isclose(_sigmoid(0.0), 0.5))
check("sigmoid is stable at extreme inputs",
      np.all(np.isfinite(_sigmoid(np.array([-800.0, 800.0])))),
      "(naive 1/(1+exp(-z)) overflows here, and a boosted model reaches it)")
check("sigmoid is monotone increasing",
      bool(np.all(np.diff(_sigmoid(np.linspace(-8, 8, 50))) > 0)))

print()
print("=" * 78)
print("BOOSTING LEARNS")
print("=" * 78)
model = GradientBoosting(n_rounds=150, learning_rate=0.08, max_depth=3).fit(X, y)
staged = model.staged_log_loss(X, y)

check("training log loss falls", staged[-1] < staged[0],
      f"({staged[0]:.4f} -> {staged[-1]:.4f})")
check("training log loss is monotone non-increasing",
      bool(np.all(np.diff(staged) < 1e-9)),
      "(each round fits the current residual, so train loss cannot rise)")
check("staged_log_loss has one entry per round", len(staged) == 150)
check("beats the base rate", accuracy(y, model.predict_proba(X)) > 0.65,
      f"({accuracy(y, model.predict_proba(X)):.4f})")
check("probabilities are valid",
      bool(np.all((model.predict_proba(X) > 0) & (model.predict_proba(X) < 1))))
check("round 0 with no trees is the base rate",
      np.isclose(_sigmoid(GradientBoosting(n_rounds=0).fit(X, y).f0_),
                 y.mean(), atol=1e-6),
      "(F0 is the log-odds of the base rate, so it must invert to it)")

# A single round with lr=1 and depth 0 must not move the prediction anywhere
# a tree cannot reach -- a smoke check that shrinkage is applied at all.
slow = GradientBoosting(n_rounds=30, learning_rate=0.01, max_depth=3).fit(X, y)
fast = GradientBoosting(n_rounds=30, learning_rate=0.30, max_depth=3).fit(X, y)
check("a larger learning rate moves further in the same rounds",
      slow.staged_log_loss(X, y)[-1] > fast.staged_log_loss(X, y)[-1],
      "(shrinkage is being applied)")

print()
print("=" * 78)
print("DIFFERENTIAL vs scikit-learn")
print("=" * 78)
sk = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                max_depth=3, random_state=0).fit(X, y)
mine = model.predict_proba(X)
theirs = sk.predict_proba(X)[:, 1]
check("accuracy is within 5 points of sklearn",
      abs(accuracy(y, mine) - accuracy(y, theirs)) < 0.05,
      f"(mine {accuracy(y, mine):.4f} vs sklearn {accuracy(y, theirs):.4f})")
check("log loss is within 0.05 of sklearn",
      abs(log_loss(y, mine) - log_loss(y, theirs)) < 0.05,
      f"(mine {log_loss(y, mine):.4f} vs sklearn {log_loss(y, theirs):.4f})")
check("predictions correlate strongly with sklearn's",
      float(np.corrcoef(mine, theirs)[0, 1]) > 0.95,
      f"(r = {float(np.corrcoef(mine, theirs)[0, 1]):.4f})")

print()
print("=" * 78)
print("MONOTONIC CONSTRAINTS — the point of the exercise")
print("=" * 78)
# Feature 2 is built to MISLEAD: its true effect is mildly negative, but a
# sparse tail (x > 2) is strongly positive. An unconstrained model will happily
# learn the tail and predict upward there. This is the same shape as
# cut_burden_diff, where 4 fights out of 8,658 sit past 78 lb and the model was
# measured extrapolating in the wrong direction.
m_rng = np.random.default_rng(1)
n2 = 2000
X2 = m_rng.normal(size=(n2, 3))
tail = X2[:, 2] > 2.0
z2 = 1.0 * X2[:, 0] + np.where(tail, 1.5, -0.1 * X2[:, 2])
y2 = (m_rng.random(n2) < _sigmoid(z2)).astype(int)

grid = np.zeros((40, 3))
grid[:, 2] = np.linspace(-3.0, 6.0, 40)

free = GradientBoosting(n_rounds=120, learning_rate=0.08, max_depth=3).fit(X2, y2)
tied = GradientBoosting(n_rounds=120, learning_rate=0.08, max_depth=3,
                        monotone_constraints={2: -1}).fit(X2, y2)
p_free, p_tied = free.predict_proba(grid), tied.predict_proba(grid)

check("the unconstrained model DOES violate the intended direction",
      int((np.diff(p_free) > 1e-12).sum()) > 0,
      f"({int((np.diff(p_free) > 1e-12).sum())} of 39 steps rise; "
      "if this ever passes trivially the fixture stopped being adversarial)")
check("the constrained model never rises",
      int((np.diff(p_tied) > 1e-12).sum()) == 0,
      f"(0 of 39 steps rise; free model rose at "
      f"{int((np.diff(p_free) > 1e-12).sum())})")
check("and it holds in the sparse tail, where there is almost no data",
      p_tied[-1] <= p_tied[len(p_tied) // 2] + 1e-12,
      f"(p at the extreme {p_tied[-1]:.3f} <= mid {p_tied[len(p_tied)//2]:.3f}; "
      "this is the guarantee that survives extrapolation)")
check("+1 constrains the other way",
      int((np.diff(GradientBoosting(
          n_rounds=60, learning_rate=0.08, max_depth=3,
          monotone_constraints={2: +1}).fit(X2, y2).predict_proba(grid))
          < -1e-12).sum()) == 0)
check("constraining costs some training fit, as it must",
      tied.staged_log_loss(X2, y2)[-1] >= free.staged_log_loss(X2, y2)[-1] - 1e-9,
      "(a constrained model cannot fit better than a free one on the same data; "
      "the constraint buys correctness outside the data, not fit inside it)")

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("All boosting checks pass.")
