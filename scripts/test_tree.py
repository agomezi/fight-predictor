"""Tests for the split search and the tree recursion.

The centerpiece is a differential test: best_split() uses a fast sort-and-sweep
with cumulative counts, which is easy to get subtly wrong. So we also write the
obvious slow version — rebuild a boolean mask per candidate threshold and score
it with the general information_gain() — and assert the two agree on thousands
of random problems, including the cases most likely to break the fast path
(duplicate values, constant columns, tiny samples, min_samples_leaf > 1).

Run from the repo root:
    python scripts/test_tree.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tree import (  # noqa: E402
    DecisionTree,
    best_split,
    build_tree,
    entropy,
    information_gain,
    predict,
    tree_depth,
)

TOL = 1e-9
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def naive_best_split(X, y, min_samples_leaf=1):
    """Deliberately slow reference: O(n^2), built only from entropy()/IG()."""
    best = (None, None, 0.0)
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2.0:
            go_left = X[:, feature] <= threshold
            left, right = y[go_left], y[~go_left]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            gain = information_gain(y, left, right)
            if gain > best[2] + TOL:
                best = (feature, float(threshold), float(gain))
    return best


# --- 1. Differential test against the naive reference ----------------------
rng = np.random.default_rng(0)
mismatches, trials = 0, 400
for _ in range(trials):
    n = int(rng.integers(2, 60))
    n_features = int(rng.integers(1, 5))
    # Small integer range forces many duplicate values and ties — the exact
    # situation where a sweep can miscount if the "values differ" guard is wrong.
    X = rng.integers(0, 5, size=(n, n_features)).astype(float)
    if rng.random() < 0.25:  # sometimes make a column constant
        X[:, 0] = 1.0
    y = rng.integers(0, 2, size=n)
    msl = int(rng.integers(1, 4))

    fast_f, fast_t, fast_gain = best_split(X, y, min_samples_leaf=msl)
    slow_f, slow_t, slow_gain = naive_best_split(X, y, min_samples_leaf=msl)
    if abs(fast_gain - slow_gain) > 1e-9:
        mismatches += 1
        continue
    # Matching gain isn't enough — the same gain at the wrong column or cut
    # point would still be a broken split. Compare the partition itself,
    # since distinct (feature, threshold) pairs can tie on gain.
    if fast_f is not None:
        fast_mask = X[:, fast_f] <= fast_t
        slow_mask = X[:, slow_f] <= slow_t
        if not (np.array_equal(fast_mask, slow_mask)
                or np.array_equal(fast_mask, ~slow_mask)):
            mismatches += 1

check(f"best_split matches naive reference over {trials} random problems",
      mismatches == 0, f"({mismatches} mismatches)")

# --- 2. Edge cases ---------------------------------------------------------
check("pure node yields no split",
      best_split(np.array([[1.0], [2.0]]), np.array([1, 1])) == (None, None, 0.0))
check("constant feature yields no split",
      best_split(np.array([[3.0], [3.0]]), np.array([0, 1])) == (None, None, 0.0))
check("min_samples_leaf blocks an otherwise perfect split",
      best_split(np.array([[1.0], [2.0]]), np.array([0, 1]),
                 min_samples_leaf=2) == (None, None, 0.0))

perfect = best_split(np.array([[1.0], [2.0]]), np.array([0, 1]))
check("perfect binary split gains a full bit", abs(perfect[2] - 1.0) < TOL)

try:
    best_split(np.array([[1.0], [2.0]]), np.array([1, 2]))
    check("non-binary labels rejected", False, "(no error raised)")
except ValueError:
    check("non-binary labels rejected", True)

try:
    build_tree(np.array([[1.0], [2.0]]), np.array([0, 1]), feature_subset=1, rng=None)
    check("feature_subset without rng rejected", False, "(no error raised)")
except ValueError:
    check("feature_subset without rng rejected", True)

check("entropy of a pure set is exactly 0.0",
      str(entropy(np.array([1, 1, 1]))) == "0.0", f"(got {entropy(np.array([1, 1, 1]))})")

# Float values whose midpoint rounds up to the right-hand value: this used to
# send every sample left and recurse forever.
huge = np.array([[1e16 + 2.0], [1e16 + 4.0]])
root = build_tree(huge, np.array([0, 1]))
check("degenerate float midpoint terminates and splits", not root.is_leaf())

# --- 3. The tree learns something it should ------------------------------
# y is exactly feature 0 > 0.5, so a correct tree must reach 100% train accuracy
# and do it with a single split.
X = rng.random((200, 3))
y = (X[:, 0] > 0.5).astype(int)
tree = DecisionTree().fit(X, y)
check("tree learns a clean rule perfectly", tree.score(X, y) == 1.0)
check("and uses only one split to do it", tree.depth == 1, f"(depth={tree.depth})")

# Pure noise: labels independent of features. An unpruned tree should memorise
# the training set; a depth-1 tree should not.
y_noise = rng.integers(0, 2, size=200)
deep = DecisionTree(max_depth=None).fit(X, y_noise)
stump = DecisionTree(max_depth=1).fit(X, y_noise)
check("unpruned tree memorises pure noise", deep.score(X, y_noise) > 0.95,
      f"(train acc {deep.score(X, y_noise):.3f})")
check("stump does not", stump.score(X, y_noise) < 0.75,
      f"(train acc {stump.score(X, y_noise):.3f})")

# --- 4. Pre-pruning knobs are actually respected --------------------------
for d in (1, 2, 3, 5):
    t = DecisionTree(max_depth=d).fit(X, y_noise)
    if t.depth > d:
        check(f"max_depth={d} respected", False, f"(got depth {t.depth})")
        break
else:
    check("max_depth is respected at every setting tried", True)


def leaf_sizes(node, X, y):
    """Row counts of every leaf, walking the same way predict() does."""
    if node.is_leaf():
        return [len(y)]
    go_left = X[:, node.feature] <= node.threshold
    return (leaf_sizes(node.left, X[go_left], y[go_left])
            + leaf_sizes(node.right, X[~go_left], y[~go_left]))


t = DecisionTree(min_samples_leaf=20).fit(X, y_noise)
sizes = leaf_sizes(t.root, X, y_noise)
check("min_samples_leaf is respected in every leaf",
      len(sizes) > 1 and min(sizes) >= 20, f"(smallest leaf {min(sizes)})")

# min_samples_split: a node with fewer rows than this must not be split, so a
# threshold above the sample count has to collapse the tree to a single leaf.
big = DecisionTree(min_samples_split=len(y) + 1).fit(X, y)
check("min_samples_split above n forces a single leaf", big.depth == 0)
split_sizes = leaf_sizes(DecisionTree(min_samples_split=80).fit(X, y_noise).root,
                         X, y_noise)
check("min_samples_split=80 leaves no splittable node unsplit",
      len(split_sizes) > 1)

# --- 5. predict() agrees with the traversal on the training rows ----------
check("predict() is consistent with the fitted tree",
      np.array_equal(predict(tree.root, X), tree.predict(X)))
check("tree_depth matches the wrapper property", tree_depth(tree.root) == tree.depth)

# --- 6. leaf probabilities ------------------------------------------------
# proba must be a real probability everywhere.
for name, arr in (("signal", y), ("noise", y_noise)):
    t6 = DecisionTree(max_depth=4, min_samples_leaf=5).fit(X, arr)
    pr = t6.predict_proba(X)
    check(f"predict_proba in [0, 1] ({name})",
          bool(np.all(pr >= 0.0) and np.all(pr <= 1.0)))
    # The hard prediction is the STRICT threshold, because _majority_class
    # breaks a tie toward 0. Using >= would flip every 50/50 leaf to class 1.
    check(f"predict() == (predict_proba() > 0.5) ({name})",
          np.array_equal(t6.predict(X), (pr > 0.5).astype(int)))
    # A tree can emit at most one distinct probability per leaf.
    check(f"distinct probabilities <= n_leaves ({name})",
          len(np.unique(pr)) <= t6.n_leaves,
          f"({len(np.unique(pr))} distinct, {t6.n_leaves} leaves)")

# Pin the tie convention directly: force a single leaf over balanced labels.
y_tied = np.array([0, 1] * 20)
tied = DecisionTree(min_samples_split=len(y_tied) + 1).fit(X[:40], y_tied)
check("50/50 leaf: proba is 0.5 and prediction is 0",
      tied.predict_proba(X[:40])[0] == 0.5 and tied.predict(X[:40])[0] == 0)

# --------------------------------------------------------------------------
print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("All tree checks pass.")
