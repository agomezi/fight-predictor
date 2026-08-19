"""Step 5 - sanity-check the from-scratch tree against scikit-learn.

The point is NOT to prove the two implementations are byte-identical. They
will not be, and expecting that misreads the exercise. The point is to
establish that:

  1. The information-gain math agrees with an independent implementation.
  2. The split search finds the same cut point on real data.
  3. Where predictions diverge, the cause is explainable (almost always ties
     between equally-good splits, which the two libraries break differently).

Run from the repo root:
    .venv/Scripts/python.exe scripts/compare_sklearn.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.features import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_table,
    chronological_split,
    to_matrix,
)
from src.tree import DecisionTree, best_split, entropy  # noqa: E402

RANDOM_SEED = 42

# Matched hyperparameters. The semantics line up between the two
# implementations: a node needs >= min_samples_split rows to be considered,
# and no split may leave a child with < min_samples_leaf rows.
PARAMS = dict(max_depth=4, min_samples_split=50, min_samples_leaf=25)


def fit_both(X, y, **params):
    """Fit ours and sklearn's on identical data. Returns models and fit times."""
    t0 = time.perf_counter()
    ours = DecisionTree(**params).fit(X, y)
    t_ours = time.perf_counter() - t0

    t0 = time.perf_counter()
    theirs = DecisionTreeClassifier(
        criterion="entropy", splitter="best", max_features=None,
        random_state=RANDOM_SEED, **params,
    ).fit(X, y)
    t_theirs = time.perf_counter() - t0

    return ours, theirs, t_ours, t_theirs


def detect_log_base(theirs, y):
    """Work out whether sklearn's entropy impurity is in bits or nats.

    Compares sklearn's root impurity against entropy() from tree.py, which is
    in bits by construction. A ratio near 1.0 means sklearn is also in bits;
    near ln(2) = 0.693 means sklearn is in nats.
    """
    ours_bits = entropy(y)
    theirs_root = float(theirs.tree_.impurity[0])
    if ours_bits == 0.0:
        return "undetermined (root is pure)", float("nan")
    ratio = theirs_root / ours_bits
    if abs(ratio - 1.0) < 1e-6:
        return "bits (log2) - same as ours", ratio
    if abs(ratio - np.log(2)) < 1e-6:
        return "nats (natural log) - ours is bits, differs by ln(2)", ratio
    return "unrecognised scaling", ratio


def sklearn_root_gain(theirs):
    """Information gain of sklearn's root split, in sklearn's own units."""
    t = theirs.tree_
    left, right = int(t.children_left[0]), int(t.children_right[0])
    if left < 0:  # root is a leaf
        return 0.0
    n = t.weighted_n_node_samples
    return float(
        t.impurity[0] - (n[left] * t.impurity[left] + n[right] * t.impurity[right]) / n[0]
    )


def count_root_ties(X, y, best_gain, tol=1e-12):
    """How many (feature, threshold) candidates tie the winning gain at the root.

    This is the diagnostic that explains disagreement. If several candidates
    are within `tol` of the best gain, the two libraries are choosing between
    equally-good splits, and picking differently is not a bug.
    """
    ties = []
    for feature in range(X.shape[1]):
        f, thr, gain = best_split(
            X, y, feature_indices=[feature],
            min_samples_leaf=PARAMS["min_samples_leaf"],
        )
        if f is not None and abs(gain - best_gain) <= tol:
            ties.append((FEATURE_NAMES[f], thr, gain))
    return ties


def compare(label, X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "=" * 78)
    print(f"{label}  (train={len(y_train)}, test={len(y_test)})")
    print("=" * 78)

    ours, theirs, t_ours, t_theirs = fit_both(X_train, y_train, **PARAMS)

    # --- 1. Does the impurity math agree? --------------------------------
    base_desc, ratio = detect_log_base(theirs, y_train)
    print(f"\nroot entropy (ours, bits) : {entropy(y_train):.6f}")
    print(f"root impurity (sklearn)   : {theirs.tree_.impurity[0]:.6f}")
    print(f"sklearn units             : {base_desc}  (ratio {ratio:.6f})")

    # --- 2. Does the split search find the same cut? ---------------------
    o_root = ours.root
    t_feat = int(theirs.tree_.feature[0])
    print("\n--- root split ---")
    if o_root.is_leaf():
        print("ours    : leaf (no split found)")
    else:
        print(f"ours    : {feature_names[o_root.feature]} <= {o_root.threshold:.6f}"
              f"  (gain {o_root.gain:.6f} bits)")
    if t_feat < 0:
        print("sklearn : leaf (no split found)")
    else:
        print(f"sklearn : {feature_names[t_feat]} <= {theirs.tree_.threshold[0]:.6f}"
              f"  (gain {sklearn_root_gain(theirs):.6f} in sklearn units)")

    same_root = (
        not o_root.is_leaf() and t_feat >= 0
        and o_root.feature == t_feat
        and abs(o_root.threshold - float(theirs.tree_.threshold[0])) < 1e-9
    )
    print(f"identical root split      : {same_root}")

    if not same_root and not o_root.is_leaf():
        ties = count_root_ties(X_train, y_train, o_root.gain)
        print(f"\ncandidates tying the best gain at the root: {len(ties)}")
        for name, thr, gain in ties:
            print(f"    {name} <= {thr:.6f}  (gain {gain:.6f})")
        if len(ties) > 1:
            print("  -> equally-good splits. Different choice is tie-breaking,")
            print("     not a bug: sklearn permutes feature order via random_state,")
            print("     best_split keeps the first candidate it encounters.")

    # --- 3. Do the predictions agree? -----------------------------------
    p_ours, p_theirs = ours.predict(X_test), theirs.predict(X_test)
    agree = float(np.mean(p_ours == p_theirs))
    print("\n--- predictions on the held-out set ---")
    print(f"row agreement             : {agree:.4f}  "
          f"({int(np.sum(p_ours != p_theirs))} of {len(y_test)} rows differ)")
    print(f"accuracy  ours / sklearn  : {ours.score(X_test, y_test):.4f} / "
          f"{theirs.score(X_test, y_test):.4f}")
    print(f"tree size ours / sklearn  : depth {ours.depth}/{theirs.get_depth()}, "
          f"leaves {ours.n_leaves}/{theirs.get_n_leaves()}")
    print(f"fit time  ours / sklearn  : {t_ours:.3f}s / {t_theirs:.3f}s "
          f"({t_ours / max(t_theirs, 1e-9):.0f}x slower)")
    return agree


def main() -> None:
    # A small, deliberately unambiguous problem first. With clean structure and
    # no ties, the two implementations should agree exactly - so a failure here
    # is a real bug, not tie-breaking.
    rng = np.random.default_rng(RANDOM_SEED)
    n = 600
    X_syn = rng.normal(size=(n, 4))
    y_syn = (X_syn[:, 0] + 0.5 * X_syn[:, 1] > 0).astype(int)
    cut = int(n * 0.7)
    compare(
        "SYNTHETIC - clean signal, expect exact agreement",
        X_syn[:cut], y_syn[:cut], X_syn[cut:], y_syn[cut:],
        [f"x{i}" for i in range(4)],
    )

    # Then the real thing: weak signal, many duplicate feature values (the
    # *_missing indicators are 0/1, so ties are common), 8 features.
    features, _ = build_feature_table(seed=RANDOM_SEED)
    train_df, test_df = chronological_split(features, test_frac=0.18)
    X_train, y_train = to_matrix(train_df)
    X_test, y_test = to_matrix(test_df)
    agree = compare(
        "REAL UFC DATA - chronological split",
        X_train, y_train, X_test, y_test, FEATURE_NAMES,
    )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if agree >= 0.95:
        print(f"{agree:.1%} row agreement - within tolerance. The split search and")
        print("gain math are validated against an independent implementation.")
        print("Residual disagreement is expected: ties broken differently, and")
        print("best_split requires gain > 0 strictly while sklearn will accept a")
        print("zero-impurity-decrease split.")
    else:
        print(f"{agree:.1%} row agreement - below the 95% tolerance. Investigate.")
        print("Start at the root split comparison above: if the roots differ and")
        print("nothing ties, the bug is in best_split, not in tie-breaking.")


if __name__ == "__main__":
    main()
