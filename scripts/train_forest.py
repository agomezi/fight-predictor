"""Train the from-scratch random forest and compare it to the single tree.

Same chronological split as scripts/train_tree.py, so the numbers are directly
comparable. Reports:
  * baseline, two single-tree references, forest — test accuracy side by side.
    The two references are deliberate: a hardcoded depth-4 tree (what earlier
    numbers were quoted against) and a tree matching the forest's own pruning
    settings. Only the second isolates what the ensemble actually buys.
  * OOB accuracy vs chronological test accuracy, including the SIGN of the gap,
    which here runs opposite to the textbook expectation
  * a sweep over n_trees, to show accuracy flattening rather than overfitting
  * feature importances aggregated across the forest

Run from the repo root (venv active):
    python scripts/train_forest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_table,
    chronological_split,
    to_matrix,
)
from src.forest import RandomForest  # noqa: E402
from src.tree import DecisionTree  # noqa: E402

RANDOM_SEED = 42


def main() -> None:
    features, stats = build_feature_table(seed=RANDOM_SEED)
    train_df, test_df = chronological_split(features, test_frac=0.18)
    X_train, y_train = to_matrix(train_df)
    X_test, y_test = to_matrix(test_df)

    majority = int(np.bincount(y_train).argmax())
    baseline = float(np.mean(y_test == majority))

    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    print(f"train / test : {len(y_train)} / {len(y_test)}")
    print(f"features     : {len(FEATURE_NAMES)}")
    print(f"baseline test accuracy: {baseline:.4f}")

    # Two single-tree reference points, because they answer different
    # questions and conflating them is how the forest's edge gets misread.
    #
    # 1. FIXED-DEPTH tree. NOT the part-3 pruned tree: part 3 selects max_depth
    #    on a chronological validation split, and lands on a different depth.
    #    This one is a hardcoded depth-4 tree, kept because the earlier numbers
    #    were reported against it. It is not a tuned model and should not be
    #    quoted as one.
    fixed_tree = DecisionTree(max_depth=4, min_samples_split=50,
                              min_samples_leaf=25).fit(X_train, y_train)
    print(f"single tree, depth 4 (fixed) : {fixed_tree.score(X_test, y_test):.4f}")

    # 2. CONFIGURATION-MATCHED tree: identical pruning settings to the trees
    #    inside the forest, but seeing every feature at every node instead of a
    #    random subset. This is the comparison that actually isolates what the
    #    ensemble buys, since it holds tree shape fixed and varies only bagging
    #    plus feature subsampling. The fixed-depth tree above differs from the
    #    forest in BOTH pruning and ensembling, so any gap against it is
    #    confounded.
    matched_tree = DecisionTree(max_depth=12, min_samples_split=10,
                                min_samples_leaf=5).fit(X_train, y_train)
    print(f"single tree, forest config   : {matched_tree.score(X_test, y_test):.4f}")

    print("\n" + "=" * 78)
    print("RANDOM FOREST")
    print("=" * 78)
    t0 = time.perf_counter()
    forest = RandomForest(n_trees=200, max_depth=12, min_samples_split=10,
                          min_samples_leaf=5, feature_subset="sqrt",
                          oob_score=True, random_state=RANDOM_SEED)
    forest.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_acc = forest.score(X_train, y_train)
    test_acc = forest.score(X_test, y_test)
    print(f"fit time      : {fit_seconds:.1f}s for {forest.n_trees} trees")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test  accuracy: {test_acc:.4f}")
    print(f"OOB   accuracy: {forest.oob_score_:.4f}" if forest.oob_score_
          else "OOB   accuracy: n/a")
    if forest.oob_score_:
        gap = forest.oob_score_ - test_acc
        print(f"OOB minus test: {gap:+.4f}")
        if gap < 0:
            print("                OOB reads LOW. A random-split estimate normally")
            print("                flatters you, so the sign is informative: the")
            print("                1994- training pool is harder than the 2023-2026")
            print("                test window. See _compute_oob_score's docstring.")
        else:
            print("                OOB is a random split; test is chronological.")
    print(f"vs single tree, depth 4      : "
          f"{test_acc - fixed_tree.score(X_test, y_test):+.4f}  (confounded)")
    print(f"vs single tree, forest config: "
          f"{test_acc - matched_tree.score(X_test, y_test):+.4f}  <- the real ensemble effect")
    print(f"vs baseline                  : {test_acc - baseline:+.4f}")

    print("\n" + "=" * 78)
    print("n_trees SWEEP — accuracy flattens, it does not degrade")
    print("=" * 78)
    print(f"{'n_trees':>8} {'train':>8} {'test':>8} {'oob':>8}")
    for n in (1, 5, 10, 25, 50, 100, 200):
        f = RandomForest(n_trees=n, max_depth=12, min_samples_split=10,
                         min_samples_leaf=5, feature_subset="sqrt",
                         oob_score=True, random_state=RANDOM_SEED)
        f.fit(X_train, y_train)
        oob = f"{f.oob_score_:.4f}" if f.oob_score_ else "   n/a"
        print(f"{n:>8} {f.score(X_train, y_train):>8.4f} "
              f"{f.score(X_test, y_test):>8.4f} {oob:>8}")

    print("\n" + "=" * 78)
    print("FEATURE IMPORTANCES (gain-weighted, across all trees)")
    print("=" * 78)
    importances = forest.feature_importances(X_train.shape[1])
    for i in np.argsort(importances)[::-1]:
        bar = "#" * int(round(importances[i] * 60))
        print(f"{FEATURE_NAMES[i]:>28} {importances[i]:.4f} {bar}")


if __name__ == "__main__":
    main()
