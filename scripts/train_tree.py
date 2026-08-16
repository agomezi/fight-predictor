"""Train the from-scratch decision tree and make overfitting visible.

Grows two trees on the identical chronological split:
  * UNPRUNED — no depth cap, splits down to single-sample purity
  * PRUNED   — pre-pruning via max_depth / min_samples_split / min_samples_leaf

The point isn't that one is "better". It's that the unpruned tree memorises
the training fights (train accuracy near 100%) while doing no better, and
usually worse, on fights it hasn't seen.

Run from the repo root (venv active):
    python scripts/train_tree.py
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
from src.tree import DecisionTree  # noqa: E402

RANDOM_SEED = 42


def confusion(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary 0/1 arrays."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def report(name, model, X_train, y_train, X_test, y_test, fit_seconds):
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    tn, fp, fn, tp = confusion(y_test, model.predict(X_test))

    print(f"\n--- {name} ---")
    print(f"params        : max_depth={model.max_depth}, "
          f"min_samples_split={model.min_samples_split}, "
          f"min_samples_leaf={model.min_samples_leaf}")
    print(f"fit time      : {fit_seconds:.2f}s")
    print(f"tree size     : depth={model.depth}, leaves={model.n_leaves}, "
          f"internal nodes={model.n_internal}")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test  accuracy: {test_acc:.4f}")
    print(f"overfit gap   : {train_acc - test_acc:+.4f}  (train minus test)")
    print(f"test confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    return train_acc, test_acc


def print_tree(node, feature_names, depth=0, max_depth=3, prefix="root"):
    """Print the top few levels so the learned splits are readable."""
    pad = "  " * depth
    if node.is_leaf():
        print(f"{pad}{prefix}: predict {node.prediction}  (n={node.n_samples})")
        return
    if depth >= max_depth:
        print(f"{pad}{prefix}: [subtree, n={node.n_samples}]")
        return
    name = feature_names[node.feature]
    print(f"{pad}{prefix}: {name} <= {node.threshold:.3f}  "
          f"(n={node.n_samples}, gain={node.gain:.4f})")
    print_tree(node.left, feature_names, depth + 1, max_depth, "yes")
    print_tree(node.right, feature_names, depth + 1, max_depth, "no ")


def main() -> None:
    features, stats = build_feature_table(seed=RANDOM_SEED)
    train_df, test_df = chronological_split(features, test_frac=0.18)
    X_train, y_train = to_matrix(train_df)
    X_test, y_test = to_matrix(test_df)

    print("=" * 78)
    print("DATA")
    print("=" * 78)
    print(f"final rows      : {stats['n_final']}")
    print(f"train / test    : {len(y_train)} / {len(y_test)}")
    print(f"train window    : {train_df['Event_Date'].min().date()} -> "
          f"{train_df['Event_Date'].max().date()}")
    print(f"test  window    : {test_df['Event_Date'].min().date()} -> "
          f"{test_df['Event_Date'].max().date()}")
    print(f"features ({len(FEATURE_NAMES)})   : {', '.join(FEATURE_NAMES)}")

    # Baseline: always predict the majority class of the TRAINING set. Any
    # model that can't beat this has learned nothing worth having.
    majority = int(np.bincount(y_train).argmax())
    baseline = float(np.mean(y_test == majority))
    print(f"\nbaseline (always predict {majority}) test accuracy: {baseline:.4f}")

    print("\n" + "=" * 78)
    print("TWO TREES, SAME SPLIT")
    print("=" * 78)

    t0 = time.perf_counter()
    unpruned = DecisionTree(max_depth=None, min_samples_split=2,
                            min_samples_leaf=1).fit(X_train, y_train)
    t_unpruned = time.perf_counter() - t0
    u_train, u_test = report("UNPRUNED", unpruned, X_train, y_train,
                             X_test, y_test, t_unpruned)

    # Pick max_depth on a VALIDATION set, never on the test set. The
    # validation set is the most recent slice of the training window, carved
    # off chronologically for the same reason the test set is: tuning against
    # the test set would make the final number optimistic.
    print("\n" + "=" * 78)
    print("TUNING max_depth ON A CHRONOLOGICAL VALIDATION SPLIT")
    print("=" * 78)
    sub_train_df, val_df = chronological_split(train_df, test_frac=0.20)
    X_sub, y_sub = to_matrix(sub_train_df)
    X_val, y_val = to_matrix(val_df)
    print(f"sub-train: {len(y_sub)} rows  -> {sub_train_df['Event_Date'].max().date()}")
    print(f"validation: {len(y_val)} rows  "
          f"{val_df['Event_Date'].min().date()} -> {val_df['Event_Date'].max().date()}")
    print(f"\n{'max_depth':>10} {'sub-train':>10} {'validation':>11}")

    best_depth, best_val = None, -1.0
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
        m = DecisionTree(max_depth=d, min_samples_split=50,
                         min_samples_leaf=25).fit(X_sub, y_sub)
        val_acc = m.score(X_val, y_val)
        print(f"{d:>10} {m.score(X_sub, y_sub):>10.4f} {val_acc:>11.4f}")
        # Strict `>` means ties go to the SHALLOWEST depth tried, since the
        # sweep runs in increasing order. That's deliberate: when two depths
        # validate identically, prefer the simpler tree.
        if val_acc > best_val:
            best_depth, best_val = d, val_acc
    print(f"\nchosen max_depth = {best_depth} (validation accuracy {best_val:.4f})")

    t0 = time.perf_counter()
    pruned = DecisionTree(max_depth=best_depth, min_samples_split=50,
                          min_samples_leaf=25).fit(X_train, y_train)
    t_pruned = time.perf_counter() - t0
    p_train, p_test = report("PRUNED", pruned, X_train, y_train,
                             X_test, y_test, t_pruned)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"unpruned: train {u_train:.4f} -> test {u_test:.4f}  "
          f"(gap {u_train - u_test:+.4f})")
    print(f"pruned  : train {p_train:.4f} -> test {p_test:.4f}  "
          f"(gap {p_train - p_test:+.4f})")
    print(f"baseline:                test {baseline:.4f}")
    print(f"\npruning changed test accuracy by {p_test - u_test:+.4f}")
    print(f"pruned tree beats baseline by     {p_test - baseline:+.4f}")

    print("\n" + "=" * 78)
    print("PRUNED TREE — top 3 levels")
    print("=" * 78)
    print_tree(pruned.root, FEATURE_NAMES, max_depth=3)

    # Reported last and read-only: this is the shape of the overfitting curve,
    # NOT how max_depth was chosen (that was the validation sweep above).
    print("\n" + "=" * 78)
    print("DEPTH SWEEP ON TEST — the overfitting curve, for illustration only")
    print("=" * 78)
    print(f"{'max_depth':>10} {'train':>8} {'test':>8} {'leaves':>8}")
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
        m = DecisionTree(max_depth=d, min_samples_split=50,
                         min_samples_leaf=25).fit(X_train, y_train)
        print(f"{d:>10} {m.score(X_train, y_train):>8.4f} "
              f"{m.score(X_test, y_test):>8.4f} {m.n_leaves:>8}")


if __name__ == "__main__":
    main()
