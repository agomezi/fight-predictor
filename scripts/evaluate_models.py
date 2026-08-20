"""Score the v1 models properly: log loss, Brier, bootstrap intervals, folds.

Extends the README's accuracy table with the metrics that read the probability
instead of the hard class, and puts an error bar on each. The three models are
the same three the README lists, so the numbers line up directly:

    fixed depth-4 tree      the reference earlier numbers were quoted against
    forest-config tree      forest pruning, all features -- the matched baseline
    random forest           200 trees

The bootstrap and walk-forward sections depend on the TODO(human) blocks in
src/evaluate.py and are skipped with a message until those are filled, so this
script is useful before they are.

Run from the repo root (venv active):
    python scripts/evaluate_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (  # noqa: E402
    METRICS,
    accuracy,
    bootstrap_ci,
    brier_score,
    ci_half_width,
    log_loss,
    reference_scores,
    reliability_table,
    run_walk_forward,
    summarise_folds,
)
from src.features import (  # noqa: E402
    build_feature_table,
    chronological_split,
    to_matrix,
)
from src.forest import RandomForest  # noqa: E402
from src.tree import DecisionTree  # noqa: E402

RANDOM_SEED = 42
N_BOOT = 2000

# (label, factory) -- factories so walk-forward can refit per fold.
MODELS = (
    ("tree, depth 4 (fixed)",
     lambda: DecisionTree(max_depth=4, min_samples_split=50, min_samples_leaf=25)),
    ("tree, forest config",
     lambda: DecisionTree(max_depth=12, min_samples_split=10, min_samples_leaf=5)),
    ("forest, 200 trees",
     lambda: RandomForest(n_trees=200, max_depth=12, min_samples_split=10,
                          min_samples_leaf=5, feature_subset="sqrt",
                          oob_score=False, random_state=RANDOM_SEED)),
)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    features, stats = build_feature_table(seed=RANDOM_SEED)
    train_df, test_df = chronological_split(features, test_frac=0.18)
    X_train, y_train = to_matrix(train_df)
    X_test, y_test = to_matrix(test_df)

    rule("SETUP")
    print(f"train / test : {len(y_train)} / {len(y_test)}")
    print(f"test window  : {test_df['Event_Date'].min().date()} -> "
          f"{test_df['Event_Date'].max().date()}")
    ref = reference_scores(y_test)
    print("\nalways-predict-the-base-rate reference (the honest zero point):")
    print(f"  accuracy {ref['accuracy']:.4f} | log_loss {ref['log_loss']:.4f} "
          f"| brier {ref['brier']:.4f}")
    print("  A model can beat this on accuracy and LOSE on log loss. That is")
    print("  what overconfidence looks like, and accuracy alone hides it.")

    # --- single chronological split -------------------------------------
    rule("SINGLE SPLIT — proper scoring rules")
    fitted = []
    print(f"{'model':<24} {'accuracy':>9} {'log_loss':>9} {'brier':>8}")
    for label, factory in MODELS:
        model = factory().fit(X_train, y_train)
        p = np.asarray(model.predict_proba(X_test), dtype=float)
        fitted.append((label, model, p))
        print(f"{label:<24} {accuracy(y_test, p):>9.4f} "
              f"{log_loss(y_test, p):>9.4f} {brier_score(y_test, p):>8.4f}")

    # --- error bars ------------------------------------------------------
    rule(f"BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOT} resamples, 95%)")
    try:
        # Probe once before printing anything, so an unimplemented bootstrap
        # does not leave a model label stranded above the skip message.
        bootstrap_ci(y_test[:8], fitted[0][2][:8], brier_score, n_boot=2,
                     rng=np.random.default_rng(0))
        for label, _, p in fitted:
            print(f"\n{label}")
            for name, fn, lower_better in METRICS:
                point, lo, hi = bootstrap_ci(
                    y_test, p, fn, n_boot=N_BOOT,
                    rng=np.random.default_rng(RANDOM_SEED),
                )
                arrow = "lower better" if lower_better else "higher better"
                print(f"  {name:<9} {point:.4f}  [{lo:.4f}, {hi:.4f}]  "
                      f"+/- {ci_half_width(lo, hi):.4f}  ({arrow})")
        print("\nRead the accuracy intervals against each other before believing")
        print("any delta. Overlapping intervals are NOT proof of no difference —")
        print("for that, bootstrap the paired difference on the same rows.")
    except NotImplementedError as exc:
        print(f"SKIPPED — {exc}")

    # --- calibration -----------------------------------------------------
    rule("CALIBRATION — predicted vs observed win rate")
    for label, _, p in fitted:
        table = reliability_table(y_test, p, n_bins=10)
        print(f"\n{label}  ({len(table)} of 10 bins populated)")
        print(f"  {'bin':<14}{'n':>7}{'predicted':>11}{'observed':>10}{'error':>9}")
        for r in table:
            err = r["mean_predicted"] - r["observed"]
            print(f"  [{r['bin_lo']:.1f}, {r['bin_hi']:.1f})".ljust(16)
                  + f"{r['n']:>5}{r['mean_predicted']:>11.3f}"
                  + f"{r['observed']:>10.3f}{err:>+9.3f}")
    print("\nA single tree can only emit one probability per leaf, so most bins")
    print("stay empty and the curve is step-shaped. The forest averages 200")
    print("trees and should populate far more bins — that difference is the")
    print("clearest argument for the ensemble that accuracy never showed.")

    # --- walk-forward ----------------------------------------------------
    rule("WALK-FORWARD FOLDS")
    dates = features.sort_values("Event_Date", kind="mergesort")["Event_Date"]
    ordered = features.sort_values("Event_Date", kind="mergesort")
    X_all, y_all = to_matrix(ordered)
    try:
        for label, factory in MODELS:
            rows = run_walk_forward(
                lambda Xt, yt, f=factory: f().fit(Xt, yt),
                X_all, y_all, dates.to_numpy(), n_folds=8, min_train_frac=0.5,
            )
            print(f"\n{label}")
            print(f"  {'fold':>4}{'n_train':>9}{'n_test':>8}  {'test window':<25}"
                  f"{'acc':>8}{'logloss':>9}{'brier':>8}")
            for r in rows:
                window = f"{str(r['test_start'])[:10]} -> {str(r['test_end'])[:10]}"
                print(f"  {r['fold']:>4}{r['n_train']:>9}{r['n_test']:>8}  "
                      f"{window:<25}{r['accuracy']:>8.4f}"
                      f"{r['log_loss']:>9.4f}{r['brier']:>8.4f}")
            summary = summarise_folds(rows)
            for name, (mean, sd) in summary.items():
                print(f"  {name:<9} mean {mean:.4f}  sd {sd:.4f}")
        print("\nThe fold-to-fold standard deviation is the point. If it swamps")
        print("the gap between two models, the single-split number could not have")
        print("distinguished them however clean it looked.")
    except NotImplementedError as exc:
        print(f"SKIPPED — {exc}")

    rule("THE QUESTION THIS EXISTS TO ANSWER")
    print("Does the forest's +4.2 points over the matched single tree survive")
    print("walk-forward, or was it an artifact of one 2023-2026 window?")
    print("Fill the two TODO(human) blocks in src/evaluate.py and this script")
    print("answers it.")


if __name__ == "__main__":
    main()
