"""Score the v1 models properly: log loss, Brier, bootstrap intervals, folds.

Extends the README's accuracy table with the metrics that read the probability
instead of the hard class, and puts an error bar on each. The three models are
the same three the README lists, so the numbers line up directly:

    fixed depth-4 tree      the reference earlier numbers were quoted against
    forest-config tree      forest pruning, all features -- the matched baseline
    random forest           200 trees

Reports proper scoring rules with bootstrap confidence intervals, a calibration
table, and a walk-forward fold breakdown so a result can be checked for whether
it holds across eras rather than in one window.

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
    delta_verdict,
    paired_bootstrap_ci,
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
from src.history import (  # noqa: E402
    EloIndex,
    HistoryIndex,
    build_event_log,
    division_priors,
)
from src.matchup import (  # noqa: E402
    FighterBios,
    build_training_matrix,
    feature_columns,
)
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

    # --- feature sets ----------------------------------------------------
    # The model config is held fixed and the FEATURE SET is varied, which is
    # the comparison every step of the accuracy work needs and which this
    # script previously could not make at all.
    rule("FEATURE SETS — does each layer earn its keep?")
    print("Building the rolling matrices (features_as_of runs once per corner")
    print("per fight, so this takes a minute on the full dataset)...")

    bios = FighterBios()
    log, log_info = build_event_log(seed=RANDOM_SEED)
    hist, priors = HistoryIndex(log), division_priors(log)
    elo_index = EloIndex(log)
    if log_info["stats_missing"]:
        print(f"  NOTE: stat columns absent from the raw table: "
              f"{log_info['stats_missing']}")

    variants = (
        ("static only", dict(index=None, priors=None, elo_index=None)),
        ("+ rolling form", dict(index=hist, priors=priors, elo_index=None)),
        ("+ rolling + live Elo", dict(index=hist, priors=priors,
                                      elo_index=elo_index)),
        # Bout context: title / women's / nonstandard-weight / scheduled rounds.
        # All already in the raw tables and never carried through. Symmetric
        # across corners, so they carry no directional signal alone and can only
        # pay via interactions -- measured, not assumed.
        ("+ rolling + bout context",
         dict(index=hist, priors=priors, elo_index=None,
              columns=feature_columns(with_rolling=True, with_bout_context=True))),
    )

    # STEP A's subtraction test, in the same pass: drop the collinear cluster
    # HANDBACK-4 identified. With sqrt(n) columns sampled per node, redundant
    # axes crowd out informative ones, so removing them can raise accuracy.
    DROP = {"win_rate_raw_diff", "elo_diff", "n_fights_diff",
            "total_fight_secs_diff"}
    variants = variants + ((
        "+ rolling, pruned",
        dict(index=hist, priors=priors, elo_index=None,
             columns=[c for c in feature_columns(with_rolling=True)
                      if c not in DROP]),
    ),)

    scored = []
    for label, kw in variants:
        Xtr, ytr, cols = build_training_matrix(train_df, bios, **kw)
        # A variant may already pin `columns`; the test matrix must use exactly
        # the order the training call resolved, so drop any duplicate key rather
        # than passing it twice.
        kw_test = {k: v for k, v in kw.items() if k != "columns"}
        Xte, yte, _ = build_training_matrix(test_df, bios, columns=cols, **kw_test)
        model = RandomForest(
            n_trees=200, max_depth=12, min_samples_split=10, min_samples_leaf=5,
            feature_subset="sqrt", oob_score=False, random_state=RANDOM_SEED,
        ).fit(Xtr, ytr)
        p = np.asarray(model.predict_proba(Xte), dtype=float)
        scored.append((label, cols, p, yte))

    print(f"\n{'feature set':<24}{'cols':>6}{'accuracy':>10}{'log_loss':>10}{'brier':>9}")
    for label, cols, p, yte in scored:
        print(f"{label:<24}{len(cols):>6}{accuracy(yte, p):>10.4f}"
              f"{log_loss(yte, p):>10.4f}{brier_score(yte, p):>9.4f}")

    # elo_diff carried no information until EloIndex was threaded in; show that
    # it is now a live column rather than asserting it.
    for label, cols, _, _ in scored:
        if "elo_diff" in cols:
            Xtr, _, c2 = build_training_matrix(
                train_df, bios,
                **dict(variants[2][1] if "Elo" in label else variants[1][1]))
            v = float(np.var(Xtr[:, c2.index("elo_diff")]))
            print(f"  elo_diff variance, {label:<22} {v:>12.4f}"
                  + ("   (inert)" if v == 0.0 else "   (live)"))

    rule("PAIRED DIFFERENCES — the test the discipline rule asks for")
    print("Same rows, same resample, both models. Cancels the shared test-set")
    print("noise, so it resolves differences two separate intervals cannot.\n")
    base_label, _, p_base, y_ref = scored[0]
    for label, _, p, _ in scored[1:]:
        print(f"{label}  vs  {base_label}")
        for name, fn, lower_better in METRICS:
            d = paired_bootstrap_ci(y_ref, p, p_base, fn, n_boot=N_BOOT,
                                    rng=np.random.default_rng(RANDOM_SEED))
            print(f"  {name:<9} {d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]  "
                  f"{delta_verdict(d, lower_better)}")
        print()
    # And the one STEP 1 actually asks: live Elo against rolling-without-Elo.
    if len(scored) == 3:
        print(f"{scored[2][0]}  vs  {scored[1][0]}   (the live-Elo question)")
        for name, fn, lower_better in METRICS:
            d = paired_bootstrap_ci(y_ref, scored[2][2], scored[1][2], fn,
                                    n_boot=N_BOOT,
                                    rng=np.random.default_rng(RANDOM_SEED))
            print(f"  {name:<9} {d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]  "
                  f"{delta_verdict(d, lower_better)}")

    # --- feature sets across folds ---------------------------------------
    # The single-tail gain above rests on one 2023-2026 window, and its lower
    # bound is thin. Walk-forward asks whether it holds across eras, which is
    # the only way to tell a real feature from a lucky window.
    rule("FEATURE SETS ACROSS WALK-FORWARD FOLDS")
    ordered_all = features.sort_values("Event_Date", kind="mergesort")
    dates_all = ordered_all["Event_Date"].to_numpy()
    for label, kw in variants:
        Xa, ya, _cols = build_training_matrix(ordered_all, bios, **kw)
        rows = run_walk_forward(
            lambda Xt, yt: RandomForest(
                n_trees=100, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, feature_subset="sqrt", oob_score=False,
                random_state=RANDOM_SEED).fit(Xt, yt),
            Xa, ya, dates_all, n_folds=8, min_train_frac=0.5,
        )
        summary = summarise_folds(rows)
        accs = " ".join(f"{r['accuracy']:.3f}" for r in rows)
        print(f"\n{label}")
        print(f"  per-fold accuracy: {accs}")
        for name, (mean, sd) in summary.items():
            print(f"  {name:<9} mean {mean:.4f}  sd {sd:.4f}")
    print("\nCompare the MEANS between feature sets against the sd WITHIN each.")
    print("A gain smaller than the fold-to-fold spread is not established by a")
    print("single tail, however cleanly its bootstrap interval cleared zero.")

    rule("THE QUESTION THIS EXISTS TO ANSWER")
    print("Does the forest's +4.2 points over the matched single tree survive")
    print("walk-forward, or was it an artifact of one 2023-2026 window? Read the")
    print("per-fold table above against the fold-to-fold standard deviation.")


if __name__ == "__main__":
    main()
