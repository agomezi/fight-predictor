"""Blend the forest with the booster, and calibrate the result.

Both questions share one setup, which is the reason they share a script: each
needs a HELD-OUT slice to fit on -- a blend weight and a calibrator are both
parameters, and fitting either on the test tail would be tuning on test.

The split is three-way and chronological throughout:

    sub-train        fit the forest and the booster
    validation       fit the calibrator, choose the blend weight
    test (the tail)  report everything

Nothing is chosen on the test rows. That is the only methodological point here,
and it is the whole reason the numbers can be believed.

Expected outcome, stated before running: calibration should improve LOG LOSS and
leave accuracy flat, because it rescales probabilities without reordering them.
Isotonic is monotone, so it cannot change the ranking at all and accuracy can
only move where it crosses 0.5. If calibration does not improve out-of-fold log
loss, it does not ship -- that result goes in the README instead.

    python scripts/experiment_blend_calibrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boosting import GradientBoosting  # noqa: E402
from src.calibrate import (  # noqa: E402
    IsotonicCalibrator,
    PlattCalibrator,
    best_blend_weight,
    blend,
)
from src.evaluate import (  # noqa: E402
    METRICS,
    accuracy,
    brier_score,
    delta_verdict,
    log_loss,
    paired_bootstrap_ci,
    reliability_table,
)
from src.features import build_feature_table, chronological_split  # noqa: E402
from src.forest import RandomForest  # noqa: E402
from src.history import HistoryIndex, build_event_log, division_priors  # noqa: E402
from src.matchup import (  # noqa: E402
    FighterBios,
    build_training_matrix,
    feature_columns,
)

SEED = 42
N_BOOT = 2000


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def show_reliability(label, y, p, n_bins=10):
    print(f"\n{label}")
    print(f"  {'bin':<14}{'n':>7}{'predicted':>11}{'observed':>10}{'gap':>8}")
    worst = 0.0
    for r in reliability_table(y, p, n_bins=n_bins):
        gap = r["mean_predicted"] - r["observed"]
        worst = max(worst, abs(gap))
        print(f"  [{r['bin_lo']:.1f}, {r['bin_hi']:.1f})".ljust(16)
              + f"{r['n']:>5}{r['mean_predicted']:>11.3f}"
              + f"{r['observed']:>10.3f}{gap:>+8.3f}")
    print(f"  worst absolute gap: {worst:.3f}")
    return worst


def main() -> None:
    features, _ = build_feature_table(seed=SEED)
    train_df, test_df = chronological_split(features, test_frac=0.18)
    # Carve validation off the END of train, chronologically -- the same reason
    # the test set is the tail. A random validation slice would let the
    # calibrator see fights later than the ones the models trained on.
    sub_df, val_df = chronological_split(train_df, test_frac=0.20)

    bios = FighterBios()
    log, _ = build_event_log(seed=SEED)
    hist, priors = HistoryIndex(log), division_priors(log)
    cols = feature_columns(with_rolling=True)

    def matrix(df, c=None):
        return build_training_matrix(df, bios, index=hist, priors=priors,
                                     elo_index=None, columns=c or cols)

    X_sub, y_sub, cols = matrix(sub_df)
    X_val, y_val, _ = matrix(val_df, cols)
    X_te, y_te, _ = matrix(test_df, cols)

    rule("SETUP — three-way chronological split")
    print(f"sub-train  {len(y_sub):>5} rows  -> {sub_df['Event_Date'].max().date()}")
    print(f"validation {len(y_val):>5} rows  "
          f"{val_df['Event_Date'].min().date()} -> {val_df['Event_Date'].max().date()}")
    print(f"test       {len(y_te):>5} rows  "
          f"{test_df['Event_Date'].min().date()} -> {test_df['Event_Date'].max().date()}")
    print("\nThe calibrator and the blend weight are fitted on VALIDATION only.")

    forest = RandomForest(n_trees=200, max_depth=12, min_samples_split=10,
                          min_samples_leaf=5, feature_subset="sqrt",
                          oob_score=False, random_state=SEED).fit(X_sub, y_sub)
    booster = GradientBoosting(n_rounds=110, learning_rate=0.05, max_depth=3,
                               min_samples_split=20, min_samples_leaf=10,
                               random_state=SEED).fit(X_sub, y_sub)

    pf_val = np.asarray(forest.predict_proba(X_val), dtype=float)
    pb_val = np.asarray(booster.predict_proba(X_val), dtype=float)
    pf_te = np.asarray(forest.predict_proba(X_te), dtype=float)
    pb_te = np.asarray(booster.predict_proba(X_te), dtype=float)

    rule("BASE MODELS on the test tail")
    print(f"{'model':<22}{'accuracy':>10}{'log_loss':>10}{'brier':>9}")
    for name, p in (("forest", pf_te), ("booster", pb_te)):
        print(f"{name:<22}{accuracy(y_te, p):>10.4f}{log_loss(y_te, p):>10.4f}"
              f"{brier_score(y_te, p):>9.4f}")

    # --- blend -----------------------------------------------------------
    rule("BLEND — weight chosen on validation log loss")
    w = best_blend_weight(pf_val, pb_val, y_val, log_loss)
    grid = np.linspace(0.0, 1.0, 11)
    print("validation log loss by forest weight:")
    print("  " + "  ".join(f"{g:.1f}" for g in grid))
    print("  " + "  ".join(f"{log_loss(y_val, blend(pf_val, pb_val, g)):.3f}"
                           for g in grid))
    print(f"\nchosen forest weight: {w:.1f}  ({1 - w:.1f} booster)")
    p_bl_te = blend(pf_te, pb_te, w)
    print(f"\n{'blend on test':<22}{accuracy(y_te, p_bl_te):>10.4f}"
          f"{log_loss(y_te, p_bl_te):>10.4f}{brier_score(y_te, p_bl_te):>9.4f}")
    for base_name, base_p in (("forest", pf_te), ("booster", pb_te)):
        print(f"\nblend vs {base_name}:")
        for name, fn, lower in METRICS:
            d = paired_bootstrap_ci(y_te, p_bl_te, base_p, fn, n_boot=N_BOOT,
                                    rng=np.random.default_rng(SEED))
            print(f"  {name:<9}{d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]  "
                  f"{delta_verdict(d, lower)}")

    # --- calibration -----------------------------------------------------
    rule("CALIBRATION — fitted on validation, reported on test")
    best = max(("forest", pf_val, pf_te), ("booster", pb_val, pb_te),
               key=lambda t: -log_loss(y_val, t[1]))
    target_name, p_val, p_te = best
    print(f"calibrating the model with the better VALIDATION log loss: "
          f"{target_name}")

    platt = PlattCalibrator().fit(p_val, y_val)
    iso = IsotonicCalibrator().fit(p_val, y_val)
    print(f"  platt   : {platt.describe()}")
    print(f"  isotonic: {iso.describe()}")

    variants = (("raw", p_te),
                ("platt", platt.transform(p_te)),
                ("isotonic", iso.transform(p_te)))
    print(f"\n{'variant':<12}{'accuracy':>10}{'log_loss':>10}{'brier':>9}")
    for name, p in variants:
        print(f"{name:<12}{accuracy(y_te, p):>10.4f}{log_loss(y_te, p):>10.4f}"
              f"{brier_score(y_te, p):>9.4f}")

    print("\npaired against the raw model, on the test rows:")
    for name, p in variants[1:]:
        print(f"\n{name}:")
        for mname, fn, lower in METRICS:
            d = paired_bootstrap_ci(y_te, p, p_te, fn, n_boot=N_BOOT,
                                    rng=np.random.default_rng(SEED))
            print(f"  {mname:<9}{d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]  "
                  f"{delta_verdict(d, lower)}")

    rule("RELIABILITY — does the number mean what it says?")
    gaps = {}
    for name, p in variants:
        gaps[name] = show_reliability(f"{target_name}, {name}", y_te, p)
    print("\nworst-gap summary:  " + "   ".join(
        f"{k} {v:.3f}" for k, v in gaps.items()))
    print("\nA calibrator that improves log loss should also shrink the worst")
    print("gap. If log loss moves and the gap does not, the gain came from a")
    print("handful of rows rather than from the probabilities being better.")


if __name__ == "__main__":
    main()
