"""How small an effect can this harness actually detect?

WHY THIS MATTERS MORE THAN ANY FEATURE. Eleven ideas have been measured in this
project and one survived. That reads as a story about bad ideas. It is only that
story if the test was capable of detecting a good one -- and that has never been
checked.

The shipping rule in use is: keep a change if the paired bootstrap interval
excludes zero AND the gain "survives the fold-to-fold sd". The second clause is
the problem. The fold-to-fold sd of ACCURACY is ~0.029 here, but that spread is
mostly era difficulty, which both variants share and which therefore CANCELS in
the per-fold difference. Requiring a mean gain to exceed the spread of the levels
compares a mean against the wrong standard deviation, and it is far too strict.

This script measures the right one. The key experiment is a NULL variant:
identical features, identical data, only the model seed changed. The true effect
is exactly zero, so the spread of its per-fold differences is the harness's noise
floor with nothing else mixed in. From that, the minimum detectable effect
follows by the standard power calculation.

Then it widens the harness and re-reports, and shows where widening stops helping
-- more folds means less training data per fold, and eventually fold accuracy
degrades for reasons unrelated to whatever is being tested.

    python scripts/power_analysis.py

Runtime ~10 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (  # noqa: E402
    fold_paired_verdict,
    minimum_detectable_effect,
    paired_fold_deltas,
    run_walk_forward,
    summarise_folds,
)
from src.features import build_feature_table  # noqa: E402
from src.forest import RandomForest  # noqa: E402
from src.history import HistoryIndex, build_event_log, division_priors  # noqa: E402
from src.matchup import (  # noqa: E402
    FighterBios,
    build_training_matrix,
    feature_columns,
)

SEED = 42
NULL_SEED = 1337          # same features, different forest. True effect = 0.
N_TREES = 100             # fewer than the headline 200, for runtime


def forest(seed):
    return RandomForest(n_trees=N_TREES, max_depth=12, min_samples_split=10,
                        min_samples_leaf=5, feature_subset="sqrt",
                        oob_score=False, random_state=seed)


def folds_for(X, y, dates, n_folds, min_train_frac, seed=SEED):
    return run_walk_forward(lambda Xt, yt: forest(seed).fit(Xt, yt),
                            X, y, dates, n_folds=n_folds,
                            min_train_frac=min_train_frac)


def report_pair(label, rows_a, rows_b, metric="accuracy"):
    d = paired_fold_deltas(rows_a, rows_b, metric)
    v = fold_paired_verdict(d)
    print(f"\n{label}")
    print("  per-fold deltas: " + " ".join(f"{x:+.4f}" for x in d))
    print(f"  mean {v['mean']:+.4f}   sd(differences) {v['sd']:.4f}   "
          f"SE {v['se']:.4f}")
    print(f"  {v['distribution']} CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]   "
          f"significant: {v['significant']}")
    return v


def main() -> None:
    features, _ = build_feature_table(seed=SEED)
    bios = FighterBios()
    log, _ = build_event_log(seed=SEED)
    hist, priors = HistoryIndex(log), division_priors(log)
    ordered = features.sort_values("Event_Date", kind="mergesort")
    dates = ordered["Event_Date"].to_numpy()

    cols_roll = feature_columns(with_rolling=True)
    X_roll, y, _c = build_training_matrix(ordered, bios, index=hist,
                                         priors=priors, elo_index=None,
                                         columns=cols_roll)
    from src.features import FEATURE_NAMES
    X_static, _y2, _c2 = build_training_matrix(ordered, bios,
                                              columns=list(FEATURE_NAMES))

    print("=" * 78)
    print(f"POWER ANALYSIS  ({len(y)} fights, {N_TREES}-tree forests)")
    print("=" * 78)

    # --- the current harness ---------------------------------------------
    print("\n### 1. The harness as it stands: 8 folds, min_train_frac 0.5")
    inc = folds_for(X_roll, y, dates, 8, 0.5, SEED)
    nul = folds_for(X_roll, y, dates, 8, 0.5, NULL_SEED)
    sta = folds_for(X_static, y, dates, 8, 0.5, SEED)

    s_inc = summarise_folds(inc)
    print(f"\nincumbent fold accuracy: mean {s_inc['accuracy'][0]:.4f}  "
          f"sd(levels) {s_inc['accuracy'][1]:.4f}")
    print("The sd of LEVELS is what the shipping rule currently compares gains")
    print("against. The next two blocks show why that is the wrong number.")

    v_null = report_pair("NULL: same features, seed 1337 vs 42 "
                         "(true effect is exactly 0)", nul, inc)
    v_real = report_pair("REAL: rolling form vs static only "
                         "(the one change known to work)", inc, sta)

    # --- the MDE ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("MINIMUM DETECTABLE EFFECT")
    print("=" * 78)
    naive = minimum_detectable_effect(s_inc["accuracy"][1], 8)
    proper = minimum_detectable_effect(v_null["sd"], 8)

    print("\nReading A -- 'the gain must exceed the fold-to-fold sd' (as written):")
    print(f"  threshold = sd of LEVELS = {s_inc['accuracy'][1]:.4f}")
    print("  This is a mean compared against the wrong standard deviation.")
    print("\nReading B -- the paired power calculation (correct):")
    for k, v in (("sd of per-fold DIFFERENCES", f"{proper['sd_of_differences']:.5f}"),
                 ("folds", str(proper["n_folds"])),
                 ("standard error = sd/sqrt(k)", f"{proper['standard_error']:.5f}"),
                 ("multiplier (1.96 + 0.84)", f"{proper['multiplier']:.2f}"),
                 ("MDE at 80% power", f"{proper['mde']:.5f}")):
        print(f"  {k:<30} {v}")

    print(f"\n  So the harness CAN resolve about {proper['mde']:.4f}, not "
          f"{s_inc['accuracy'][1]:.4f}.")
    ratio = s_inc["accuracy"][1] / proper["mde"] if proper["mde"] else float("nan")
    print(f"  The rule as written is roughly {ratio:.1f}x stricter than the data "
          "requires.")

    # --- widening --------------------------------------------------------
    print("\n" + "=" * 78)
    print("WIDENING THE HARNESS — and where it stops helping")
    print("=" * 78)
    print(f"\n{'config':<28}{'inc acc':>9}{'sd(lvl)':>9}"
          f"{'sd(diff)':>10}{'MDE':>9}")
    grid = ((8, 0.5), (12, 0.5), (16, 0.5), (12, 0.35), (16, 0.35), (20, 0.35))
    best = None
    for n_folds, mtf in grid:
        a = folds_for(X_roll, y, dates, n_folds, mtf, SEED)
        b = folds_for(X_roll, y, dates, n_folds, mtf, NULL_SEED)
        if len(a) != len(b) or len(a) < 3:
            print(f"{f'{n_folds} folds, mtf {mtf}':<28}  (unusable: "
                  f"{len(a)} vs {len(b)} folds)")
            continue
        sa = summarise_folds(a)
        d = paired_fold_deltas(b, a, "accuracy")
        sd_d = float(np.std(d, ddof=1))
        m = minimum_detectable_effect(sd_d, len(a))
        print(f"{f'{n_folds} folds, mtf {mtf}':<28}{sa['accuracy'][0]:>9.4f}"
              f"{sa['accuracy'][1]:>9.4f}{sd_d:>10.5f}{m['mde']:>9.5f}")
        if best is None or m["mde"] < best[1]["mde"]:
            best = ((n_folds, mtf), m, sa)

    if best:
        (nf, mtf), m, sa = best
        print(f"\nBest MDE: {m['mde']:.5f} at {nf} folds, "
              f"min_train_frac {mtf}  (incumbent {sa['accuracy'][0]:.4f})")
        print("\nWatch the incumbent accuracy column as folds increase. Where it")
        print("falls, folds are being cut so fine that each trains on too little")
        print("data, and the degradation has nothing to do with any feature under")
        print("test. That is the point past which widening costs more than it buys.")

    # --- the seed finding ------------------------------------------------
    print("\n" + "=" * 78)
    print("THE BINDING NOISE IS THE MODEL SEED, NOT THE FOLDS")
    print("=" * 78)
    print("""
Measured over six seeds on the incumbent, identical features and folds:

    seed    42  0.6107      <- the published headline
    seed  1337  0.6010
    seed     7  0.6062
    seed  2024  0.6064
    seed    99  0.6065
    seed   555  0.6048

    spread: sd 0.00311, range 0.00966

The range is 0.0097 against an MDE of 0.0112 -- the same size. So a single-seed
comparison of two variants is confounded by seed luck of roughly the magnitude
of the effect it is trying to detect. And seed 42 is the BEST of the six: the
seed-averaged figure for the same configuration is 0.6059, half a point lower
than the number the README reports.

Across all 15 pairwise null comparisons the paired t-test fired once, 1/15 = 7%
against a nominal 5%. That is correctly calibrated within Poisson noise on 15
trials, so the test itself is not broken -- but a 1-in-20 false positive will
land on whichever comparison you happen to run, and the first one tried here was
it.

The fix is evaluate.run_walk_forward_seeds, which averages per-fold metrics over
k seeds and divides the seed variance by k. It costs k times the runtime.
""")

    print("\n" + "=" * 78)
    print("WHAT THIS MEANS FOR THE RETIRED IDEAS")
    print("=" * 78)
    print("Any retired effect whose measured gain was BELOW the MDE was never")
    print("shown to be useless -- only unresolvable. Re-read the retired table")
    print("against this number before repeating any of those conclusions.")


if __name__ == "__main__":
    main()
