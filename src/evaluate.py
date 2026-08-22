"""Evaluation harness: proper scoring rules, bootstrap intervals, walk-forward.

Why this exists before any feature work.

v1 measured everything with accuracy on one chronological tail of 1,512 fights,
and that turned out to be too blunt to answer its own questions. The forest's
edge over a single tree read +0.002 against one baseline and +0.042 against
another; telling those apart needed a hand-waved "+/- 2.5 points" rule of thumb
rather than a measured interval. The README's other open question -- a
validation-selected max_depth losing to a hardcoded one by 1.1 points -- is
unanswerable for the same reason.

Every feature added from here (Elo, rolling as-of-fight aggregates) has exactly
that problem: the number will move, and without a ruler there is no way to know
whether the feature helped or the noise shifted. So the ruler comes first.

Three things it adds over `score()`:

  1. PROPER SCORING RULES. Accuracy asks only whether the hard class was right,
     so a model correct at 0.51 and one correct at 0.95 are identical to it.
     Log loss and the Brier score read the probability itself. They are the
     primary metrics here; accuracy is reported alongside for continuity with
     the v1 numbers.

  2. BOOTSTRAP INTERVALS. Resampling the test set gives a measured error bar,
     which is what lets a delta be called real or not.

  3. WALK-FORWARD FOLDS. One tail is hostage to which fights landed in it.
     Sliding the cut across the timeline shows whether a result holds across
     eras -- and this dataset spans 1994-2026, over which data quality changes
     enough to have already flipped the sign of the OOB gap.

Everything here takes probabilities, not labels: `p` is always P(y=1), from
DecisionTree.predict_proba or RandomForest.predict_proba.
"""

from __future__ import annotations

import numpy as np

# Clip applied before taking a logarithm. A tree leaf can be pure, so
# predict_proba legitimately returns exactly 0.0 or 1.0, and a single
# confidently-wrong row would otherwise make log loss infinite.
#
# Worth understanding rather than accepting: the clip does not merely avoid a
# crash, it sets the price of being confidently wrong. At 1e-15 one such row
# contributes -log(1e-15) ~ 34.5 to the sum. For a model with many pure leaves
# -- the unpruned tree especially -- log loss therefore becomes close to "how
# many rows did it get confidently wrong, times 34.5", and the magnitude is as
# much a property of this constant as of the model. Comparisons between models
# are still valid because the constant is shared; absolute values are not
# meaningful on their own.
LOG_LOSS_EPS = 1e-15


# ---------------------------------------------------------------------------
# Metrics. All take (y_true, p) where p is P(y=1); lower is better except
# accuracy.
# ---------------------------------------------------------------------------
def accuracy(y_true, p, threshold=0.5):
    """Fraction of rows whose hard class is right.

    The threshold is STRICT (`p > threshold`), matching _majority_class in
    tree.py, which breaks a 50/50 leaf toward class 0. Using `>=` would flip
    every tied leaf to class 1 and shift the number.
    """
    y_true = np.asarray(y_true)
    return float(np.mean((np.asarray(p) > threshold).astype(int) == y_true))


def log_loss(y_true, p, eps=LOG_LOSS_EPS):
    """Mean negative log likelihood, in nats. Lower is better.

    -mean( y*log(p) + (1-y)*log(1-p) )

    The strictly proper scoring rule for probabilities: it is minimised only by
    reporting your true belief, so it punishes overconfidence hard. A perfectly
    calibrated coin-flip model scores log(2) = 0.6931, which is the number to
    beat and the one to compare against -- not zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def brier_score(y_true, p):
    """Mean squared error of the probability. Lower is better.

    mean( (p - y)^2 )

    Also strictly proper, but bounded in [0, 1] and far less punishing of
    confident errors than log loss -- no clip is needed and no single row can
    dominate. Reporting both is deliberate: if log loss ranks two models
    differently from Brier, the difference is concentrated in a few
    confidently-wrong rows, which is itself worth knowing.

    An always-0.5 model scores 0.25.
    """
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y_true) ** 2))


# Registry so the bootstrap and the fold loop can be pointed at any metric.
# (name, fn, lower_is_better)
METRICS = (
    ("accuracy", accuracy, False),
    ("log_loss", log_loss, True),
    ("brier", brier_score, True),
)


def reference_scores(y_true):
    """What a model that has learned nothing scores. The real baseline row.

    Returns the three metrics for an always-predict-the-base-rate model, which
    is the honest zero point: on this symmetrised data the base rate is ~0.50,
    so log loss ~0.693 and Brier ~0.250. Any model that cannot beat these has
    added nothing, and a model can beat accuracy while LOSING on log loss --
    which is exactly what an overconfident tree does.
    """
    y_true = np.asarray(y_true)
    base = float(np.mean(y_true)) if len(y_true) else 0.5
    p = np.full(len(y_true), base, dtype=float)
    return {name: fn(y_true, p) for name, fn, _ in METRICS}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, p, metric_fn, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap confidence interval for a metric.

    THE IDEA. We have one test set and one number, and we want to know how much
    that number would move if we had drawn a different test set from the same
    population. We cannot redraw -- but resampling the rows we do have, with
    replacement, approximates it. Recompute the metric on each resample and the
    spread of those values estimates the sampling distribution of the metric.
    The middle (1 - alpha) of them is the interval.

    This is what replaces the "+/- 2.5 points" rule of thumb, and it is what
    decides whether any future feature's delta clears its own error bar.

    Args:
        y_true: 1-D array of true labels, length n.
        p: 1-D array of P(y=1), same length and same row order as y_true.
        metric_fn: callable (y_true, p) -> float. Any of the metrics above.
        n_boot: number of resamples. 2000 is ample for a 95% percentile
            interval; a couple of hundred is visibly lumpy, and 100k is wasted
            work.
        alpha: 0.05 gives a 95% interval.
        rng: numpy Generator, for reproducibility. None means make a fresh one.

    Returns:
        (point, lo, hi) as floats. `point` is the metric on the ORIGINAL data,
        not the mean of the resamples -- those differ by the bootstrap's bias
        and the observed value is the estimate we are actually reporting.

    Watch out for:
      * Resample INDICES, then index both arrays with them. Resampling y_true
        and p separately silently destroys the pairing and produces a garbage
        interval that looks perfectly reasonable.
      * Draw n indices, where n = len(y_true). Drawing fewer inflates the
        interval; that is a different (and wrong) estimator.
      * A resample can come out class-degenerate -- every row label 1, say.
        accuracy, log_loss and brier all still evaluate on such a sample, so
        nothing needs special-casing here, but a metric requiring both classes
        (ROC AUC, were it added later) would raise or return NaN and would need
        handling.
      * Percentile bounds come from the resample distribution via
        np.percentile at alpha/2 and 1 - alpha/2.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y_true)
    if rng is None:
        rng = np.random.default_rng()

    point = metric_fn(y_true, p)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = metric_fn(y_true[idx], p[idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 *(1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def ci_half_width(lo, hi):
    """Half the interval width, for reporting a metric as `point +/- x`."""
    return (hi - lo) / 2.0


def paired_bootstrap_ci(y_true, p_a, p_b, metric_fn, n_boot=2000, alpha=0.05,
                        rng=None):
    """Confidence interval for metric(A) - metric(B) on the SAME rows.

    THE POINT, and why this is not just two intervals compared by eye. Both
    models are scored on one test set, so their errors are correlated: a test
    set that happens to be full of upsets makes BOTH models look bad. Comparing
    two independent intervals throws that shared noise away and is far too
    conservative -- overlapping intervals routinely hide a real difference.

    Resampling the row indices ONCE per iteration and scoring both models on
    the same resample cancels the shared component, leaving only the part that
    actually distinguishes them. The resulting interval is materially tighter,
    and it is the correct test for "did this feature pay?".

    Args:
        y_true: labels, length n.
        p_a, p_b: each model's P(y=1), same length and row order as y_true.
        metric_fn: (y_true, p) -> float.
        n_boot, alpha, rng: as bootstrap_ci.

    Returns:
        (delta, lo, hi) for metric(A) - metric(B). An interval excluding zero
        is evidence of a real difference; one containing zero is not evidence
        of equivalence, only of insufficient resolution.
    """
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if not (len(y_true) == len(p_a) == len(p_b)):
        raise ValueError("y_true, p_a and p_b must be the same length and order")
    if rng is None:
        rng = np.random.default_rng()

    n = len(y_true)
    delta = metric_fn(y_true, p_a) - metric_fn(y_true, p_b)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        # One index draw, both models. Drawing separately would reintroduce
        # exactly the noise this function exists to cancel.
        boot[i] = metric_fn(yb, p_a[idx]) - metric_fn(yb, p_b[idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(delta), float(lo), float(hi)


def delta_verdict(delta_ci, lower_is_better) -> str:
    """Turn a paired interval into the sentence the discipline rule asks for.

    Deliberately blunt, and only three outcomes. "inside noise" is not a
    finding of equivalence -- it means this test set cannot resolve the
    difference, which is usually the honest answer for a two-point move on
    1,500 fights.
    """
    delta, lo, hi = delta_ci
    if lo <= 0.0 <= hi:
        return "inside noise"
    improved = (delta < 0) if lower_is_better else (delta > 0)
    return "REAL improvement" if improved else "REAL regression"


def deltas_overlap(ci_a, ci_b):
    """True if two (point, lo, hi) intervals overlap.

    A blunt instrument, and deliberately so: non-overlapping intervals are
    strong evidence of a real difference, but OVERLAPPING intervals are not
    evidence of no difference -- the correct test for that is a bootstrap of
    the paired DIFFERENCE on the same rows, which is tighter because it cancels
    the shared test-set noise. Use this for a first look, not for a verdict.
    """
    _, lo_a, hi_a = ci_a
    _, lo_b, hi_b = ci_b
    return not (hi_a < lo_b or hi_b < lo_a)


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
def walk_forward_folds(dates, n_folds=8, min_train_frac=0.5):
    """Yield (train_idx, test_idx) pairs that slide a chronological cutoff.

    THE IDEA. One train/test cut gives one number, and that number depends on
    which fights happened to land in the tail. Sliding the cut forward produces
    several test windows, so a result can be checked for whether it holds across
    eras instead of in one window. With this dataset that matters more than
    usual: it spans 1994-2026, and the change in data quality across that span
    already flipped the sign of the OOB-minus-test gap.

    Args:
        dates: 1-D array-like of event dates, one per row, ALREADY SORTED
            ascending. The caller sorts; this function does not, because
            silently reordering rows would break the caller's alignment with X
            and y. Assert the ordering rather than fixing it.
        n_folds: how many (train, test) pairs to produce.
        min_train_frac: the first fold trains on at least this fraction of the
            rows, so the earliest fold is not fitted on a handful of 1994
            fights.

    Yields:
        (train_idx, test_idx): integer index arrays into `dates`. Every index in
        test_idx must correspond to a date strictly later than every date in
        train_idx.

    Watch out for:
      * CUT ON DATE BOUNDARIES, NOT ROW INDICES. This is the same trap
        chronological_split already handles: a cut placed at a row index can
        land in the middle of one event's card, putting some of a single night's
        fights in train and the rest in test. Move the boundary to a date edge
        so no event date appears on both sides.
      * Test must never precede train. There is no shuffling anywhere in this
        function; if you find yourself wanting to shuffle, the fold design is
        wrong.
      * EXPANDING vs SLIDING window is a real choice and it is yours to make.
        Expanding (train on everything before T) uses more data each fold but
        mixes eras, so late folds train on a 30-year mixture. Sliding (a
        fixed-width window ending at T) keeps folds comparable to each other but
        trains on less. Pick one, say which in the docstring, and know why --
        this is a question an interviewer will ask.
      * The last fold should reach the final date, or the most recent fights
        are silently never tested.
      * A fold whose test window comes out empty (possible when many rows share
        few distinct dates) must be skipped or raise, not yielded.

    This implementation uses an EXPANDING window: each fold trains on every
    fight before its cutoff (`dates < c_start`), so the training set grows every
    fold. Chosen over a sliding window because the dataset is not large and the
    extra history outweighs the era-mixing cost — see the trade-off above.
    """
    dates = np.asarray(dates)
    if not np.all(dates[:-1] <= dates[1:]):
        raise ValueError("dates must be sorted ascending before folding")
    unique_dates = np.unique(dates)
    final = unique_dates[-1]

    # Skip the first min_train_frac of distinct dates, then place n_folds+1
    # evenly-spaced cutoffs from there to the final date. Consecutive cutoffs
    # bound each fold's test window; the last one is the final date so the most
    # recent fights are tested.
    start = int(np.ceil(min_train_frac * len(unique_dates)))
    positions = np.unique(np.linspace(start, len(unique_dates) - 1, n_folds + 1).astype(int))
    cutoffs = unique_dates[positions]

    for c_start, c_end in zip(cutoffs[:-1], cutoffs[1:]):
        train_idx = np.flatnonzero(dates < c_start)
        if c_end == final:
            test_idx = np.flatnonzero(dates >= c_start)          # last fold: to the end
        else:
            test_idx = np.flatnonzero((dates >= c_start) & (dates < c_end))
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx 


def run_walk_forward(fit_fn, X, y, dates, n_folds=8, min_train_frac=0.5):
    """Fit and score a model across every walk-forward fold.

    Args:
        fit_fn: callable (X_train, y_train) -> fitted model exposing
            predict_proba(X). A lambda closing over hyperparameters is the
            intended use, so the same driver can score a tree and a forest.
        X: 2-D feature matrix, row-aligned with `dates`.
        y: 1-D label array, row-aligned with `dates`.
        dates: 1-D date array, sorted ascending (see walk_forward_folds).
        n_folds, min_train_frac: passed through.

    Returns:
        list of per-fold dicts: fold index, train/test sizes, the test window's
        first and last date, and every metric in METRICS.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    dates = np.asarray(dates)

    rows = []
    for i, (tr, te) in enumerate(walk_forward_folds(dates, n_folds, min_train_frac)):
        model = fit_fn(X[tr], y[tr])
        p = np.asarray(model.predict_proba(X[te]), dtype=float)
        row = {
            "fold": i,
            "n_train": len(tr),
            "n_test": len(te),
            "test_start": dates[te].min(),
            "test_end": dates[te].max(),
        }
        row.update({name: fn(y[te], p) for name, fn, _ in METRICS})
        rows.append(row)
    return rows


def summarise_folds(rows):
    """Mean and standard deviation of each metric across folds.

    The spread is the point of the exercise. A metric whose fold-to-fold
    standard deviation swamps the difference between two models is a metric
    that cannot distinguish them, however clean the single-split number looked.
    """
    out = {}
    for name, _, _ in METRICS:
        vals = np.array([r[name] for r in rows], dtype=float)
        out[name] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Calibration, as text (the repo has no plotting dependency, and a table is
# easier to paste into a README than an image anyway)
# ---------------------------------------------------------------------------
def reliability_table(y_true, p, n_bins=10):
    """Bucket predictions and compare predicted vs observed win rate.

    A well calibrated model that says 0.70 should win about 70% of those
    fights. This is where a model with decent accuracy and bad probabilities
    gets caught, and it is the diagnostic log loss summarises into one number.

    Returns a list of dicts (bin_lo, bin_hi, n, mean_predicted, observed).
    Empty bins are omitted -- a single tree has only as many distinct
    probabilities as leaves, so most bins WILL be empty, and that coarseness is
    itself the finding.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Last bin closes on the right so p == 1.0 is not dropped.
        in_bin = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(in_bin.sum())
        if not n:
            continue
        out.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "mean_predicted": float(p[in_bin].mean()),
            "observed": float(y_true[in_bin].mean()),
        })
    return out
