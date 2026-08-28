"""Probability calibration, and blending two models' probabilities.

WHY CALIBRATION IS A SEPARATE JOB FROM ACCURACY. A model can rank fights
perfectly and still state the wrong numbers: if everything it calls 0.70 wins
55% of the time, accuracy is untouched and log loss is bad. Accuracy asks "was
the hard class right"; calibration asks "did the number mean what it said". Log
loss is the metric that notices, which is why calibration is measured on log
loss and expected to leave accuracy flat.

THE ONE RULE. A calibrator is fitted on model output, so fitting it on the same
rows the model was fitted on learns the model's training-set overconfidence
rather than its real-world overconfidence, and helps nothing. It must be fitted
on a HELD-OUT validation slice and then applied to the test tail. Everything
here takes separate (validation, test) arrays for exactly that reason, and
scripts/experiment_blend_calibrate.py carves the validation fold
chronologically, never randomly.

Two calibrators, deliberately:

    PlattCalibrator     a 1-D logistic on the model's log-odds. Two parameters,
                        so it cannot overfit a validation fold, and it can only
                        rescale -- it will not invent structure.
    IsotonicCalibrator  a monotone step function, non-parametric. Strictly more
                        flexible, and correspondingly happier to overfit a small
                        fold. Wrapped from sklearn the same way compare_sklearn
                        uses it: as an independent implementation to check
                        against rather than the thing being learned.

Platt is hand-written because "what is Platt scaling" is a question worth being
able to answer, and because two-parameter Newton is short enough to read.
"""

from __future__ import annotations

import numpy as np

# Probabilities are clipped before the logit so a pure-leaf 0.0 or 1.0 does not
# become an infinity. Same reasoning as evaluate.LOG_LOSS_EPS, looser bound
# because this one feeds an optimiser rather than a report.
CLIP = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


class PlattCalibrator:
    """Fit sigmoid(a * logit(p) + b) by Newton's method. Two parameters.

    Reading the parameters is the useful part:
        a < 1  the model is OVERCONFIDENT -- its probabilities get pulled
               toward 0.5. This is the usual finding for trees, whose leaves
               report the purity of a handful of training rows.
        a > 1  underconfident, probabilities pushed outward.
        b != 0 a base-rate shift, i.e. the model's average probability is off.

    Newton rather than gradient descent because with two parameters the 2x2
    Hessian is trivial and it converges in a handful of iterations, with no
    learning rate to pick.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-10,
                 max_slope: float = 20.0):
        self.max_iter = max_iter
        self.tol = tol
        # A separable validation fold drives the slope to infinity: if every
        # high-p row won and every low-p row lost, the likelihood is maximised
        # by an infinitely sharp step. That is the optimiser being right and the
        # fold being too small, and the result would blow up log loss on the
        # first test row it gets wrong. Capping the slope and reporting
        # non-convergence is honest; silently shipping a=44 is not.
        self.max_slope = max_slope
        self.a = 1.0
        self.b = 0.0
        self.n_iter_ = 0
        self.converged_ = False

    def fit(self, p_val, y_val):
        z = _logit(p_val)
        y = np.asarray(y_val, dtype=float)
        a, b = 1.0, 0.0
        for i in range(self.max_iter):
            q = _sigmoid(a * z + b)
            # Gradient of the negative log likelihood.
            resid = q - y
            g = np.array([np.sum(resid * z), np.sum(resid)])
            # Hessian: w = q(1-q) is the Bernoulli variance at each point.
            w = q * (1.0 - q)
            h = np.array([[np.sum(w * z * z), np.sum(w * z)],
                          [np.sum(w * z),     np.sum(w)]])
            # A degenerate Hessian means the validation slice carries no
            # information to fit on (one class, or constant p). Leave the
            # calibrator as the identity rather than inventing a fit.
            try:
                step = np.linalg.solve(h + 1e-12 * np.eye(2), g)
            except np.linalg.LinAlgError:
                break
            a, b = a - step[0], b - step[1]
            self.n_iter_ = i + 1
            if np.max(np.abs(step)) < self.tol:
                self.converged_ = True
                break
            if abs(a) > self.max_slope:
                # Diverging. Fall back to the identity rather than a wild
                # rescale fitted to a fold that cannot support one.
                a, b = 1.0, 0.0
                break
        self.a, self.b = float(a), float(b)
        return self

    def transform(self, p):
        return _sigmoid(self.a * _logit(p) + self.b)

    def describe(self) -> str:
        if not self.converged_ and self.a == 1.0 and self.b == 0.0:
            return (f"DIVERGED after {self.n_iter_} steps -- fell back to the "
                    "identity (validation fold is separable)")
        verdict = ("overconfident" if self.a < 0.98 else
                   "underconfident" if self.a > 1.02 else "well scaled")
        note = "" if self.converged_ else "  [NOT CONVERGED -- treat with care]"
        return (f"a={self.a:.4f} b={self.b:+.4f} "
                f"({self.n_iter_} Newton steps, model reads {verdict}){note}")


class IsotonicCalibrator:
    """Monotone step-function calibration, via sklearn.

    Strictly more flexible than Platt: it can correct a non-monotone-in-
    magnitude miscalibration that a two-parameter rescale cannot. The cost is
    that it has effectively as many parameters as the validation fold has
    distinct probabilities, so on a small fold it will happily fit noise and
    look better in-fold while being worse out-of-fold. That is precisely why the
    experiment reports OUT-of-fold numbers and why both are measured rather than
    assuming the flexible one wins.
    """

    def __init__(self):
        self.model = None

    def fit(self, p_val, y_val):
        from sklearn.isotonic import IsotonicRegression

        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                        y_max=1.0)
        self.model.fit(np.asarray(p_val, dtype=float),
                       np.asarray(y_val, dtype=float))
        return self

    def transform(self, p):
        if self.model is None:
            raise RuntimeError("fit before transform")
        return np.clip(self.model.predict(np.asarray(p, dtype=float)),
                       CLIP, 1.0 - CLIP)

    def describe(self) -> str:
        thresholds = getattr(self.model, "X_thresholds_", None)
        n = 0 if thresholds is None else len(thresholds)
        return f"{n} monotone segments"


def blend(p_a, p_b, weight: float = 0.5):
    """Weighted average of two models' probabilities. weight is p_a's share.

    Averaging probabilities rather than picking a winner works when two models
    make DIFFERENT mistakes: the errors partially cancel while the shared signal
    does not. Here the forest and the booster are genuinely different -- the
    booster has a tighter fold spread and better log loss, the forest better
    accuracy -- which is the condition under which a blend can beat both.

    It can also beat neither, which is why the weight is chosen on a validation
    fold and the result is reported either way.
    """
    return weight * np.asarray(p_a, dtype=float) + (1.0 - weight) * np.asarray(
        p_b, dtype=float)


def best_blend_weight(p_a_val, p_b_val, y_val, metric_fn, grid=None,
                      lower_is_better: bool = True) -> float:
    """Pick the blend weight on VALIDATION output, never on the test tail.

    A coarse grid is deliberate. The optimum is flat near the top -- 0.4 and 0.5
    are usually indistinguishable -- so a finer search buys nothing except a
    weight fitted more tightly to one fold's noise.
    """
    grid = np.linspace(0.0, 1.0, 11) if grid is None else np.asarray(grid)
    scores = [metric_fn(y_val, blend(p_a_val, p_b_val, w)) for w in grid]
    idx = int(np.argmin(scores) if lower_is_better else np.argmax(scores))
    return float(grid[idx])
