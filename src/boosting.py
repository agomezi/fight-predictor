"""Gradient-boosted trees from scratch, with monotonic constraints.

TWO IDEAS, and they are separable.

1. BOOSTING. A random forest grows trees independently and averages them, so
   every tree is trying to solve the whole problem and the ensemble's job is
   variance reduction. Boosting instead grows trees in SEQUENCE, each one fitted
   to what the running model is still getting wrong. Later trees spend their
   capacity only on the hard rows. On tabular data this is the standard winner.

   The arithmetic is smaller than its reputation. Working in log-odds space,
   the negative gradient of log loss with respect to the current prediction is
   exactly `y - p`. So "fit a tree to the gradient" means "fit a tree to the
   residual", and the whole loop is:

       F  = log-odds of the base rate
       repeat:  p = sigmoid(F);  fit tree h to (y - p);  F += lr * h(x)

2. MONOTONIC CONSTRAINTS. This is the part that answers a question the data
   cannot. The UFC does not book a lightweight champion against a heavyweight,
   so only 4 fights in 8,658 carry a cut burden above 78 lb. Asked about such a
   matchup an unconstrained model extrapolates, and it was measured doing so in
   the WRONG direction -- favouring the drained heavyweight slightly more at
   170 lb than at 265.

   No amount of fitting invents data that is not there. But we do not need the
   data to know the SIGN: cutting a large amount of weight impairs performance,
   which is physiology, not statistics. A monotonic constraint lets that be
   asserted as structure rather than learned as signal -- the tree is forbidden
   from producing a split that violates the declared direction, so the property
   holds everywhere including where no training row exists.

   This is honest only because it is DECLARED. The constraint is domain
   knowledge injected deliberately, it is named in the model's repr, and the
   aggregate data is at least consistent with it (side A wins 47.6% when it is
   >30 lb further from the limit, against 52.8% when its opponent is). It is
   not a fudge to make one prediction look right; it is a statement that a
   direction is known a priori.

WHY A SEPARATE MODULE. `tree.py`'s `best_split` sweeps a cumsum of POSITIVE
LABEL COUNTS, which is exact for binary classification and meaningless for a
continuous residual. The variance sweep here is its structural sibling -- same
sort, same prefix-sum trick -- but carries sums and sums-of-squares instead.
Keeping it here leaves the classification path, which six suites validate,
untouched.
"""

from __future__ import annotations

import numpy as np

from src.tree import Node

# Guard for the Newton denominator. p(1-p) collapses toward zero once the model
# is confident, and dividing by it unguarded produces enormous leaf values that
# destabilise the next round.
MIN_HESSIAN = 1e-6


def _sigmoid(z):
    """Numerically stable logistic. Splitting on the sign avoids exp overflow.

    exp(710) is inf in float64, and a confident boosted model reaches large
    |z| quickly, so the naive 1/(1+exp(-z)) overflows on real data rather than
    only in theory.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _best_split_variance(X, g, h, feature_indices=None, min_samples_leaf=1,
                         monotone=None, lower=-np.inf, upper=np.inf):
    """Best split by variance reduction on the residual. Sibling of best_split.

    Scores a split by the reduction in squared error of the residual, using the
    same sort-once-and-sweep structure as `tree.best_split`: the column is
    sorted once and a running prefix sum gives every candidate threshold's
    child statistics in O(1), so the cost per feature is the O(n log n) sort.

    `g` is the residual (the negative gradient) and `h` the Hessian weight; the
    leaf value implied by a group is `sum(g) / sum(h)`, and the score of a split
    is the usual sum of squared-error reductions. With `h` all ones this reduces
    to plain variance reduction on `g`.

    MONOTONIC CONSTRAINTS. `monotone` maps a feature index to +1 (prediction
    must not decrease as the feature increases) or -1 (must not increase). A
    candidate split on a constrained feature is REJECTED outright when its two
    child values would violate the declared direction. Rejecting at the split
    is what makes the guarantee structural rather than approximate -- combined
    with the bound propagation in `build_regression_tree`, no path through the
    tree can violate it.

    Returns (feature, threshold, score, left_value, right_value).
    """
    n = len(g)
    if n < 2 * min_samples_leaf:
        return None, None, 0.0, None, None
    if feature_indices is None:
        feature_indices = range(X.shape[1])
    monotone = monotone or {}

    total_g, total_h = float(g.sum()), float(h.sum())
    parent_score = (total_g * total_g) / max(total_h, MIN_HESSIAN)

    best = (None, None, 0.0, None, None)
    for feature in feature_indices:
        column = X[:, feature]
        order = np.argsort(column, kind="mergesort")
        x_sorted = column[order]
        g_cum = np.cumsum(g[order])
        h_cum = np.cumsum(h[order])

        # A cut is legal only between two DISTINCT values; splitting inside a
        # run of equal values would not partition the rows the tree claims.
        distinct = np.flatnonzero(x_sorted[:-1] < x_sorted[1:])
        if distinct.size == 0:
            continue
        legal = distinct[(distinct + 1 >= min_samples_leaf)
                         & (n - distinct - 1 >= min_samples_leaf)]
        if legal.size == 0:
            continue

        gl = g_cum[legal]
        hl = np.maximum(h_cum[legal], MIN_HESSIAN)
        gr = total_g - gl
        hr = np.maximum(total_h - h_cum[legal], MIN_HESSIAN)
        scores = (gl * gl) / hl + (gr * gr) / hr - parent_score

        direction = monotone.get(feature, 0)
        if direction:
            # Left child holds the LOWER feature values. For +1 the left value
            # must not exceed the right; for -1 it must not be below it.
            left_val, right_val = gl / hl, gr / hr
            ok = (left_val <= right_val) if direction > 0 else (left_val >= right_val)
            scores = np.where(ok, scores, -np.inf)

        k = int(np.argmax(scores))
        if scores[k] > best[2]:
            i = int(legal[k])
            threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
            lv = float(np.clip(gl[k] / hl[k], lower, upper))
            rv = float(np.clip(gr[k] / hr[k], lower, upper))
            best = (feature, threshold, float(scores[k]), lv, rv)
    return best


def build_regression_tree(X, g, h, depth=0, max_depth=3, min_samples_split=2,
                          min_samples_leaf=1, monotone=None,
                          lower=-np.inf, upper=np.inf):
    """Grow a regression tree on the residual, respecting monotone constraints.

    Mirrors `tree.build_tree`, with two differences: leaves hold a continuous
    value (`sum(g) / sum(h)`) rather than a class, and each node carries BOUNDS.

    THE BOUNDS ARE THE MONOTONICITY GUARANTEE. Rejecting a violating split at
    one node is not enough on its own -- a later split deeper in the left
    subtree could still produce a value exceeding something in the right
    subtree, breaking the property globally. So when a split on a constrained
    feature is accepted, the children are given tightened bounds around the
    midpoint of the two child values: everything in the lower-feature subtree is
    capped there, everything in the higher-feature subtree floored there. The
    constraint then holds for every pair of paths, not just siblings.
    """
    n_samples = X.shape[0]
    value = float(np.clip(g.sum() / max(h.sum(), MIN_HESSIAN), lower, upper))

    if (depth >= max_depth or n_samples < min_samples_split
            or np.allclose(g, g[0] if n_samples else 0.0)):
        return Node(prediction=value, n_samples=n_samples, proba=value)

    feature, threshold, score, left_val, right_val = _best_split_variance(
        X, g, h, min_samples_leaf=min_samples_leaf, monotone=monotone,
        lower=lower, upper=upper,
    )
    if feature is None or score <= 0.0:
        return Node(prediction=value, n_samples=n_samples, proba=value)

    go_left = X[:, feature] <= threshold
    if not go_left.any() or go_left.all():
        return Node(prediction=value, n_samples=n_samples, proba=value)

    direction = (monotone or {}).get(feature, 0)
    lo_l = hi_l = lo_r = hi_r = None
    if direction:
        mid = (left_val + right_val) / 2.0
        if direction > 0:
            (lo_l, hi_l), (lo_r, hi_r) = (lower, mid), (mid, upper)
        else:
            (lo_l, hi_l), (lo_r, hi_r) = (mid, upper), (lower, mid)
    else:
        (lo_l, hi_l), (lo_r, hi_r) = (lower, upper), (lower, upper)

    left = build_regression_tree(
        X[go_left], g[go_left], h[go_left], depth + 1, max_depth,
        min_samples_split, min_samples_leaf, monotone, lo_l, hi_l)
    right = build_regression_tree(
        X[~go_left], g[~go_left], h[~go_left], depth + 1, max_depth,
        min_samples_split, min_samples_leaf, monotone, lo_r, hi_r)
    return Node(feature=feature, threshold=threshold, left=left, right=right,
                n_samples=n_samples, gain=score)


def _predict_tree(root, X):
    """Vectorless walk, matching tree.predict_proba's traversal convention."""
    out = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        node = root
        while not node.is_leaf():
            node = node.left if row[node.feature] <= node.threshold else node.right
        out[i] = node.prediction
    return out


class GradientBoosting:
    """Gradient-boosted trees for binary classification, with monotone support.

    Deliberately tuned in the opposite direction to the forest: many SHALLOW
    trees with a small learning rate, rather than a few deep ones. Each tree is
    a small correction, so depth 2-4 is normal and depth 12 would overfit hard.

    Unlike a forest, MORE ROUNDS IS NOT FREE -- boosting can and does overfit.
    Use `staged_log_loss` on a validation fold to pick `n_rounds`; the round
    where validation loss stops improving is the answer, and watching that curve
    turn is the most instructive output this class produces.
    """

    def __init__(self, n_rounds=200, learning_rate=0.05, max_depth=3,
                 min_samples_split=20, min_samples_leaf=10, newton=True,
                 monotone_constraints=None, random_state=None):
        """
        Args:
            n_rounds: number of boosting rounds (trees).
            learning_rate: shrinkage applied to each tree's contribution.
            max_depth: depth of each weak learner. 2-4.
            newton: True uses the Newton leaf value sum(g)/sum(p(1-p)), which
                is what XGBoost does and converges faster. False uses the plain
                mean residual, i.e. h = 1 everywhere.
            monotone_constraints: {feature_index: +1 or -1}. See the module
                docstring -- this asserts a direction as structure where the
                data is too thin to learn it.
            random_state: unused today; accepted so the constructor matches
                RandomForest's shape for the evaluation harness.
        """
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.newton = newton
        self.monotone_constraints = dict(monotone_constraints or {})
        self.random_state = random_state
        self.trees_ = []
        self.f0_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Start from the base rate in log-odds. Clipped so an all-one-class
        # training set cannot produce an infinite starting point.
        base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.f0_ = float(np.log(base / (1.0 - base)))
        F = np.full(len(y), self.f0_, dtype=float)

        self.trees_ = []
        for _ in range(self.n_rounds):
            p = _sigmoid(F)
            g = y - p                       # negative gradient of log loss
            h = (p * (1.0 - p)) if self.newton else np.ones_like(p)
            tree = build_regression_tree(
                X, g, np.maximum(h, MIN_HESSIAN), depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                monotone=self.monotone_constraints,
            )
            F += self.learning_rate * _predict_tree(tree, X)
            self.trees_.append(tree)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.f0_, dtype=float)
        for tree in self.trees_:
            F += self.learning_rate * _predict_tree(tree, X)
        return F

    def predict_proba(self, X):
        return _sigmoid(self.decision_function(X))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def staged_log_loss(self, X, y):
        """Log loss after each round. This is how `n_rounds` gets chosen.

        Returns a list of length n_rounds. On training data it falls
        monotonically; on a held-out fold it falls, flattens, then RISES -- and
        the minimum is the round to stop at. Fitting past it is the overfitting
        a forest does not suffer from.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        F = np.full(X.shape[0], self.f0_, dtype=float)
        out = []
        for tree in self.trees_:
            F += self.learning_rate * _predict_tree(tree, X)
            p = np.clip(_sigmoid(F), 1e-15, 1 - 1e-15)
            out.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
        return out
