"""Random forest built on top of the from-scratch decision tree.

A forest is just many trees whose errors are as uncorrelated as possible,
averaged together. Averaging k independent estimators with variance v gives
variance v/k — but only if they're independent. A single tree is high-variance
(change a few training rows and the whole structure moves), so the entire game
is manufacturing independence between trees. Two sources of randomness do it:

  1. Bootstrap sampling (bagging) — each tree trains on n rows drawn WITH
     replacement from the n training rows. About 63.2% of the rows appear at
     least once, the rest are duplicated; the ~36.8% left out are that tree's
     out-of-bag (OOB) rows, which give a free validation estimate.
  2. Random feature subsets per node — already implemented in build_tree via
     `feature_subset`. Without it, one dominant feature would be the root split
     of nearly every tree and the trees would look nearly identical despite
     different bootstrap samples.

Prediction is a majority vote across trees.

Bias/variance framing: each tree is grown deep (low bias, high variance) and
the ensemble kills the variance. That's why forests use LESS pre-pruning than
a single tuned tree — pruning each tree hard would trade the variance the
ensemble already handles for bias it cannot undo.
"""

from __future__ import annotations

import numpy as np

from .tree import DecisionTree


def bootstrap_indices(n_samples, rng):
    """Draw one bootstrap sample of row indices.

    Args:
        n_samples: number of training rows, n.
        rng: numpy Generator (np.random.default_rng(...)).

    Returns:
        (in_bag, out_of_bag):
            in_bag: int array of length n — n indices drawn from [0, n) WITH
                replacement. Length n, not the number of distinct rows: the
                tree must see the same sample size, duplicates included, or it
                isn't the same learning problem.
            out_of_bag: int array of the indices in [0, n) that do NOT appear
                in in_bag, sorted ascending. Typically ~36.8% of n, but it is a
                random quantity — do not hardcode that fraction, and handle the
                (rare, small-n) case where it comes out empty.
    """
    in_bag = rng.integers(0, n_samples, size=n_samples)
    seen = np.zeros(n_samples, dtype=bool)
    seen[in_bag] = True
    out_of_bag = np.flatnonzero(~seen)
    return in_bag, out_of_bag


def majority_vote(tree_predictions):
    """Combine per-tree predictions into one prediction per sample.

    Args:
        tree_predictions: 2-D int array, shape (n_trees, n_samples). Row t is
            tree t's predictions for every sample. Entries are in {0, 1}, and
            may contain -1 for "this tree did not vote on this sample" — the
            OOB scorer uses that to mark samples a tree was trained on.

    Returns:
        1-D int array of length n_samples: the class most trees voted for.
        Ties break toward 0 (matching _majority_class in tree.py, so the forest
        and a single tree agree on the convention). A column where every tree
        abstained (-1) should return -1 so the caller can exclude it.
    """
    votes = np.asarray(tree_predictions, dtype=int)
    ones = (votes == 1).sum(axis=0)
    zeros = (votes == 0).sum(axis=0)
    result = np.where(ones > zeros, 1, 0)
    result[(ones + zeros) == 0] = -1
    return result


class RandomForest:
    """Bagged ensemble of DecisionTrees, sklearn-style API.

    Attributes set by fit():
        trees_: list of fitted DecisionTree
        oob_indices_: list of out-of-bag index arrays, parallel to trees_
        oob_score_: OOB accuracy, or None if oob_score=False
    """

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, feature_subset="sqrt", oob_score=True,
                 random_state=None):
        """
        Args:
            n_trees: number of trees to grow. Accuracy rises steeply for the
                first few dozen then flattens; more trees never overfit, they
                only cost time.
            max_depth / min_samples_split / min_samples_leaf: passed through to
                each tree. Defaults are deliberately unpruned — see the module
                docstring.
            feature_subset: features considered per node. "sqrt" -> round
                sqrt(n_features) (the standard choice for classification), an
                int for an explicit count, or None for all features (which
                makes this plain bagging, no feature randomness — worth running
                once to see the accuracy difference).
            oob_score: compute the out-of-bag accuracy during fit.
            random_state: seed for reproducibility. One Generator is created in
                fit and used for every tree, so the whole forest is
                deterministic given this seed.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_subset = feature_subset
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees_ = []
        self.oob_indices_ = []
        self.oob_score_ = None

    def _resolve_feature_subset(self, n_features):
        """Turn the feature_subset setting into an int (or None) for build_tree."""
        if self.feature_subset is None:
            return None
        if self.feature_subset == "sqrt":
            return max(1, int(round(np.sqrt(n_features))))
        if self.feature_subset == "log2":
            return max(1, int(round(np.log2(n_features))))
        return int(self.feature_subset)

    def fit(self, X, y):
        """Grow n_trees trees, each on its own bootstrap sample.

        Loop shape:
            for each tree:
                in_bag, oob = bootstrap_indices(n, rng)
                fit a DecisionTree on X[in_bag], y[in_bag]
                stash the tree and its oob indices

        Each tree needs its own seed derived from the forest rng, so that the
        per-node feature sampling differs between trees but the forest as a
        whole stays reproducible.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        max_features = self._resolve_feature_subset(n_features)

        self.trees_ = []
        self.oob_indices_ = []
        for _ in range(self.n_trees):
            in_bag, oob = bootstrap_indices(n_samples, rng)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                feature_subset=max_features,
                random_state=int(rng.integers(0, 2**32)),
            )
            tree.fit(X[in_bag], y[in_bag])
            self.trees_.append(tree)
            self.oob_indices_.append(oob)

        self.oob_score_ = self._compute_oob_score(X, y) if self.oob_score else None
        return self

    def predict(self, X):
        """Majority vote across all trees."""
        X = np.asarray(X, dtype=float)
        votes = np.array([t.predict(X) for t in self.trees_], dtype=int)
        return majority_vote(votes)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _compute_oob_score(self, X, y):
        """Accuracy over samples, each predicted only by trees that never saw it.

        Builds an (n_trees, n_samples) vote matrix filled with -1, writes each
        tree's predictions only into ITS out-of-bag columns, then majority-votes
        the columns. Samples that landed in-bag for every tree get no vote and
        are excluded from the accuracy.

        This is the forest's built-in validation set — no separate holdout
        needed. Caveat for this project: OOB rows are scattered across the whole
        training window, so OOB accuracy is a RANDOM split estimate. It will
        read optimistically compared to the chronological test set, and that gap
        is itself worth reporting.
        """
        n_samples = len(y)
        votes = np.full((len(self.trees_), n_samples), -1, dtype=int)
        for t, (tree, oob) in enumerate(zip(self.trees_, self.oob_indices_)):
            if len(oob):
                votes[t, oob] = tree.predict(X[oob])
        combined = majority_vote(votes)
        voted = combined != -1
        if not voted.any():
            return None
        return float(np.mean(combined[voted] == np.asarray(y)[voted]))

    def feature_importances(self, n_features):
        """Total information gain contributed by each feature, normalised.

        Sums `gain * n_samples` over every internal node that split on a given
        feature, across every tree, then divides by the total. Weighting by
        n_samples matters: a 0.05-gain split at the root over 6000 rows is worth
        far more than a 0.05-gain split over 12 rows near a leaf.
        """
        totals = np.zeros(n_features, dtype=float)

        def walk(node):
            if node.is_leaf():
                return
            totals[node.feature] += node.gain * node.n_samples
            walk(node.left)
            walk(node.right)

        for tree in self.trees_:
            walk(tree.root)
        return totals / totals.sum() if totals.sum() > 0 else totals
