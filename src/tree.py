"""Decision tree building blocks: entropy, information gain, and the Node
data container.

The math:
    Entropy(S) = -sum_i p_i * log2(p_i)
    IG(S, split) = Entropy(S) - sum_children (|child| / |S|) * Entropy(child)
"""

import numpy as np


def entropy(labels):
    """Shannon entropy of a label array, in bits.

    Args:
        labels: 1-D array-like of class labels (any hashable values —
            ints, strings, booleans). May be empty.

    Returns:
        float: entropy in [0, log2(n_classes)]. For binary labels this is
        [0, 1]. Pure set -> 0.0; 50/50 binary set -> 1.0. An empty array
        should return 0.0 (a convention that keeps IG well-defined when a
        candidate split leaves one side empty).
    """

    if len(labels) == 0:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)

    probabilities = counts / len(labels)

    # `+ 0.0` normalises the -0.0 that negating a zero sum would otherwise
    # produce for a pure set.
    return -np.sum(probabilities * np.log2(probabilities)) + 0.0


def information_gain(parent_labels, left_labels, right_labels):
    """Information gain of a binary split, in bits.

    Args:
        parent_labels: 1-D array-like — labels at the node before splitting.
        left_labels:  1-D array-like — labels routed to the left child.
        right_labels: 1-D array-like — labels routed to the right child.
            left + right together must be exactly the parent set
            (this function does not verify that; the split-finder will
            guarantee it).

    Returns:
        float: entropy(parent) minus the size-weighted average entropy of
        the two children. >= 0 always; 0 means the split carries no
        information about the label.
    """
    parent_entropy = entropy(parent_labels)
    weight_left = len(left_labels) / len(parent_labels)
    weight_right = len(right_labels) / len(parent_labels)
    weighted_child_entropy = (weight_left * entropy(left_labels) + weight_right * entropy(right_labels))
    return parent_entropy - weighted_child_entropy


def best_split(X, y, feature_indices=None, min_samples_leaf=1):
    """Brute-force search for the split with the highest information gain.

    Tries every candidate (feature, threshold) pair and keeps the best one.
    Candidate thresholds for a feature are the midpoints between consecutive
    distinct sorted values — that covers every partition the feature can
    actually produce, without testing redundant cut points.

    Args:
        X: 2-D numpy array, shape (n_samples, n_features). Numeric only —
            categoricals like stance_matchup must already be encoded.
        y: 1-D array of labels, length n_samples.
        feature_indices: optional list of column indices to consider. None
            means "try all features". The random forest will pass a random
            subset here — that's the only change needed to reuse this
            function for the forest.
        min_samples_leaf: reject any candidate split that would leave a child
            with fewer than this many samples. Enforced here rather than
            after the fact, because a split rejected later would otherwise
            have already beaten a legal split on gain.

    Returns:
        (feature, threshold, gain) — the winning column index, its cut point,
        and the information gain achieved. Returns (None, None, 0.0) if no
        split produces positive gain (node is pure, or every feature is
        constant across the samples). Samples with X[:, feature] <= threshold
        go left, matching Node's convention.

    Complexity:
        The direct implementation rescores each candidate threshold by
        rebuilding a boolean mask over all n samples: O(n) per threshold,
        O(n^2) per feature. This version sorts the column once and sweeps the
        cut point left to right, carrying a running count of positive labels
        via cumsum, so each candidate's child label counts come out of a
        prefix sum in O(1). Cost per feature is O(n log n), dominated by the
        sort.

        The sweep represents a child as (size, positive count), which only
        works for binary labels in {0, 1} — hence the check below.
        entropy() and information_gain() above remain general, and serve as
        the independent reference the tests compare against.
    """
    y = np.asarray(y)
    if y.dtype == bool:
        y = y.astype(np.int64)
    # The cumsum sweep below reads y as counts of positives, so anything other
    # than {0, 1} would silently produce a meaningless "gain". Fail loudly
    # instead — entropy() above stays general, this function does not.
    if y.size and not np.all((y == 0) | (y == 1)):
        raise ValueError(
            "best_split requires binary labels in {0, 1}; got "
            f"{np.unique(y)[:5]}"
        )

    n_samples = len(y)
    if n_samples < 2 * min_samples_leaf:
        return None, None, 0.0

    if feature_indices is None:
        feature_indices = range(X.shape[1])

    parent_entropy = entropy(y)
    if parent_entropy == 0.0:  # already pure — nothing to gain
        return None, None, 0.0

    best_feature, best_threshold, best_gain = None, None, 0.0

    for feature in feature_indices:
        column = X[:, feature]
        order = np.argsort(column, kind="mergesort")
        x_sorted = column[order]
        y_sorted = y[order]

        # left_positives[i] = number of 1-labels in the first (i+1) samples,
        # i.e. the left child if we cut immediately after position i.
        left_positives = np.cumsum(y_sorted)[:-1]
        left_n = np.arange(1, n_samples)
        right_n = n_samples - left_n
        right_positives = y_sorted.sum() - left_positives

        # A cut is only real if the values on either side actually differ —
        # otherwise identical feature values would land in different children.
        valid = x_sorted[:-1] < x_sorted[1:]
        valid &= (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
        if not valid.any():
            continue

        weighted_child = (
            left_n * _binary_entropy(left_positives, left_n)
            + right_n * _binary_entropy(right_positives, right_n)
        ) / n_samples
        gains = np.where(valid, parent_entropy - weighted_child, -np.inf)

        i = int(np.argmax(gains))
        if gains[i] > best_gain:
            # Midpoint of the two bracketing values. In float arithmetic the
            # midpoint of two very close (or very large) values can round up
            # to the right-hand value, which would send EVERY sample left and
            # recurse forever on an identical node. Fall back to the left
            # value, which always separates the two.
            with np.errstate(over="ignore"):
                threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
            if not threshold < x_sorted[i + 1]:
                threshold = x_sorted[i]
            best_gain = float(gains[i])
            best_feature = int(feature)
            best_threshold = float(threshold)

    if best_feature is None:
        return None, None, 0.0
    return best_feature, best_threshold, best_gain


def _binary_entropy(positives, totals):
    """Vectorised entropy for many binary children at once, in bits.

    Args:
        positives: array of positive-label counts, one per candidate child.
        totals: array of child sizes (same shape, all > 0).

    Returns the same value as entropy() would for each child, computed from
    counts alone so the split sweep never has to materialise the label arrays.
    """
    p = positives / totals
    q = 1.0 - p
    # 0 * log2(0) is defined as 0 here; np.where alone would still evaluate
    # log2(0) and warn, so clip the input and zero the term afterwards.
    with np.errstate(divide="ignore", invalid="ignore"):
        term_p = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
        term_q = np.where(q > 0, q * np.log2(np.where(q > 0, q, 1.0)), 0.0)
    return -(term_p + term_q)


class Node:
    """A single node in the decision tree. Pure data container.

    Internal node: feature/threshold set, left/right point to child Nodes,
    prediction is None. Samples with feature value <= threshold go left,
    > threshold go right.

    Leaf: prediction set (the majority class), everything else None.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 prediction=None, n_samples=0, gain=0.0):
        self.feature = feature        # column index to split on
        self.threshold = threshold    # split point: <= goes left
        self.left = left              # Node
        self.right = right            # Node
        self.prediction = prediction  # class label if leaf, else None
        self.n_samples = n_samples    # training rows that reached this node
        self.gain = gain              # information gain of this node's split

    def is_leaf(self):
        return self.prediction is not None


# ---------------------------------------------------------------------------
# Growing the tree
# ---------------------------------------------------------------------------
def _majority_class(y):
    """Most common label; ties break toward the smaller label value."""
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2,
               min_samples_leaf=1, feature_subset=None, rng=None):
    """Recursively grow a decision tree and return its root Node.

    The recursion is the whole algorithm: find the best split here, partition
    the rows, and repeat on each half. Every branch ends at a leaf because at
    least one stopping rule always eventually fires — the node runs out of
    depth, out of samples, out of label variety, or out of splits that gain
    anything.

    Args:
        X: 2-D numeric array (n_samples, n_features).
        y: 1-D binary label array in {0, 1}.
        depth: current depth; 0 at the root. Managed by the recursion.
        max_depth: stop splitting past this depth. None = grow until the
            other rules stop it (the deliberately overfit tree).
        min_samples_split: a node with fewer rows than this becomes a leaf
            without even looking for a split.
        min_samples_leaf: no split may produce a child smaller than this.
        feature_subset: number of features to sample at each node, or None to
            consider all of them. This is the random forest hook — sampling a
            fresh subset PER NODE (not per tree) is what decorrelates the
            trees in the ensemble.
        rng: numpy Generator, required when feature_subset is set.

    Returns:
        Node: the root of the (sub)tree.
    """
    n_samples, n_features = X.shape

    # Stopping rules that don't require searching for a split at all.
    if (
        (max_depth is not None and depth >= max_depth)
        or n_samples < min_samples_split
        or len(np.unique(y)) == 1
    ):
        return Node(prediction=_majority_class(y), n_samples=n_samples)

    if feature_subset is None:
        feature_indices = None
    else:
        if rng is None:
            raise ValueError("feature_subset requires an rng (numpy Generator)")
        k = min(feature_subset, n_features)
        feature_indices = rng.choice(n_features, size=k, replace=False)

    feature, threshold, gain = best_split(
        X, y, feature_indices=feature_indices, min_samples_leaf=min_samples_leaf
    )

    # No split beat zero gain (constant features, or every legal cut is
    # uninformative) -> this node is as good as it gets.
    if feature is None:
        return Node(prediction=_majority_class(y), n_samples=n_samples)

    go_left = X[:, feature] <= threshold
    # Belt-and-braces: best_split already guarantees both children are
    # non-empty, so an empty side would mean a threshold/partition mismatch.
    # Becoming a leaf here is strictly better than recursing forever.
    if not go_left.any() or go_left.all():
        return Node(prediction=_majority_class(y), n_samples=n_samples)

    left = build_tree(
        X[go_left], y[go_left], depth + 1, max_depth, min_samples_split,
        min_samples_leaf, feature_subset, rng,
    )
    right = build_tree(
        X[~go_left], y[~go_left], depth + 1, max_depth, min_samples_split,
        min_samples_leaf, feature_subset, rng,
    )
    return Node(feature=feature, threshold=threshold, left=left, right=right,
                n_samples=n_samples, gain=gain)


def predict_one(node, row):
    """Walk one sample from the root to a leaf and return that leaf's label."""
    while not node.is_leaf():
        node = node.left if row[node.feature] <= node.threshold else node.right
    return node.prediction


def predict(root, X):
    """Predict labels for every row of X."""
    return np.array([predict_one(root, row) for row in X])


def tree_depth(node):
    """Length of the longest root-to-leaf path (a single leaf has depth 0)."""
    if node.is_leaf():
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_nodes(node):
    """Return (n_leaves, n_internal_nodes)."""
    if node.is_leaf():
        return 1, 0
    left_leaves, left_internal = count_nodes(node.left)
    right_leaves, right_internal = count_nodes(node.right)
    return left_leaves + right_leaves, left_internal + right_internal + 1


class DecisionTree:
    """Thin sklearn-style wrapper around build_tree/predict.

    Exists so the training scripts and the random forest can hold a fitted
    model as one object instead of passing a bare root Node around.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 feature_subset=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_subset = feature_subset
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.root = build_tree(
            np.asarray(X, dtype=float), np.asarray(y),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            feature_subset=self.feature_subset,
            rng=rng,
        )
        return self

    def predict(self, X):
        return predict(self.root, np.asarray(X, dtype=float))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def depth(self):
        return tree_depth(self.root)

    @property
    def n_leaves(self):
        return count_nodes(self.root)[0]

    @property
    def n_internal(self):
        return count_nodes(self.root)[1]
