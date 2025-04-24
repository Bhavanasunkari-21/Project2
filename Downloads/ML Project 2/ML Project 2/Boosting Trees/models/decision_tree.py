
from __future__ import annotations
import numpy as np


class _Node:
    """Internal helper class for tree nodes."""

    _id_counter = 0  # class‑level counter so every leaf has a unique id

    def __init__(self,
                 value: float | None = None,
                 feature: int | None = None,
                 threshold: float | None = None,
                 left: "_Node | None" = None,
                 right: "_Node | None" = None):
        self.id = _Node._id_counter  # unique identifier (used by apply/set_leaf_value)
        _Node._id_counter += 1
        self.value = value          # prediction stored in this node (only valid for leaves)
        self.feature = feature      # int – feature index used to split (None for leaf)
        self.threshold = threshold  # float – splitting threshold
        self.left = left            # _Node
        self.right = right          # _Node

    # convenience predicates
    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """Very small regression tree (variance reduction) for gradient boosting."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root: _Node | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Grow tree recursively to minimise squared error."""
        self.root = self._build(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for row in X])

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the *leaf id* each sample ends in."""
        return np.array([self._apply_row(row, self.root) for row in X])

    def set_leaf_value(self, leaf_id: int, value: float):
        """Replace the stored value in the leaf with `leaf_id`."""
        self._set_leaf(self.root, leaf_id, value)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # stopping conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) <= self.min_samples_leaf:
            return _Node(value=float(np.mean(y)))

        n_samples, n_features = X.shape
        best_feat, best_thr, best_loss = None, None, np.inf

        # brute‑force scan of all (feature, threshold) pairs
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                loss = (np.var(y[left_mask]) * left_mask.sum()) + (np.var(y[right_mask]) * right_mask.sum())
                if loss < best_loss:
                    best_feat, best_thr, best_loss = feat, thr, loss
        # if no split improved variance, make leaf
        if best_feat is None:
            return _Node(value=float(np.mean(y)))

        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask
        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=best_feat, threshold=best_thr, left=left_child, right=right_child)

    # single row traversal helpers
    def _predict_row(self, row: np.ndarray, node: _Node) -> float:
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def _apply_row(self, row: np.ndarray, node: _Node) -> int:
        if node.is_leaf:
            return node.id
        if row[node.feature] <= node.threshold:
            return self._apply_row(row, node.left)
        return self._apply_row(row, node.right)

    def _set_leaf(self, node: _Node, leaf_id: int, value: float):
        if node is None:
            return
        if node.is_leaf and node.id == leaf_id:
            node.value = value
            return
        if not node.is_leaf:
            self._set_leaf(node.left, leaf_id, value)
            self._set_leaf(node.right, leaf_id, value)
