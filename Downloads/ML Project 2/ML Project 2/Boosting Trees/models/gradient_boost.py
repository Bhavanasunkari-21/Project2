import numpy as np
from models.decision_tree import DecisionTree


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class GradientBoostClassifier:
    """Gradient‑boosting classifier implemented exactly as Algorithm 10.3 (ESL 2e)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: list[DecisionTree] = []
        self.init_pred: float = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit M = n_estimators trees using logistic deviance loss."""
        y = y.astype(float)
        # 1️⃣   initial constant = log‑odds
        pos = np.clip(y.sum(), 1e-6, None)
        neg = np.clip(len(y) - pos, 1e-6, None)
        self.init_pred = np.log(pos / neg)

        F = np.full_like(y, self.init_pred, dtype=float)  # current f(x_i)

        for m in range(self.n_estimators):
            # 2(a) pseudo‑residuals   r = y − p
            p = _sigmoid(F)
            residual = y - p

            # 2(b) fit regression tree on residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)

            # ---- STEP 2(c): compute γ_{jm} for each leaf ------
            regions = tree.apply(X)  # leaf id per sample
            for leaf_id in np.unique(regions):
                idx = regions == leaf_id
                p_i = p[idx]
                numerator = np.sum(residual[idx])          # ∑ (y - p)
                denominator = np.sum(p_i * (1 - p_i))      # ∑ p(1-p)
                gamma = numerator / (denominator + 1e-12)  # Newton step; avoid /0
                tree.set_leaf_value(leaf_id, gamma)

            # predictions of the *scaled* tree (γ already stored)
            update = tree.predict(X)

            # 2(d) update F and stash the tree
            F += self.learning_rate * update
            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full((len(X),), self.init_pred, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return _sigmoid(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# Convenience alias
gb_binary = GradientBoostClassifier
