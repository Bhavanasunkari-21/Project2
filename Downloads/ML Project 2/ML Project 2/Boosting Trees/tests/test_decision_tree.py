import numpy as np
import unittest
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decision_tree import DecisionTree


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset for testing
        np.random.seed(42)
        self.X = np.random.rand(100, 5)
        # Create a target variable that depends on one feature
        self.y = (self.X[:, 0] > 0.5).astype(float)
        
    def test_initialization(self):
        # Test that initialization works correctly
        tree = DecisionTree(max_depth=3, min_samples_leaf=2)
        self.assertEqual(tree.max_depth, 3)
        self.assertEqual(tree.min_samples_leaf, 2)
        self.assertIsNone(tree.root)
        
    def test_fit_predict(self):
        # Test that fitting and prediction work
        tree = DecisionTree(max_depth=3, min_samples_leaf=2)
        tree.fit(self.X, self.y)
        
        # Test that the root node is created
        self.assertIsNotNone(tree.root)
        
        # Test prediction
        predictions = tree.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        
        # Since this is a simple problem, accuracy should be good
        accuracy = np.mean((predictions > 0.5) == self.y)
        self.assertGreater(accuracy, 0.7)
        
    def test_apply(self):
        # Test the apply method returns leaf ids
        tree = DecisionTree(max_depth=3, min_samples_leaf=2)
        tree.fit(self.X, self.y)
        
        leaf_ids = tree.apply(self.X)
        self.assertEqual(len(leaf_ids), len(self.X))
        self.assertTrue(np.all(leaf_ids >= 0))
        
    def test_set_leaf_value(self):
        # Test that we can set leaf values
        tree = DecisionTree(max_depth=3, min_samples_leaf=2)
        tree.fit(self.X, self.y)
        
        leaf_ids = tree.apply(self.X)
        unique_ids = np.unique(leaf_ids)
        
        # Set all leaves to a fixed value
        test_value = 0.75
        for leaf_id in unique_ids:
            tree.set_leaf_value(leaf_id, test_value)
        
        # Verify all predictions equal the test value
        predictions = tree.predict(self.X)
        self.assertTrue(np.allclose(predictions, test_value))
        
    def test_max_depth_limits_tree(self):
        # Test that max_depth parameter actually limits tree depth
        for max_depth in [1, 2, 3]:
            tree = DecisionTree(max_depth=max_depth)
            tree.fit(self.X, self.y)
            
            # Count depth of the tree
            def count_depth(node):
                if node is None or node.is_leaf:
                    return 0
                return 1 + max(count_depth(node.left), count_depth(node.right))
            
            self.assertLessEqual(count_depth(tree.root), max_depth)


if __name__ == "__main__":
    unittest.main()
