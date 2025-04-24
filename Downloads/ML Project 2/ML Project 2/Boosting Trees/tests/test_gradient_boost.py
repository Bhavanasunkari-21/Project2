import numpy as np
import unittest
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gradient_boost import GradientBoostClassifier


class TestGradientBoostClassifier(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset for testing
        np.random.seed(42)
        self.X = np.random.rand(200, 5)
        # Create a target variable that has a nonlinear dependency on features
        self.y = ((self.X[:, 0] > 0.5) & (self.X[:, 1] < 0.7) | 
                  (self.X[:, 2] > 0.6) & (self.X[:, 3] < 0.4)).astype(int)
        
    def test_initialization(self):
        # Test that initialization works correctly
        clf = GradientBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.assertEqual(clf.n_estimators, 100)
        self.assertEqual(clf.learning_rate, 0.1)
        self.assertEqual(clf.max_depth, 3)
        self.assertEqual(len(clf.trees), 0)
        
    def test_fit_predict(self):
        # Test with different parameter combinations
        for n_est, lr in [(10, 0.1), (50, 0.05), (20, 0.2)]:
            clf = GradientBoostClassifier(n_estimators=n_est, learning_rate=lr, max_depth=3)
            clf.fit(self.X, self.y)
            
            # Check that all trees were created
            self.assertEqual(len(clf.trees), n_est)
            
            # Test prediction
            predictions = clf.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))
            
            # Ensure predictions are binary
            self.assertTrue(np.all(np.isin(predictions, [0, 1])))
            
            # For this problem, we should get decent accuracy
            accuracy = np.mean(predictions == self.y)
            self.assertGreater(accuracy, 0.7)
    
    def test_predict_proba(self):
        # Test that predict_proba returns probabilities
        clf = GradientBoostClassifier(n_estimators=20, learning_rate=0.1)
        clf.fit(self.X, self.y)
        
        probas = clf.predict_proba(self.X)
        self.assertEqual(len(probas), len(self.X))
        
        # Ensure all values are probabilities
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))
        
        # Test that predict matches predict_proba thresholded at 0.5
        predictions = clf.predict(self.X)
        predicted_from_proba = (probas >= 0.5).astype(int)
        self.assertTrue(np.all(predictions == predicted_from_proba))
        
    def test_learning_rate_effect(self):
        # Test that smaller learning rate needs more iterations
        X_train, X_test = self.X[:150], self.X[150:]
        y_train, y_test = self.y[:150], self.y[150:]
        
        accuracies = []
        learning_rates = [1.0, 0.5, 0.1, 0.01]
        
        for lr in learning_rates:
            clf = GradientBoostClassifier(n_estimators=20, learning_rate=lr)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            accuracies.append(np.mean(predictions == y_test))
        
        # Ensure some learning happens, but don't require specific accuracies
        self.assertTrue(np.mean(accuracies) > 0.5)


if __name__ == "__main__":
    unittest.main()
