# CS584

Illinois Institute of Technology
Team Members:

Name:Suraj Sajjanrao Ghatage
CWID:A20539840

Name: Bhavana Sunkari
Cwid:A20543227

# Test Result 
C:\Users\pamid\Downloads\ML Project 2\ML Project 2>python "Boosting Trees/experiments.py"

==================================================
Experiment: Simple Linear Dataset
==================================================
Training time: 28.05 seconds

Train Accuracy: 0.9187, F1: 0.9144
Test Accuracy: 0.8650, F1: 0.8670
Confusion Matrix:
 [[85 18]
 [ 9 88]]

==================================================
Experiment: Nonlinear Circular Dataset
==================================================
Training time: 27.60 seconds

Train Accuracy: 0.9137, F1: 0.8089
Test Accuracy: 0.8650, F1: 0.7216
Confusion Matrix:
 [[138   6]
 [ 21  35]]

==================================================
Experiment: XOR Dataset
==================================================
Training time: 57.28 seconds

Train Accuracy: 0.9463, F1: 0.9466
Test Accuracy: 0.8900, F1: 0.8804
Confusion Matrix:
 [[97  9]
 [13 81]]

==================================================
Learning Curve Experiment: Simple Linear Dataset
==================================================

==================================================
Learning Curve Experiment: Nonlinear Circular Dataset
==================================================

==================================================
Learning Curve Experiment: XOR Dataset
==================================================


==================================================
Hyperparameter Experiment: Simple Linear Dataset
==================================================
==================================================
Hyperparameter Experiment: Nonlinear Circular Dataset
==================================================

✅ All experiments completed. Check the 'figures' folder for plots.

# Gradient Boosting Tree Classification

This project implements a gradient boosting tree classifier from first principles, following Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). The implementation includes a decision tree regressor and a gradient boosting classifier with the standard fit-predict interface.

## Implementation Details

The implementation consists of two main components:

1. **Decision Tree (`decision_tree.py`)**: A simple decision tree that minimizes squared error for regression tasks. It includes methods for:
   - `fit`: Training the tree on input data
   - `predict`: Making predictions
   - `apply`: Finding which leaf each sample falls into
   - `set_leaf_value`: Changing the prediction value of a specific leaf

2. **Gradient Boosting Classifier (`gradient_boost.py`)**: Implements Algorithm 10.3 from ESL using binary logistic deviance as the loss function. It includes:
   - `fit`: Trains a sequence of trees, each focused on the negative gradient
   - `predict_proba`: Returns class probabilities
   - `predict`: Returns class predictions

## What the Model Does and When to Use It

The gradient boosting tree classifier is an ensemble method that sequentially trains decision trees, where each tree attempts to correct the errors made by the previous trees. 

This algorithm:
- Builds a strong classifier from many weak learners (shallow decision trees)
- Uses gradient descent in function space to minimize a loss function
- For classification, uses the logistic deviance loss

**When to use it:**
- When you need a powerful classifier with high predictive accuracy
- For problems where the decision boundary is complex and nonlinear
- When interpretability is somewhat important (compared to black-box models)
- When you have enough data for training multiple trees
- For problems with heterogeneous features (mix of categorical and numerical)

The model is particularly effective because:
- It can capture complex nonlinear relationships
- It's resistant to outliers
- It handles mixed data types well
- It can be tuned for the bias-variance tradeoff using parameters

## Testing Methodology

To verify the implementation is working correctly, I created multiple tests:

1. **Unit tests** (`tests/test_decision_tree.py` and `tests/test_gradient_boost.py`):
   - Test the initialization, fitting, prediction, and individual components
   - Verify each method works as expected with simple inputs
   - Ensure proper behavior with edge cases

2. **Synthetic datasets** (`tests/data_generator.py`):
   - Created three synthetic datasets with different patterns:
     - Simple linear decision boundary
     - Nonlinear circular decision boundary
     - XOR pattern (historically challenging for simple models)

3. **Comprehensive experiments** (`experiments.py`):
   - Performance evaluation on all datasets (accuracy, F1 score)
   - Learning curve analysis (performance vs. number of trees)
   - Hyperparameter sensitivity analysis (learning rate, max depth)
   - Decision boundary visualization

The test results show that:
- The model achieves high accuracy on the linear dataset (as expected)
- It can learn the nonlinear circular pattern with sufficient trees
- It successfully captures the XOR pattern with appropriate hyperparameters
- The model demonstrates the expected behavior of improving with more trees until convergence

These tests confirm the implementation follows the principles described in ESL and behaves like a proper gradient boosting classifier.

## Parameters for Tuning Performance

The implementation exposes the following parameters for tuning:

1. **`n_estimators` (default=100)**:
   - Number of boosting stages (trees) to perform
   - Higher values usually improve performance at the cost of training time
   - Too many can lead to overfitting on some datasets

2. **`learning_rate` (default=0.1)**:
   - Shrinks the contribution of each tree
   - Smaller values require more trees but can lead to better test performance
   - Trade-off between n_estimators and learning_rate (smaller learning rates need more trees)

3. **`max_depth` (default=3)**:
   - Maximum depth of the individual regression trees
   - Controls complexity of the base learners
   - Smaller values prevent overfitting but might underfit if too small

### Basic Usage Examples

```python
# Import the classifier
from models.gradient_boost import GradientBoostClassifier

# Create and train the model
clf = GradientBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Get class probabilities
y_proba = clf.predict_proba(X_test)
```

### Example of Parameter Tuning

For simpler problems with clear boundaries:
```python
clf = GradientBoostClassifier(n_estimators=50, learning_rate=0.5, max_depth=2)
```

For complex nonlinear relationships:
```python
clf = GradientBoostClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
```

## Limitations and Future Improvements

### Current Limitations

1. **Binary Classification Only**: The current implementation only handles binary classification problems (0 and 1 classes).

2. **Computational Efficiency**: The implementation doesn't include many optimizations found in professional libraries, making it slower on large datasets.

3. **No Regularization**: Unlike some implementations, this version doesn't have explicit regularization parameters like min_samples_split or min_impurity_decrease.

4. **Limited Feature Processing**: The implementation expects numerical features and doesn't handle categorical features automatically.

### Potential Improvements

Given more time, these limitations could be addressed by:

1. Extending to multiclass classification using one-vs-rest or more sophisticated approaches
2. Implementing early stopping based on validation performance
3. Adding regularization parameters and feature importance calculation
4. Implementing feature subsampling and row subsampling (similar to random forests)
5. Adding support for different loss functions
6. Adding proper categorical feature handling

## Running the Code

To run the implementation and tests:

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Run the unit tests:
```
python -m pytest "Boosting Trees/tests" -v

```

3. Run the experiments:
```
python "Boosting Trees/experiments.py"

```

The experiments will generate visualizations in the `Boosting Trees/figures` directory, showing the datasets, model predictions, decision boundaries, and parameter impact.

## Conclusion

This implementation provides a working gradient boosting classifier that follows the principles described in Elements of Statistical Learning. While it lacks some optimizations and features of professional libraries, it demonstrates the core algorithm and provides a solid foundation for understanding how gradient boosting works for classification tasks.
