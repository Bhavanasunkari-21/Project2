import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_simple_dataset(n_samples=500, noise=0.1, save=False):
    """
    Generate a simple dataset with a linear decision boundary.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add to the target
    save : bool
        Whether to save the dataset to .npy files
        
    Returns:
    --------
    X : numpy.ndarray
        Features (n_samples, 2)
    y : numpy.ndarray
        Binary target (n_samples,)
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, 2)
    
    # Linear decision boundary: x1 > x2
    y_clean = (X[:, 0] > X[:, 1]).astype(int)
    
    # Add noise
    flip_idx = np.random.choice(
        n_samples, 
        size=int(n_samples * noise), 
        replace=False
    )
    y = y_clean.copy()
    y[flip_idx] = 1 - y[flip_idx]
    
    if save:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'X_simple.npy'), X)
        np.save(os.path.join(data_dir, 'y_simple.npy'), y)
    
    return X, y

def generate_nonlinear_dataset(n_samples=500, noise=0.1, save=False):
    """
    Generate a dataset with a nonlinear decision boundary.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add to the target
    save : bool
        Whether to save the dataset to .npy files
        
    Returns:
    --------
    X : numpy.ndarray
        Features (n_samples, 2)
    y : numpy.ndarray
        Binary target (n_samples,)
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, 2) * 4 - 2  # Scale to [-2, 2]
    
    # Circular decision boundary
    radius = 1.0
    distance = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y_clean = (distance < radius).astype(int)
    
    # Add noise
    flip_idx = np.random.choice(
        n_samples, 
        size=int(n_samples * noise), 
        replace=False
    )
    y = y_clean.copy()
    y[flip_idx] = 1 - y[flip_idx]
    
    if save:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'X_nonlinear.npy'), X)
        np.save(os.path.join(data_dir, 'y_nonlinear.npy'), y)
    
    return X, y

def generate_xor_dataset(n_samples=500, noise=0.1, save=False):
    """
    Generate a dataset with an XOR pattern.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add to the target
    save : bool
        Whether to save the dataset to .npy files
        
    Returns:
    --------
    X : numpy.ndarray
        Features (n_samples, 2)
    y : numpy.ndarray
        Binary target (n_samples,)
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, 2) * 2 - 1  # Scale to [-1, 1]
    
    # XOR pattern
    y_clean = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    # Add noise
    flip_idx = np.random.choice(
        n_samples, 
        size=int(n_samples * noise), 
        replace=False
    )
    y = y_clean.copy()
    y[flip_idx] = 1 - y[flip_idx]
    
    if save:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'X_xor.npy'), X)
        np.save(os.path.join(data_dir, 'y_xor.npy'), y)
    
    return X, y

def visualize_dataset(X, y, title="Dataset Visualization"):
    """
    Visualize a 2D dataset with different colors for each class.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features (n_samples, 2)
    y : numpy.ndarray
        Binary target (n_samples,)
    title : str
        Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create directory for figures if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.savefig(os.path.join(fig_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()

def visualize_predictions(X, y_true, y_pred, title="Model Predictions"):
    """
    Visualize predictions of a model on a 2D dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features (n_samples, 2)
    y_true : numpy.ndarray
        True binary target (n_samples,)
    y_pred : numpy.ndarray
        Predicted binary target (n_samples,)
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # True labels
    plt.subplot(1, 2, 1)
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1], c='blue', label='Class 0', alpha=0.5)
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], c='red', label='Class 1', alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('True Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', label='Pred Class 0', alpha=0.5)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', label='Pred Class 1', alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Predicted Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Create directory for figures if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.savefig(os.path.join(fig_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()

def visualize_decision_boundary(X, y, model, title="Decision Boundary", grid_step=0.02):
    """
    Visualize the decision boundary of a model on a 2D dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features (n_samples, 2)
    y : numpy.ndarray
        Binary target (n_samples,)
    model : object
        Model with predict method
    title : str
        Plot title
    grid_step : float
        Step size for the meshgrid
    """
    plt.figure(figsize=(8, 6))
    
    # Determine the plot boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a meshgrid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    
    # Make predictions on the meshgrid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    
    # Plot the training points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', edgecolors='k', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', edgecolors='k', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    # Create directory for figures if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.savefig(os.path.join(fig_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()

if __name__ == "__main__":
    # Generate and save all datasets
    X_simple, y_simple = generate_simple_dataset(save=True)
    X_nonlinear, y_nonlinear = generate_nonlinear_dataset(save=True)
    X_xor, y_xor = generate_xor_dataset(save=True)
    
    # Visualize all datasets
    visualize_dataset(X_simple, y_simple, "Simple Linear Dataset")
    visualize_dataset(X_nonlinear, y_nonlinear, "Nonlinear Circular Dataset")
    visualize_dataset(X_xor, y_xor, "XOR Dataset")
    
    print("Generated and saved datasets to the data directory.")
    print("Generated visualizations in the figures directory.")
