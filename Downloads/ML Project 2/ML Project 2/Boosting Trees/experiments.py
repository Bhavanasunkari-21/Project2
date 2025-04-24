import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add model and test folders to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models'))
sys.path.append(os.path.join(current_dir, 'tests'))

# Correct imports
from gradient_boost import GradientBoostClassifier
from data_generator import (
    generate_simple_dataset, 
    generate_nonlinear_dataset, 
    generate_xor_dataset,
    visualize_dataset, 
    visualize_predictions,
    visualize_decision_boundary
)

def run_experiment(X, y, dataset_name, n_estimators=100, learning_rate=0.1, max_depth=3):
    print(f"\n{'='*50}")
    print(f"Experiment: {dataset_name}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"Training time: {duration:.2f} seconds")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\nTrain Accuracy: {accuracy_score(y_train, y_train_pred):.4f}, F1: {f1_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}, F1: {f1_score(y_test, y_test_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    visualize_predictions(X_test, y_test, y_test_pred, title=f"{dataset_name} - Predictions")
    if X.shape[1] == 2:
        visualize_decision_boundary(X, y, model, title=f"{dataset_name} - Decision Boundary")

    return model

def run_learning_curve_experiment(X, y, dataset_name, n_estimators_list=[10, 50, 100, 200]):
    print(f"\n{'='*50}\nLearning Curve Experiment: {dataset_name}\n{'='*50}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_accuracies = []
    test_accuracies = []

    for n in n_estimators_list:
        model = GradientBoostClassifier(n_estimators=n, learning_rate=0.1, max_depth=3)
        model.fit(X_train, y_train)

        train_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
        test_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_accuracies, 'o-', label='Train Accuracy')
    plt.plot(n_estimators_list, test_accuracies, 'o-', label='Test Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title(f"{dataset_name} - Learning Curve")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(current_dir, 'figures')
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"learning_curve_{dataset_name.lower().replace(' ', '_')}.png"))
    plt.close()

def run_hyperparameter_experiment(X, y, dataset_name):
    print(f"\n{'='*50}\nHyperparameter Experiment: {dataset_name}\n{'='*50}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    learning_rates = [0.01, 0.1, 0.5, 1.0]
    max_depths = [1, 3, 5, 7]

    results = []
    for lr in learning_rates:
        for depth in max_depths:
            model = GradientBoostClassifier(n_estimators=50, learning_rate=lr, max_depth=depth)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            results.append((lr, depth, acc))

    heatmap_data = np.zeros((len(max_depths), len(learning_rates)))
    for i, depth in enumerate(max_depths):
        for j, lr in enumerate(learning_rates):
            for (lr_val, depth_val, acc_val) in results:
                if lr_val == lr and depth_val == depth:
                    heatmap_data[i, j] = acc_val

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis')
    plt.colorbar(label='Test Accuracy')
    plt.xticks(np.arange(len(learning_rates)), learning_rates)
    plt.yticks(np.arange(len(max_depths)), max_depths)
    plt.xlabel('Learning Rate')
    plt.ylabel('Max Depth')
    plt.title(f"{dataset_name} - Hyperparameter Grid Search")
    for i in range(len(max_depths)):
        for j in range(len(learning_rates)):
            plt.text(j, i, f"{heatmap_data[i, j]:.2f}", ha='center', va='center', color='white')

    output_path = os.path.join(current_dir, 'figures')
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"hyperparameter_heatmap_{dataset_name.lower().replace(' ', '_')}.png"))
    plt.close()

if __name__ == "__main__":
    X1, y1 = generate_simple_dataset(n_samples=1000, noise=0.1)
    X2, y2 = generate_nonlinear_dataset(n_samples=1000, noise=0.1)
    X3, y3 = generate_xor_dataset(n_samples=1000, noise=0.1)

    os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'figures'), exist_ok=True)

    visualize_dataset(X1, y1, "Simple Linear Dataset")
    visualize_dataset(X2, y2, "Nonlinear Circular Dataset")
    visualize_dataset(X3, y3, "XOR Dataset")

    run_experiment(X1, y1, "Simple Linear Dataset")
    run_experiment(X2, y2, "Nonlinear Circular Dataset")
    run_experiment(X3, y3, "XOR Dataset", n_estimators=200)

    run_learning_curve_experiment(X1, y1, "Simple Linear Dataset")
    run_learning_curve_experiment(X2, y2, "Nonlinear Circular Dataset")
    run_learning_curve_experiment(X3, y3, "XOR Dataset")

    run_hyperparameter_experiment(X1, y1, "Simple Linear Dataset")
    run_hyperparameter_experiment(X2, y2, "Nonlinear Circular Dataset")
    run_hyperparameter_experiment(X3, y3, "XOR Dataset")

    print("\n✅ All experiments completed. Check the 'figures' folder for plots.")
