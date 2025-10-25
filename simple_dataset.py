"""
Simple Dataset 1 - Binary Classification Dataset Generator
Generates synthetic datasets for SVM testing (from Machine Learning Scratch Sprint)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_simple_dataset_1(n_samples=100, noise=0.1, random_state=42):
    """
    Generate Simple Dataset 2 for binary classification.
    Creates two classes that are linearly separable with some overlap.
    
    Parameters
    ----------
    n_samples : int, default=100
        Total number of samples to generate
    noise : float, default=0.1
        Standard deviation of Gaussian noise
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray, shape (n_samples, 2)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Labels (-1 or 1)
    """
    np.random.seed(random_state)
    
    # Generate class 1 (label = 1)
    n_class1 = n_samples // 2
    mean1 = [2, 2]
    cov1 = [[1.0, 0.5], [0.5, 1.0]]
    X_class1 = np.random.multivariate_normal(mean1, cov1, n_class1)
    y_class1 = np.ones(n_class1)
    
    # Generate class 2 (label = -1)
    n_class2 = n_samples - n_class1
    mean2 = [-2, -2]
    cov2 = [[1.0, -0.5], [-0.5, 1.0]]
    X_class2 = np.random.multivariate_normal(mean2, cov2, n_class2)
    y_class2 = -np.ones(n_class2)
    
    # Combine classes
    X = np.vstack([X_class1, X_class2])
    y = np.hstack([y_class1, y_class2])
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_nonlinear_dataset(n_samples=200, noise=0.15, random_state=42):
    """
    Generate a non-linearly separable dataset for testing polynomial kernel.
    Creates concentric circles pattern.
    
    Parameters
    ----------
    n_samples : int, default=200
        Total number of samples to generate
    noise : float, default=0.15
        Standard deviation of Gaussian noise
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray, shape (n_samples, 2)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Labels (-1 or 1)
    """
    np.random.seed(random_state)
    
    n_class1 = n_samples // 2
    n_class2 = n_samples - n_class1
    
    # Inner circle (class 1)
    theta1 = np.random.rand(n_class1) * 2 * np.pi
    r1 = np.random.rand(n_class1) * 2 + 1
    X_class1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y_class1 = np.ones(n_class1)
    
    # Outer circle (class 2)
    theta2 = np.random.rand(n_class2) * 2 * np.pi
    r2 = np.random.rand(n_class2) * 2 + 4
    X_class2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y_class2 = -np.ones(n_class2)
    
    # Combine
    X = np.vstack([X_class1, X_class2])
    y = np.hstack([y_class1, y_class2])
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def visualize_dataset(X, y, title="Dataset Visualization"):
    """
    Visualize a 2D binary classification dataset.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, 2)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Labels (-1 or 1)
    title : str
        Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Plot class 1
    mask_class1 = (y == 1)
    plt.scatter(X[mask_class1, 0], X[mask_class1, 1], 
                c='blue', marker='o', s=50, alpha=0.7, 
                label='Class 1 (y=1)', edgecolors='k')
    
    # Plot class 2
    mask_class2 = (y == -1)
    plt.scatter(X[mask_class2, 0], X[mask_class2, 1], 
                c='red', marker='s', s=50, alpha=0.7, 
                label='Class 2 (y=-1)', edgecolors='k')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate and visualize Simple Dataset 1
    print("Generating Simple Dataset 1...")
    X, y = generate_simple_dataset_1(n_samples=100, noise=0.1)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Class 1: {np.sum(y == 1)}, Class -1: {np.sum(y == -1)}")
    
    visualize_dataset(X, y, title="Simple Dataset 1 (Binary Classification)")
    
    # Generate and visualize non-linear dataset
    print("\nGenerating Non-linear Dataset...")
    X_nl, y_nl = generate_nonlinear_dataset(n_samples=200, noise=0.15)
    print(f"Dataset shape: {X_nl.shape}")
    print(f"Class distribution: Class 1: {np.sum(y_nl == 1)}, Class -1: {np.sum(y_nl == -1)}")
    
    visualize_dataset(X_nl, y_nl, title="Non-linear Dataset (Concentric Circles)")

