"""
Visualization Script for SVM Decision Boundaries
Generates plots and saves them to plots/ folder
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segmentation faults
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

from scratch_svm import ScratchSVMClassifier
from simple_dataset import generate_simple_dataset_1, generate_nonlinear_dataset


def plot_decision_boundary(model, X, y, title, filename, is_scratch=True):
    """
    Plot decision boundary and support vectors, save to plots folder
    
    Parameters
    ----------
    model : SVM classifier
        Trained SVM model
    X : ndarray, shape (n_samples, 2)
        Feature matrix (scaled)
    y : ndarray, shape (n_samples,)
        Labels
    title : str
        Plot title
    filename : str
        Filename to save (will be saved in plots/)
    is_scratch : bool
        Whether the model is scratch implementation
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions
    if is_scratch:
        Z = model.predict(mesh_points).ravel()
        decision_values = model.decision_function(mesh_points)
    else:
        Z = model.predict(mesh_points)
        decision_values = model.decision_function(mesh_points)
    
    Z = Z.reshape(xx.shape)
    decision_values = decision_values.reshape(xx.shape)
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], 
                colors=['#FFAAAA', '#AAAAFF'])
    
    # Plot decision boundary (f(x) = 0)
    ax.contour(xx, yy, decision_values, levels=[0], 
               colors='black', linewidths=2, linestyles='solid')
    
    # Plot margins (f(x) = ±1)
    ax.contour(xx, yy, decision_values, levels=[-1, 1], 
               colors='black', linewidths=1, linestyles='dashed', alpha=0.7)
    
    # Plot data points
    mask_class1 = (y == 1)
    mask_class2 = (y == -1)
    
    ax.scatter(X[mask_class1, 0], X[mask_class1, 1], 
               c='blue', marker='o', s=50, alpha=0.8, 
               label='Class +1', edgecolors='k', linewidth=1)
    ax.scatter(X[mask_class2, 0], X[mask_class2, 1], 
               c='red', marker='s', s=50, alpha=0.8, 
               label='Class -1', edgecolors='k', linewidth=1)
    
    # Highlight support vectors
    if is_scratch:
        if model.n_support_vectors > 0:
            sv_indices = model.index_support_vectors
            ax.scatter(X[sv_indices, 0], X[sv_indices, 1], 
                      s=200, linewidth=2.5, facecolors='none', 
                      edgecolors='yellow', 
                      label=f'Support Vectors (n={model.n_support_vectors})')
    else:
        sv_indices = model.support_
        ax.scatter(X[sv_indices, 0], X[sv_indices, 1], 
                  s=200, linewidth=2.5, facecolors='none', 
                  edgecolors='yellow', 
                  label=f'Support Vectors (n={len(sv_indices)})')
    
    ax.set_xlabel('Feature 1 (Standardized)', fontsize=12)
    ax.set_ylabel('Feature 2 (Standardized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to plots folder
    os.makedirs('plots', exist_ok=True)
    filepath = os.path.join('plots', filename)
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Warning: Could not save {filepath}: {e}")
    finally:
        plt.close(fig)
        plt.clf()  # Clear the current figure


def visualize_linear_kernel():
    """
    Visualize decision boundaries for linear kernel (Problem 5)
    """
    print("\n" + "="*70)
    print("  PROBLEM 5: Visualization of Decision Area (Linear Kernel)")
    print("="*70)
    
    # Generate dataset
    print("\n[1] Generating Simple Dataset 1...")
    X, y = generate_simple_dataset_1(n_samples=100, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Scratch SVM
    print("\n[2] Training Scratch SVM (Linear Kernel)...")
    scratch_svm = ScratchSVMClassifier(
        num_iter=100, lr=0.01, kernel='linear', 
        threshold=1e-5, verbose=False
    )
    scratch_svm.fit(X_train_scaled, y_train)
    print(f"    Support Vectors: {scratch_svm.n_support_vectors}")
    
    # Train sklearn SVM
    print("\n[3] Training sklearn SVM (Linear Kernel)...")
    sklearn_svm = SVC(kernel='linear', C=1000.0)
    sklearn_svm.fit(X_train_scaled, y_train)
    print(f"    Support Vectors: {len(sklearn_svm.support_)}")
    
    # Generate plots
    print("\n[4] Generating visualizations...")
    plot_decision_boundary(
        scratch_svm, X_train_scaled, y_train,
        "Scratch SVM - Linear Kernel\nDecision Boundary and Support Vectors",
        "scratch_svm_linear.png",
        is_scratch=True
    )
    
    plot_decision_boundary(
        sklearn_svm, X_train_scaled, y_train,
        "scikit-learn SVM - Linear Kernel\nDecision Boundary and Support Vectors",
        "sklearn_svm_linear.png",
        is_scratch=False
    )


def visualize_polynomial_kernel():
    """
    Visualize decision boundaries for polynomial kernel (Problem 6 - Advanced)
    """
    print("\n" + "="*70)
    print("  PROBLEM 6 (Advanced): Polynomial Kernel on Non-linear Dataset")
    print("="*70)
    
    # Generate non-linear dataset
    print("\n[1] Generating Non-linear Dataset...")
    X, y = generate_nonlinear_dataset(n_samples=150, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Scratch SVM with Polynomial Kernel
    print("\n[2] Training Scratch SVM (Polynomial Kernel)...")
    scratch_svm_poly = ScratchSVMClassifier(
        num_iter=150, lr=0.005, kernel='poly', 
        degree=2, gamma=1.0, theta_0=1.0,
        threshold=1e-5, verbose=False
    )
    scratch_svm_poly.fit(X_train_scaled, y_train)
    print(f"    Support Vectors: {scratch_svm_poly.n_support_vectors}")
    
    # Train sklearn SVM with Polynomial Kernel
    print("\n[3] Training sklearn SVM (Polynomial Kernel)...")
    sklearn_svm_poly = SVC(kernel='poly', degree=2, gamma=1.0, coef0=1.0, C=1000.0)
    sklearn_svm_poly.fit(X_train_scaled, y_train)
    print(f"    Support Vectors: {len(sklearn_svm_poly.support_)}")
    
    # Generate plots
    print("\n[4] Generating visualizations...")
    plot_decision_boundary(
        scratch_svm_poly, X_train_scaled, y_train,
        "Scratch SVM - Polynomial Kernel (degree=2)\nDecision Boundary and Support Vectors",
        "scratch_svm_polynomial.png",
        is_scratch=True
    )
    
    plot_decision_boundary(
        sklearn_svm_poly, X_train_scaled, y_train,
        "scikit-learn SVM - Polynomial Kernel (degree=2)\nDecision Boundary and Support Vectors",
        "sklearn_svm_polynomial.png",
        is_scratch=False
    )


def create_comparison_plot():
    """
    Create side-by-side comparison of linear vs polynomial kernel
    """
    print("\n" + "="*70)
    print("  ADDITIONAL: Kernel Comparison")
    print("="*70)
    
    # Generate dataset
    X, y = generate_simple_dataset_1(n_samples=100, noise=0.15, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train with Linear Kernel
    print("\n[1] Training with Linear Kernel...")
    svm_linear = ScratchSVMClassifier(
        num_iter=100, lr=0.01, kernel='linear', 
        threshold=1e-5, verbose=False
    )
    svm_linear.fit(X_scaled, y)
    
    # Train with Polynomial Kernel
    print("\n[2] Training with Polynomial Kernel...")
    svm_poly = ScratchSVMClassifier(
        num_iter=100, lr=0.01, kernel='poly', 
        degree=2, gamma=1.0, theta_0=1.0,
        threshold=1e-5, verbose=False
    )
    svm_poly.fit(X_scaled, y)
    
    # Create side-by-side plot
    print("\n[3] Generating comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Linear Kernel
    ax = axes[0]
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_linear.predict(mesh_points).ravel().reshape(xx.shape)
    decision_values = svm_linear.decision_function(mesh_points).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['#FFAAAA', '#AAAAFF'])
    ax.contour(xx, yy, decision_values, levels=[0], colors='black', linewidths=2, linestyles='solid')
    ax.contour(xx, yy, decision_values, levels=[-1, 1], colors='black', linewidths=1, linestyles='dashed', alpha=0.7)
    
    mask1 = (y == 1)
    mask2 = (y == -1)
    ax.scatter(X_scaled[mask1, 0], X_scaled[mask1, 1], c='blue', marker='o', s=50, alpha=0.8, label='Class +1', edgecolors='k')
    ax.scatter(X_scaled[mask2, 0], X_scaled[mask2, 1], c='red', marker='s', s=50, alpha=0.8, label='Class -1', edgecolors='k')
    
    if svm_linear.n_support_vectors > 0:
        sv_indices = svm_linear.index_support_vectors
        ax.scatter(X_scaled[sv_indices, 0], X_scaled[sv_indices, 1], s=200, linewidth=2.5, 
                  facecolors='none', edgecolors='yellow', label=f'Support Vectors (n={svm_linear.n_support_vectors})')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'Linear Kernel', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot Polynomial Kernel
    ax = axes[1]
    Z = svm_poly.predict(mesh_points).ravel().reshape(xx.shape)
    decision_values = svm_poly.decision_function(mesh_points).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['#FFAAAA', '#AAAAFF'])
    ax.contour(xx, yy, decision_values, levels=[0], colors='black', linewidths=2, linestyles='solid')
    ax.contour(xx, yy, decision_values, levels=[-1, 1], colors='black', linewidths=1, linestyles='dashed', alpha=0.7)
    
    ax.scatter(X_scaled[mask1, 0], X_scaled[mask1, 1], c='blue', marker='o', s=50, alpha=0.8, label='Class +1', edgecolors='k')
    ax.scatter(X_scaled[mask2, 0], X_scaled[mask2, 1], c='red', marker='s', s=50, alpha=0.8, label='Class -1', edgecolors='k')
    
    if svm_poly.n_support_vectors > 0:
        sv_indices = svm_poly.index_support_vectors
        ax.scatter(X_scaled[sv_indices, 0], X_scaled[sv_indices, 1], s=200, linewidth=2.5, 
                  facecolors='none', edgecolors='yellow', label=f'Support Vectors (n={svm_poly.n_support_vectors})')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'Polynomial Kernel (degree=2)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    filepath = os.path.join('plots', 'kernel_comparison.png')
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Warning: Could not save {filepath}: {e}")
    finally:
        plt.close(fig)
        plt.clf()  # Clear the current figure


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SVM VISUALIZATION - GENERATING DECISION BOUNDARY PLOTS")
    print("="*70)
    
    # Generate all visualizations
    visualize_linear_kernel()
    visualize_polynomial_kernel()
    create_comparison_plot()
    
    print("\n" + "="*70)
    print("  ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print(f"\nPlots saved in 'plots/' folder:")
    print("  - scratch_svm_linear.png")
    print("  - sklearn_svm_linear.png")
    print("  - scratch_svm_polynomial.png")
    print("  - sklearn_svm_polynomial.png")
    print("  - kernel_comparison.png")
    print("="*70)

