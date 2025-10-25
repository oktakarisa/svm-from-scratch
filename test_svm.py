"""
Comprehensive Test Script for SVM Implementation
Tests scratch SVM implementation and compares with scikit-learn
Saves results to reports/ folder
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import os
from datetime import datetime

from scratch_svm import ScratchSVMClassifier
from simple_dataset import generate_simple_dataset_1, generate_nonlinear_dataset

# Create reports folder
os.makedirs('reports', exist_ok=True)

# Create report file
report_file = os.path.join('reports', f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
report_lines = []

def log_print(message, to_file=True):
    """Print to console and optionally save to report file"""
    print(message)
    if to_file:
        report_lines.append(message)


def print_section(title):
    """Print formatted section header"""
    log_print("\n" + "="*70)
    log_print(f"  {title}")
    log_print("="*70)


def calculate_metrics(y_true, y_pred, dataset_name="Dataset"):
    """
    Calculate and print classification metrics
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    dataset_name : str
        Name of the dataset for display
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    log_print(f"\n{dataset_name} Metrics:")
    log_print(f"  Accuracy:  {accuracy:.4f}")
    log_print(f"  Precision: {precision:.4f}")
    log_print(f"  Recall:    {recall:.4f}")
    log_print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    log_print(f"\n  Confusion Matrix:")
    log_print(f"                Predicted")
    log_print(f"                -1    1")
    log_print(f"  Actual  -1   {cm[0,0]:4d}  {cm[0,1]:4d}")
    log_print(f"          1    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def test_problem_4_linear_kernel():
    """
    Problem 4: Learning and Estimation on Simple Dataset 1 with Linear Kernel
    """
    print_section("Problem 4: Learning and Estimation on Simple Dataset 1")
    
    # Generate dataset
    log_print("\n[1] Generating Simple Dataset 1...")
    X, y = generate_simple_dataset_1(n_samples=200, noise=0.1, random_state=42)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    log_print(f"    Training samples: {len(X_train)}")
    log_print(f"    Test samples: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Scratch SVM
    log_print("\n[2] Training Scratch SVM (Linear Kernel)...")
    start_time = time.time()
    scratch_svm = ScratchSVMClassifier(
        num_iter=100,
        lr=0.01,
        kernel='linear',
        threshold=1e-5,
        verbose=False  # Set to True for detailed output
    )
    scratch_svm.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    scratch_time = time.time() - start_time
    
    # Predictions
    y_train_pred_scratch = scratch_svm.predict(X_train_scaled).ravel()
    y_test_pred_scratch = scratch_svm.predict(X_test_scaled).ravel()
    
    # Metrics for Scratch SVM
    log_print("\n[3] Scratch SVM Results:")
    log_print(f"    Training time: {scratch_time:.4f} seconds")
    scratch_train_metrics = calculate_metrics(y_train, y_train_pred_scratch, "Training")
    scratch_test_metrics = calculate_metrics(y_test, y_test_pred_scratch, "Test")
    
    # Train scikit-learn SVM
    log_print("\n[4] Training scikit-learn SVM (Linear Kernel)...")
    start_time = time.time()
    sklearn_svm = SVC(kernel='linear', C=1000.0, random_state=42)
    sklearn_svm.fit(X_train_scaled, y_train)
    sklearn_time = time.time() - start_time
    
    # Predictions
    y_train_pred_sklearn = sklearn_svm.predict(X_train_scaled)
    y_test_pred_sklearn = sklearn_svm.predict(X_test_scaled)
    
    # Metrics for scikit-learn SVM
    log_print("\n[5] scikit-learn SVM Results:")
    log_print(f"    Training time: {sklearn_time:.4f} seconds")
    sklearn_train_metrics = calculate_metrics(y_train, y_train_pred_sklearn, "Training")
    sklearn_test_metrics = calculate_metrics(y_test, y_test_pred_sklearn, "Test")
    
    # Comparison
    log_print("\n[6] Comparison Summary:")
    log_print(f"    {'Metric':<15} {'Scratch (Test)':<15} {'sklearn (Test)':<15} {'Difference':<15}")
    log_print(f"    {'-'*60}")
    log_print(f"    {'Accuracy':<15} {scratch_test_metrics['accuracy']:<15.4f} {sklearn_test_metrics['accuracy']:<15.4f} {abs(scratch_test_metrics['accuracy'] - sklearn_test_metrics['accuracy']):<15.4f}")
    log_print(f"    {'Precision':<15} {scratch_test_metrics['precision']:<15.4f} {sklearn_test_metrics['precision']:<15.4f} {abs(scratch_test_metrics['precision'] - sklearn_test_metrics['precision']):<15.4f}")
    log_print(f"    {'Recall':<15} {scratch_test_metrics['recall']:<15.4f} {sklearn_test_metrics['recall']:<15.4f} {abs(scratch_test_metrics['recall'] - sklearn_test_metrics['recall']):<15.4f}")
    log_print(f"    {'F1-Score':<15} {scratch_test_metrics['f1']:<15.4f} {sklearn_test_metrics['f1']:<15.4f} {abs(scratch_test_metrics['f1'] - sklearn_test_metrics['f1']):<15.4f}")
    
    log_print(f"\n    {'Support Vectors':<15} {scratch_svm.n_support_vectors:<15} {len(sklearn_svm.support_):<15}")
    
    return {
        'scratch_svm': scratch_svm,
        'sklearn_svm': sklearn_svm,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }


def test_problem_6_polynomial_kernel():
    """
    Problem 6 (Advanced): Test Polynomial Kernel on Non-linear Dataset
    """
    print_section("Problem 6 (Advanced): Polynomial Kernel on Non-linear Dataset")
    
    # Generate non-linear dataset
    log_print("\n[1] Generating Non-linear Dataset...")
    X, y = generate_nonlinear_dataset(n_samples=200, noise=0.2, random_state=42)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    log_print(f"    Training samples: {len(X_train)}")
    log_print(f"    Test samples: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Scratch SVM with Polynomial Kernel
    log_print("\n[2] Training Scratch SVM (Polynomial Kernel)...")
    start_time = time.time()
    scratch_svm_poly = ScratchSVMClassifier(
        num_iter=150,
        lr=0.005,
        kernel='poly',
        degree=2,
        gamma=1.0,
        theta_0=1.0,
        threshold=1e-5,
        verbose=False  # Set to True for detailed output
    )
    scratch_svm_poly.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    scratch_time = time.time() - start_time
    
    # Predictions
    y_train_pred_scratch = scratch_svm_poly.predict(X_train_scaled).ravel()
    y_test_pred_scratch = scratch_svm_poly.predict(X_test_scaled).ravel()
    
    # Metrics for Scratch SVM
    log_print("\n[3] Scratch SVM (Polynomial) Results:")
    log_print(f"    Training time: {scratch_time:.4f} seconds")
    scratch_train_metrics = calculate_metrics(y_train, y_train_pred_scratch, "Training")
    scratch_test_metrics = calculate_metrics(y_test, y_test_pred_scratch, "Test")
    
    # Train scikit-learn SVM with Polynomial Kernel
    log_print("\n[4] Training scikit-learn SVM (Polynomial Kernel)...")
    start_time = time.time()
    sklearn_svm_poly = SVC(kernel='poly', degree=2, gamma=1.0, coef0=1.0, C=1000.0, random_state=42)
    sklearn_svm_poly.fit(X_train_scaled, y_train)
    sklearn_time = time.time() - start_time
    
    # Predictions
    y_train_pred_sklearn = sklearn_svm_poly.predict(X_train_scaled)
    y_test_pred_sklearn = sklearn_svm_poly.predict(X_test_scaled)
    
    # Metrics for scikit-learn SVM
    log_print("\n[5] scikit-learn SVM (Polynomial) Results:")
    log_print(f"    Training time: {sklearn_time:.4f} seconds")
    sklearn_train_metrics = calculate_metrics(y_train, y_train_pred_sklearn, "Training")
    sklearn_test_metrics = calculate_metrics(y_test, y_test_pred_sklearn, "Test")
    
    # Comparison
    log_print("\n[6] Comparison Summary:")
    log_print(f"    {'Metric':<15} {'Scratch (Test)':<15} {'sklearn (Test)':<15} {'Difference':<15}")
    log_print(f"    {'-'*60}")
    log_print(f"    {'Accuracy':<15} {scratch_test_metrics['accuracy']:<15.4f} {sklearn_test_metrics['accuracy']:<15.4f} {abs(scratch_test_metrics['accuracy'] - sklearn_test_metrics['accuracy']):<15.4f}")
    log_print(f"    {'Precision':<15} {scratch_test_metrics['precision']:<15.4f} {sklearn_test_metrics['precision']:<15.4f} {abs(scratch_test_metrics['precision'] - sklearn_test_metrics['precision']):<15.4f}")
    log_print(f"    {'Recall':<15} {scratch_test_metrics['recall']:<15.4f} {sklearn_test_metrics['recall']:<15.4f} {abs(scratch_test_metrics['recall'] - sklearn_test_metrics['recall']):<15.4f}")
    log_print(f"    {'F1-Score':<15} {scratch_test_metrics['f1']:<15.4f} {sklearn_test_metrics['f1']:<15.4f} {abs(scratch_test_metrics['f1'] - sklearn_test_metrics['f1']):<15.4f}")
    
    return {
        'scratch_svm': scratch_svm_poly,
        'sklearn_svm': sklearn_svm_poly,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }


def test_hyperparameter_comparison():
    """
    Additional Test: Compare different hyperparameters
    """
    print_section("Additional Test: Hyperparameter Comparison")
    
    # Generate dataset
    X, y = generate_simple_dataset_1(n_samples=150, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different learning rates
    log_print("\n[1] Testing Different Learning Rates:")
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    
    for lr in learning_rates:
        log_print(f"\n    Learning Rate = {lr}")
        svm = ScratchSVMClassifier(num_iter=100, lr=lr, kernel='linear', 
                                   threshold=1e-5, verbose=False)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        log_print(f"    Test Accuracy: {accuracy:.4f}, Support Vectors: {svm.n_support_vectors}")
    
    # Test different thresholds
    log_print("\n[2] Testing Different Threshold Values:")
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3]
    
    for thresh in thresholds:
        log_print(f"\n    Threshold = {thresh}")
        svm = ScratchSVMClassifier(num_iter=100, lr=0.01, kernel='linear', 
                                   threshold=thresh, verbose=False)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        log_print(f"    Test Accuracy: {accuracy:.4f}, Support Vectors: {svm.n_support_vectors}")


if __name__ == "__main__":
    log_print("\n" + "="*70)
    log_print("  SVM FROM SCRATCH - ASSIGNMENT TEST SUITE")
    log_print("="*70)
    log_print(f"\n  Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run assignment tests
    results_linear = test_problem_4_linear_kernel()
    results_poly = test_problem_6_polynomial_kernel()
    test_hyperparameter_comparison()
    
    log_print("\n" + "="*70)
    log_print("  ALL TESTS COMPLETED SUCCESSFULLY!")
    log_print("="*70)
    log_print("\n  Next step: Run 'python visualize_svm.py' to see decision boundaries.")
    log_print(f"\n  Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save report to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n  Report saved to: {report_file}")
    print("="*70)

