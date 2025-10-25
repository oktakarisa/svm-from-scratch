"""
Scratch Implementation of Support Vector Machine (SVM) Classifier
Hard Margin SVM for Binary Classification
"""

import numpy as np


class ScratchSVMClassifier():
    """
    Scratch implementation of SVM classifier

    Parameters
    ----------
    num_iter : int
      Number of iterations
    lr : float
      Learning rate
    kernel : str
      Kernel type. Linear kernel ('linear') or polynomial kernel ('poly')
    threshold : float
      Threshold for choosing a support vector
    verbose : bool
      True to output the learning process
    gamma : float
      Gamma parameter for polynomial kernel (default=1.0)
    theta_0 : float
      Theta_0 parameter for polynomial kernel (default=0.0)
    degree : int
      Degree parameter for polynomial kernel (default=2)

    Attributes
    ----------
    self.n_support_vectors : int
      Number of support vectors
    self.index_support_vectors : ndarray, shape (n_support_vectors,)
      Support vector index
    self.X_sv : ndarray, shape (n_support_vectors, n_features)
      Support vector features
    self.lam_sv : ndarray, shape (n_support_vectors, 1)
      Support vector Lagrange multipliers
    self.y_sv : ndarray, shape (n_support_vectors, 1)
      Support vector label
    """

    def __init__(self, num_iter, lr, kernel='linear', threshold=1e-5, verbose=False,
                 gamma=1.0, theta_0=0.0, degree=2):
        # Record hyperparameters as attributes
        self.iter = num_iter
        self.lr = lr
        self.kernel = kernel
        self.threshold = threshold
        self.verbose = verbose
        
        # Polynomial kernel parameters
        self.gamma = gamma
        self.theta_0 = theta_0
        self.degree = degree
        
        # Initialize support vector attributes
        self.n_support_vectors = 0
        self.index_support_vectors = None
        self.X_sv = None
        self.lam_sv = None
        self.y_sv = None
        
        # Store training data for kernel computations
        self.X_train = None
        self.y_train = None
        self.lam = None

    def _linear_kernel(self, X1, X2):
        """
        Linear kernel function: K(x_i, x_j) = x_i^T * x_j
        
        Parameters
        ----------
        X1 : ndarray, shape (n_samples_1, n_features)
        X2 : ndarray, shape (n_samples_2, n_features)
        
        Returns
        -------
        K : ndarray, shape (n_samples_1, n_samples_2)
            Kernel matrix
        """
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1, X2):
        """
        Polynomial kernel function: K(x_i, x_j) = (gamma * x_i^T * x_j + theta_0)^d
        
        Parameters
        ----------
        X1 : ndarray, shape (n_samples_1, n_features)
        X2 : ndarray, shape (n_samples_2, n_features)
        
        Returns
        -------
        K : ndarray, shape (n_samples_1, n_samples_2)
            Kernel matrix
        """
        return (self.gamma * np.dot(X1, X2.T) + self.theta_0) ** self.degree
    
    def _kernel_function(self, X1, X2):
        """
        Compute kernel function based on the selected kernel type
        
        Parameters
        ----------
        X1 : ndarray, shape (n_samples_1, n_features)
        X2 : ndarray, shape (n_samples_2, n_features)
        
        Returns
        -------
        K : ndarray, shape (n_samples_1, n_samples_2)
            Kernel matrix
        """
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}. Use 'linear' or 'poly'.")

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Learn the SVM classifier. If verification data is input, 
        the accuracy for it is also calculated for each iteration.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features of training data
        y : ndarray, shape (n_samples,)
            Correct answer value of training data (should be -1 or 1)
        X_val : ndarray, shape (n_samples, n_features), optional
            Features of verification data
        y_val : ndarray, shape (n_samples,), optional
            Correct value of verification data
        """
        # Store training data
        self.X_train = X
        self.y_train = y.reshape(-1, 1)  # Make sure y is column vector
        
        n_samples = X.shape[0]
        
        # Initialize Lagrange multipliers
        self.lam = np.zeros((n_samples, 1))
        
        # Precompute kernel matrix for efficiency
        K = self._kernel_function(X, X)
        
        # Training loop - Gradient Descent on Lagrange multipliers
        for iteration in range(self.iter):
            # Update each Lagrange multiplier
            for i in range(n_samples):
                # Compute: sum_{j=1}^{n} lambda_j * y_i * y_j * K(x_i, x_j)
                sum_term = 0
                for j in range(n_samples):
                    sum_term += self.lam[j] * self.y_train[i] * self.y_train[j] * K[i, j]
                
                # Update rule: lambda_i^new = lambda_i + alpha * (1 - sum_term)
                self.lam[i] = self.lam[i] + self.lr * (1 - sum_term)
                
                # Constraint: lambda_i >= 0
                if self.lam[i] < 0:
                    self.lam[i] = 0
            
            # Output training process if verbose
            if self.verbose and (iteration % 10 == 0 or iteration == self.iter - 1):
                # Calculate training accuracy
                train_pred = self.predict(X)
                train_accuracy = np.mean(train_pred.ravel() == y.ravel())
                
                output_str = f"Iteration {iteration + 1}/{self.iter}: Train Accuracy = {train_accuracy:.4f}"
                
                # Calculate validation accuracy if validation data provided
                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_accuracy = np.mean(val_pred.ravel() == y_val.ravel())
                    output_str += f", Val Accuracy = {val_accuracy:.4f}"
                
                print(output_str)
        
        # Determine support vectors (Problem 2)
        self._determine_support_vectors()
        
        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Number of support vectors: {self.n_support_vectors}/{n_samples}")

    def _determine_support_vectors(self):
        """
        Determine support vectors based on threshold.
        A sample is a support vector if lambda_i > threshold.
        """
        # Find indices where lambda > threshold
        support_vector_indices = np.where(self.lam.ravel() > self.threshold)[0]
        
        self.index_support_vectors = support_vector_indices
        self.n_support_vectors = len(support_vector_indices)
        
        # Store support vector data
        self.X_sv = self.X_train[support_vector_indices]
        self.lam_sv = self.lam[support_vector_indices]
        self.y_sv = self.y_train[support_vector_indices]

    def predict(self, X):
        """
        Estimate the label using the SVM classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Sample

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Estimated result by SVM classifier
        """
        # Check if model has been trained
        if self.lam is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # If support vectors haven't been determined, use all training data
        if self.X_sv is None or self.n_support_vectors == 0:
            # Use all training samples
            X_support = self.X_train
            lam_support = self.lam
            y_support = self.y_train
        else:
            # Use only support vectors
            X_support = self.X_sv
            lam_support = self.lam_sv
            y_support = self.y_sv
        
        # Compute kernel between test samples and support vectors
        K = self._kernel_function(X, X_support)
        
        # Compute f(x) = sum_{n=1}^{N} lambda_n * y_sv_n * K(x, s_n)
        # K shape: (n_test_samples, n_support_vectors)
        # lam_support shape: (n_support_vectors, 1)
        # y_support shape: (n_support_vectors, 1)
        
        # Element-wise multiplication and sum
        f_x = np.sum(K * (lam_support * y_support).T, axis=1)
        
        # Classification: sign of f(x)
        y_pred = np.sign(f_x)
        
        # Handle zero case (assign to class 1)
        y_pred[y_pred == 0] = 1
        
        return y_pred.reshape(-1, 1)

    def decision_function(self, X):
        """
        Compute the decision function (f(x)) for visualization purposes.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Sample
        
        Returns
        -------
        f_x : ndarray, shape (n_samples,)
            Decision function values
        """
        # Use support vectors if available, otherwise all training data
        if self.X_sv is None or self.n_support_vectors == 0:
            X_support = self.X_train
            lam_support = self.lam
            y_support = self.y_train
        else:
            X_support = self.X_sv
            lam_support = self.lam_sv
            y_support = self.y_sv
        
        # Compute kernel
        K = self._kernel_function(X, X_support)
        
        # Compute f(x)
        f_x = np.sum(K * (lam_support * y_support).T, axis=1)
        
        return f_x

