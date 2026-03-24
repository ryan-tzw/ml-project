"""Logistic Regression implementation from scratch."""

import numpy as np


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_hat):
        """Binary cross-entropy loss"""
        n = len(y)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return -(1 / n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def gradients(self, X, y, y_hat):
        n = len(y)
        dw = (1 / n) * X.T @ (y_hat - y)
        db = (1 / n) * np.sum(y_hat - y)
        return dw, db

    def train(self, X, y, bs, epochs, lr):
        """
        Params:
        - X: Training data, shape (n_samples, n_features)
        - y: Target labels, shape (n_samples,)
        - bs: Batch size
        - epochs: Number of training epochs
        - lr: Learning rate
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):
            # Gradient descent
            for i in range(0, n_samples, bs):
                X_batch = X[i : i + bs]
                y_batch = y[i : i + bs]

                y_hat = self.sigmoid(X_batch @ self.weights + self.bias)
                dw, db = self.gradients(X_batch, y_batch, y_hat)

                self.weights -= lr * dw
                self.bias -= lr * db

    def predict(self, X):
        linear_output = X @ self.weights + self.bias
        y_hat = self.sigmoid(linear_output)
        return (y_hat >= 0.5).astype(int)
