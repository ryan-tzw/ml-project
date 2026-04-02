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


class MultiClassLogisticRegression:
    def __init__(self):
        self.models = []
        self.n_classes: int | None = None

    def train(self, X, y, bs=32, epochs=100, lr=0.01):
        """
        Train one-vs-rest classifiers for multi-class classification
        - X: (n_samples, n_features)
        y: (n_samples,) integer labels from 0 to n_classes-1
        """
        self.n_classes = len(np.unique(y))  # Counts unique classes in y
        self.models = []  # List to store binary classifiers (one per class)

        # Each iteration trains a binary classifer for one class
        for c in range(self.n_classes):
            print(f"Training classifier for class {c}...")
            # Convert to binary labels for this class
            y_binary = (y == c).astype(int)
            # Reuses the LogisticRegression class for binary classification
            model = LogisticRegression()
            model.train(X, y_binary, bs=bs, epochs=epochs, lr=lr)
            self.models.append(model)
        print("All classifiers trained!")

    def predict(self, X):
        """
        Predict class labels for X
        """
        if self.n_classes is None:
            raise ValueError("Model has not been trained. Call train(...) first.")

        # Store predicted probabilities for each classifier
        probs = np.zeros((X.shape[0], self.n_classes))
        for c, model in enumerate(self.models):
            # Compute sigmoid of linear input
            linear_output = X @ model.weights + model.bias
            probs[:, c] = model.sigmoid(linear_output)
        # Select class with highest probability
        return np.argmax(probs, axis=1)
