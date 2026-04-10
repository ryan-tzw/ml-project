"""Logistic Regression implementation from scratch."""

import numpy as np
from tqdm.auto import tqdm


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

    def gradients(self, X, y, y_hat, reg_strength, sample_weight):
        """
        sample_weight applies per-row class weights (pos_w/neg_w) to errors.
        This helps prevent the majority class from dominating updates.
        """
        weight_sum = float(np.sum(sample_weight))
        error = (y_hat - y) * sample_weight
        dw = (1 / weight_sum) * X.T @ error + reg_strength * self.weights
        db = (1 / weight_sum) * np.sum(error)
        return dw, db

    def train(
        self,
        X,
        y,
        bs,
        epochs,
        lr,
        reg_strength=1e-4,
    ):
        """
        Params:
        - X: Training data, shape (n_samples, n_features)
        - y: Target labels, shape (n_samples,)
        - bs: Batch size
        - epochs: Number of training epochs
        - lr: Learning rate
        - reg_strength: L2 regularization strength (weights only)
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_arr = np.asarray(y)

        # Compute class weights for balanced updates
        pos_count = int(np.sum(y_arr == 1))
        neg_count = n_samples - pos_count
        if pos_count == 0 or neg_count == 0:
            raise ValueError(
                "Cannot compute balanced class weights when one class is missing."
            )
        pos_w = n_samples / (2.0 * pos_count)
        neg_w = n_samples / (2.0 * neg_count)

        for _ in range(epochs):
            # Shuffle each epoch so batches are not biased.
            order = np.random.permutation(n_samples)
            X_epoch = X[order]
            y_epoch = y_arr[order]

            # Gradient descent
            for i in range(0, n_samples, bs):
                X_batch = X_epoch[i : i + bs]
                y_batch = y_epoch[i : i + bs]
                sample_weight = np.where(y_batch == 1, pos_w, neg_w).astype(np.float64)

                y_hat = self.sigmoid(X_batch @ self.weights + self.bias)
                dw, db = self.gradients(
                    X_batch,
                    y_batch,
                    y_hat,
                    reg_strength,
                    sample_weight=sample_weight,
                )

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

    def train(
        self,
        X,
        y,
        bs=32,
        epochs=100,
        lr=0.01,
        reg_strength=1e-4,
    ):
        """
        Train one-vs-rest classifiers for multi-class classification
        - X: (n_samples, n_features)
        y: (n_samples,) integer labels from 0 to n_classes-1
        """
        self.n_classes = len(np.unique(y))  # Counts unique classes in y
        self.models = []  # List to store binary classifiers (one per class)

        # Each iteration trains a binary classifer for one class
        for c in tqdm(range(self.n_classes), desc="Training OvR classifiers"):
            # Convert to binary labels for this class
            y_binary = (y == c).astype(int)
            # Reuses the LogisticRegression class for binary classification
            model = LogisticRegression()
            model.train(
                X,
                y_binary,
                bs=bs,
                epochs=epochs,
                lr=lr,
                reg_strength=reg_strength,
            )
            self.models.append(model)

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
