"""Generate Task 2 test predictions for feature-selection experiments."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Feature-selection sweep settings.
FEATURE_SIZES = [2000, 1000, 500, 100]

# KNN classifier settings.
N_NEIGHBORS = 2
KNN_WEIGHTS = "distance"
KNN_METRIC = "cosine"
KNN_ALGORITHM = "brute"
KNN_N_JOBS = 4


def _output_path_for(max_features: int) -> Path:
    return SUBMISSIONS_DIR / f"task2_knn_fs_{max_features}.csv"


def _predict_with_knn(
    X_train,
    y_train: np.ndarray,
    X_test,
) -> np.ndarray:
    model = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights=KNN_WEIGHTS,
        metric=KNN_METRIC,
        algorithm=KNN_ALGORITHM,
        n_jobs=KNN_N_JOBS,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(np.int64)


def main() -> None:
    overall_start = time.perf_counter()

    print("Loading cleaned train/test data...")
    train_df, test_df = load_train_test_cleaned_data(PROJECT_ROOT)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("Model: knn")

    train_texts = train_df["cleaned_abstract"].tolist()
    test_texts = test_df["cleaned_abstract"].tolist()
    y_train = train_df["label_id"].to_numpy(dtype=np.int64)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    for max_features in FEATURE_SIZES:
        component_start = time.perf_counter()
        print(f"\n=== Running max_features={max_features} ===")
        vectorizer_cfg = {**TFIDF_DEFAULTS, "max_features": max_features}
        vectorizer = TfidfVectorizer(**vectorizer_cfg)

        print("Vectorizing text with TF-IDF...")
        tfidf_start = time.perf_counter()
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        tfidf_elapsed = time.perf_counter() - tfidf_start
        print(f"Train TF-IDF shape: {X_train.shape}")
        print(f"Test TF-IDF shape: {X_test.shape}")
        print(f"TF-IDF vectorization done in: {tfidf_elapsed:.2f}s")

        print(f"Training KNN (k={N_NEIGHBORS})...")
        model_start = time.perf_counter()
        y_pred = _predict_with_knn(X_train, y_train, X_test)
        model_elapsed = time.perf_counter() - model_start

        submission_df = pd.DataFrame(
            {
                "id": test_df["id"].astype(np.int64),
                "label_id": y_pred,
            }
        )
        if len(submission_df) != len(test_df):
            raise ValueError("Submission row count does not match test set row count.")

        output_path = _output_path_for(max_features)
        submission_df.to_csv(output_path, index=False)
        print(f"Saved submission to: {output_path}")
        print(f"Submission rows: {len(submission_df)}")
        print(
            f"[max_features={max_features}] timings -> "
            f"TFIDF={tfidf_elapsed:.2f}s, Model={model_elapsed:.2f}s, "
            f"Total={time.perf_counter() - component_start:.2f}s"
        )

    print(f"\nAll submissions generated in {time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
