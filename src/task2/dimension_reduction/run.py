"""Generate Task 2 test predictions for TruncatedSVD dimension reduction."""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Dimension-reduction sweep settings.
N_COMPONENTS_LIST = [2000, 1000, 500, 100]
SVD_N_ITER = 7

# KNN classifier settings.
N_NEIGHBORS = 2
KNN_WEIGHTS = "distance"
KNN_METRIC = "cosine"
KNN_ALGORITHM = "brute"
KNN_N_JOBS = 4


def _output_path_for(n_components: int) -> Path:
    return SUBMISSIONS_DIR / f"task2_knn_svd_{n_components}.csv"


def main() -> None:
    overall_start = time.perf_counter()

    print("Loading cleaned train/test data...")
    train_df, test_df = load_train_test_cleaned_data(PROJECT_ROOT)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    train_texts = train_df["cleaned_abstract"].tolist()
    test_texts = test_df["cleaned_abstract"].tolist()
    y_train = train_df["label_id"].to_numpy(dtype=np.int64)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nVectorizing full data with TF-IDF...")
    tfidf_start = time.perf_counter()
    vectorizer = TfidfVectorizer(**TFIDF_DEFAULTS)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    tfidf_elapsed = time.perf_counter() - tfidf_start
    print(f"Train TF-IDF shape: {X_train.shape}")
    print(f"Test TF-IDF shape: {X_test.shape}")
    print(f"TF-IDF vectorization done in: {tfidf_elapsed:.2f}s")

    for n_components in N_COMPONENTS_LIST:
        component_start = time.perf_counter()
        print(f"\n=== Running TruncatedSVD n_components={n_components} ===")

        print("Fitting and applying TruncatedSVD...")
        svd_start = time.perf_counter()
        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=SVD_N_ITER,
        )
        X_train_reduced = svd.fit_transform(X_train).astype(np.float32, copy=False)
        X_test_reduced = svd.transform(X_test).astype(np.float32, copy=False)
        svd_elapsed = time.perf_counter() - svd_start

        print("Normalizing SVD vectors...")
        norm_start = time.perf_counter()
        normalize(X_train_reduced, norm="l2", copy=False)
        normalize(X_test_reduced, norm="l2", copy=False)
        norm_elapsed = time.perf_counter() - norm_start

        print("Training KNN and predicting test labels...")
        knn_start = time.perf_counter()
        clf = KNeighborsClassifier(
            n_neighbors=N_NEIGHBORS,
            weights=KNN_WEIGHTS,
            metric=KNN_METRIC,
            algorithm=KNN_ALGORITHM,
            n_jobs=KNN_N_JOBS,
        )
        clf.fit(X_train_reduced, y_train)
        y_pred = clf.predict(X_test_reduced).astype(np.int64)
        knn_elapsed = time.perf_counter() - knn_start

        submission_df = pd.DataFrame(
            {
                "id": test_df["id"].astype(np.int64),
                "label_id": y_pred,
            }
        )
        if len(submission_df) != len(test_df):
            raise ValueError("Submission row count does not match test set row count.")

        output_path = _output_path_for(n_components)
        submission_df.to_csv(output_path, index=False)
        print(f"Saved submission to: {output_path}")
        print(f"Submission rows: {len(submission_df)}")
        print(
            f"[n_components={n_components}] timings -> "
            f"SVD={svd_elapsed:.2f}s, Normalize={norm_elapsed:.2f}s, "
            f"KNN train+predict={knn_elapsed:.2f}s, "
            f"Total={time.perf_counter() - component_start:.2f}s"
        )

        del X_train_reduced, X_test_reduced, clf, y_pred, submission_df, svd
        gc.collect()

    print(f"\nAll submissions generated in {time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
