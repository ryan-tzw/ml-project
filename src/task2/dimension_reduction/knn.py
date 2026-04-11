"""Task 2 dimension-reduction CV using sklearn Pipeline + cross_validate."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from tqdm.auto import tqdm

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "task2"
    / "dimension_reduction_knn_cv_results_sklearn.csv"
)

# Dimension-reduction sweep settings.
N_COMPONENTS_LIST = [2000, 1000, 500, 100]
SVD_N_ITER = 7

# Cross-validation settings.
N_SPLITS = 5
RANDOM_STATE = 42
CV_N_JOBS = 4

# KNN classifier settings.
N_NEIGHBORS = 2
KNN_WEIGHTS = "distance"
KNN_METRIC = "cosine"
KNN_ALGORITHM = "brute"


def _build_pipeline(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_DEFAULTS)),
            (
                "svd",
                TruncatedSVD(
                    n_components=n_components,
                    n_iter=SVD_N_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("normalize", Normalizer(norm="l2")),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=N_NEIGHBORS,
                    weights=KNN_WEIGHTS,
                    metric=KNN_METRIC,
                    algorithm=KNN_ALGORITHM,
                    n_jobs=1,
                ),
            ),
        ]
    )


def run_cv() -> pd.DataFrame:
    overall_start = time.perf_counter()

    print("Loading cleaned train data...")
    train_df, _ = load_train_test_cleaned_data(PROJECT_ROOT)
    X = train_df["cleaned_abstract"].to_numpy(dtype=str)
    y = train_df["label_id"].to_numpy(dtype=np.int64)
    print(f"Train shape: {train_df.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"macro_f1": "f1_macro", "accuracy": "accuracy"}

    results: list[dict[str, float | int]] = []
    for n_components in tqdm(N_COMPONENTS_LIST, desc="SVD components", unit="size"):
        print(f"\n=== Running sklearn CV n_components={n_components} ===")
        pipe = _build_pipeline(n_components)

        cv_result = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=CV_N_JOBS,
            return_train_score=False,
            error_score="raise",
        )

        macro_f1_scores = cv_result["test_macro_f1"]
        accuracy_scores = cv_result["test_accuracy"]
        fit_times = cv_result["fit_time"]
        score_times = cv_result["score_time"]

        mean_f1 = float(np.mean(macro_f1_scores))
        std_f1 = float(np.std(macro_f1_scores))
        mean_acc = float(np.mean(accuracy_scores))
        std_acc = float(np.std(accuracy_scores))
        mean_train_time = float(np.mean(fit_times))
        mean_fold_time = float(np.mean(fit_times + score_times))

        results.append(
            {
                "n_components": int(n_components),
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_train_time_sec": mean_train_time,
                "mean_fold_time_sec": mean_fold_time,
            }
        )

        print(
            f"[n_components={n_components}] summary -> "
            f"mean F1={mean_f1:.4f}, mean Acc={mean_acc:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["mean_macro_f1", "mean_accuracy"],
        ascending=[False, False],
    )
    results_df = results_df.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Task 2 Dimension-Reduction CV Summary (sklearn Pipeline + CV) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(f"Total runtime: {time.perf_counter() - overall_start:.2f}s")
    return results_df


def main() -> None:
    run_cv()


if __name__ == "__main__":
    main()
