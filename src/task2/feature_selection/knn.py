"""Task 2 feature-size CV experiments using KNN (n_neighbors=2)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "task2" / "feature_selection_knn_cv_results.csv"
)

FEATURE_SIZES = [2000, 1000, 500, 100]
N_SPLITS = 5
RANDOM_STATE = 42
N_NEIGHBORS = 2
KNN_WEIGHTS = "distance"
KNN_METRIC = "cosine"
KNN_ALGORITHM = "brute"


def run_cv() -> pd.DataFrame:
    overall_start = time.perf_counter()

    print("Loading cleaned train data...")
    train_df, _ = load_train_test_cleaned_data(PROJECT_ROOT)
    y = train_df["label_id"].to_numpy(dtype=np.int64)
    cleaned_texts = train_df["cleaned_abstract"].tolist()
    print(f"Train shape: {train_df.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(cleaned_texts, y))

    results: list[dict[str, float | int]] = []
    size_iter = tqdm(FEATURE_SIZES, desc="Feature sizes", unit="size")
    for max_features in size_iter:
        size_iter.set_description(f"Feature sizes (current: {max_features})")
        fold_f1_scores: list[float] = []
        fold_acc_scores: list[float] = []
        fold_train_times: list[float] = []
        fold_total_times: list[float] = []

        fold_iter = tqdm(
            enumerate(fold_indices, start=1),
            total=N_SPLITS,
            desc=f"Folds @ {max_features}",
            unit="fold",
            leave=False,
        )
        for fold_idx, (train_idx, val_idx) in fold_iter:
            fold_start = time.perf_counter()

            train_texts = [cleaned_texts[i] for i in train_idx]
            val_texts = [cleaned_texts[i] for i in val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            vectorizer_cfg = {**TFIDF_DEFAULTS, "max_features": max_features}
            vectorizer = TfidfVectorizer(**vectorizer_cfg)
            X_train = vectorizer.fit_transform(train_texts)
            X_val = vectorizer.transform(val_texts)

            clf = KNeighborsClassifier(
                n_neighbors=N_NEIGHBORS,
                weights=KNN_WEIGHTS,
                metric=KNN_METRIC,
                algorithm=KNN_ALGORITHM,
            )
            train_start = time.perf_counter()
            clf.fit(X_train, y_train)
            train_elapsed = time.perf_counter() - train_start
            fold_train_times.append(train_elapsed)

            y_pred = clf.predict(X_val).astype(np.int64)
            fold_f1 = float(f1_score(y_val, y_pred, average="macro"))
            fold_acc = float(accuracy_score(y_val, y_pred))
            fold_f1_scores.append(fold_f1)
            fold_acc_scores.append(fold_acc)

            fold_elapsed = time.perf_counter() - fold_start
            fold_total_times.append(fold_elapsed)
            tqdm.write(
                f"[max_features={max_features}] Fold {fold_idx}/{N_SPLITS} "
                f"Macro F1={fold_f1:.4f}, Accuracy={fold_acc:.4f}, "
                f"Train={train_elapsed:.2f}s, Total={fold_elapsed:.2f}s"
            )

        mean_f1 = float(np.mean(fold_f1_scores))
        std_f1 = float(np.std(fold_f1_scores))
        mean_acc = float(np.mean(fold_acc_scores))
        std_acc = float(np.std(fold_acc_scores))
        mean_train_time = float(np.mean(fold_train_times))
        mean_fold_time = float(np.mean(fold_total_times))

        results.append(
            {
                "model": "knn",
                "max_features": int(max_features),
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_train_time_sec": mean_train_time,
                "mean_fold_time_sec": mean_fold_time,
            }
        )
        tqdm.write(
            f"[max_features={max_features}] summary -> "
            f"mean F1={mean_f1:.4f}, mean Acc={mean_acc:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["mean_macro_f1", "mean_accuracy"],
        ascending=[False, False],
    )
    results_df = results_df.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Task 2 Feature-Size CV Summary (KNN) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(f"Total runtime: {time.perf_counter() - overall_start:.2f}s")
    return results_df


def main() -> None:
    run_cv()


if __name__ == "__main__":
    main()
