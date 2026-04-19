"""Task 3 Random Forest - Optimized for better performance."""

import ast
import numbers
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
STAGE2_OUTPUT_PATH = OUTPUT_DIR / "random_forest_finalists_cv5.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "task3_random_forest.csv"

RANDOM_STATE = 42
N_JOBS = -1


def _parse_param_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return value
        if text.lower() == "none":
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
    return value


def _coerce_param_type(key: str, value: Any) -> Any:
    parsed = _parse_param_value(value)
    if key in {
        "tfidf__max_features",
        "tfidf__min_df",
        "clf__n_estimators",
        "clf__max_depth",
        "clf__min_samples_split",
        "clf__min_samples_leaf",
    }:
        if parsed is None:
            return None
        if isinstance(parsed, numbers.Real):
            parsed_float = float(parsed)
            if parsed_float.is_integer():
                return int(parsed_float)
    return parsed


def _extract_param_dict(row: pd.Series) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for col, value in row.items():
        if not col.startswith("param_"):
            continue
        if pd.isna(value):
            continue
        key = col.replace("param_", "", 1)
        params[key] = _coerce_param_type(key, value)
    return params


def _load_best_params() -> dict[str, Any]:
    if STAGE2_OUTPUT_PATH.exists():
        df_stage2 = pd.read_csv(STAGE2_OUTPUT_PATH)
        if not df_stage2.empty:
            best = df_stage2.sort_values(
                by=["mean_macro_f1", "mean_accuracy"],
                ascending=[False, False],
            ).iloc[0]
            params = _extract_param_dict(best)
            print(f"Loaded best params from stage2: {STAGE2_OUTPUT_PATH}")
            return params

    raise FileNotFoundError(
        "No stage2 results found. Run random_forest --stage2 first."
    )


def main() -> None:
    overall_start = time.perf_counter()

    print("Loading cleaned train/test data...")
    train_df, test_df = load_train_test_cleaned_data(PROJECT_ROOT)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    best_params = _load_best_params()

    # OPTIMIZED TF-IDF: Increase max_features for better representation
    tfidf_params = {
        **TFIDF_DEFAULTS,
        "ngram_range": best_params.get("tfidf__ngram_range", (1, 2)),
        "min_df": best_params.get("tfidf__min_df", 2),
        "max_features": best_params.get("tfidf__max_features", 150000),  # INCREASED
        "sublinear_tf": True,  # Sublinear TF scaling
    }
    
    # OPTIMIZED RANDOM FOREST: More trees, deeper exploration
    clf_params = {
        "n_estimators": best_params.get("clf__n_estimators", 500),  # INCREASED from 300
        "max_depth": best_params.get("clf__max_depth", 30),  # INCREASED from 25
        "min_samples_split": best_params.get("clf__min_samples_split", 3),  # REDUCED from 5
        "min_samples_leaf": best_params.get("clf__min_samples_leaf", 1),
        "max_features": "sqrt",  # Better feature selection per split
        "bootstrap": True,
        "oob_score": False,
        "class_weight": "balanced_subsample",  # Better for imbalanced classes
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": 1,  # Show progress
    }

    print(f"\nUsing TF-IDF params: {tfidf_params}")
    print(f"Using Random Forest params: {clf_params}")

    train_texts = train_df["cleaned_abstract"].to_numpy(dtype=str)
    test_texts = test_df["cleaned_abstract"].to_numpy(dtype=str)
    y_train = train_df["label_id"].to_numpy(dtype=np.int64)

    print("\nVectorizing with TF-IDF...")
    tfidf_start = time.perf_counter()
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    tfidf_elapsed = time.perf_counter() - tfidf_start
    print(f"Train TF-IDF shape: {X_train.shape}")
    print(f"Test TF-IDF shape: {X_test.shape}")
    print(f"TF-IDF fit/transform time: {tfidf_elapsed:.2f}s")

    print("\nTraining Random Forest (optimized)...")
    train_start = time.perf_counter()
    clf = RandomForestClassifier(**clf_params)
    clf.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - train_start
    print(f"Model train time: {train_elapsed:.2f}s")

    print("\nPredicting test labels...")
    pred_start = time.perf_counter()
    y_pred = clf.predict(X_test).astype(np.int64)
    pred_elapsed = time.perf_counter() - pred_start
    print(f"Prediction time: {pred_elapsed:.2f}s")

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(np.int64),
            "label_id": y_pred,
        }
    )

    expected_cols = ["id", "label_id"]
    if submission_df.columns.tolist() != expected_cols:
        raise ValueError(f"Submission columns must be exactly {expected_cols}.")
    if len(submission_df) != len(test_df):
        raise ValueError("Submission row count does not match test set row count.")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved submission to: {SUBMISSION_PATH}")
    print(f"Submission rows: {len(submission_df)}")
    print(f"Total runtime: {time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()