"""Task 3 LinearSVC full-train submission generator."""

import ast
import numbers
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
STAGE1_OUTPUT_PATH = OUTPUT_DIR / "linear_svc_halving_results.csv"
STAGE2_OUTPUT_PATH = OUTPUT_DIR / "linear_svc_finalists_cv5.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "task3_linear_svc.csv"

RANDOM_STATE = 42
LINEAR_SVC_MAX_ITER = 5000


def _parse_param_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return value
        if text.lower() == "none":
            return None
        try:
            return ast.literal_eval(text)
        except ValueError, SyntaxError:
            return value
    return value


def _coerce_param_type(key: str, value: Any) -> Any:
    parsed = _parse_param_value(value)
    if key in {"tfidf__max_features", "tfidf__min_df"}:
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

    if STAGE1_OUTPUT_PATH.exists():
        df_stage1 = pd.read_csv(STAGE1_OUTPUT_PATH)
        if not df_stage1.empty:
            sort_col = (
                "rank_test_score"
                if "rank_test_score" in df_stage1.columns
                else "mean_test_score"
            )
            ascending = sort_col == "rank_test_score"
            best = df_stage1.sort_values(by=sort_col, ascending=ascending).iloc[0]
            params = _extract_param_dict(best)
            print(f"Loaded best params from stage1: {STAGE1_OUTPUT_PATH}")
            return params

    raise FileNotFoundError(
        "No stage results found. Run stage1/stage2 first to select submission params."
    )


def main() -> None:
    overall_start = time.perf_counter()

    print("Loading cleaned train/test data...")
    train_df, test_df = load_train_test_cleaned_data(PROJECT_ROOT)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    best_params = _load_best_params()

    tfidf_params = {
        **TFIDF_DEFAULTS,
        "ngram_range": best_params.get("tfidf__ngram_range", (1, 1)),
        "min_df": best_params.get("tfidf__min_df", TFIDF_DEFAULTS.get("min_df", 2)),
        "sublinear_tf": best_params.get(
            "tfidf__sublinear_tf", TFIDF_DEFAULTS.get("sublinear_tf", True)
        ),
        "max_features": best_params.get(
            "tfidf__max_features", TFIDF_DEFAULTS.get("max_features")
        ),
    }
    clf_params = {
        "C": best_params.get("clf__C", 1.0),
        "class_weight": best_params.get("clf__class_weight", "balanced"),
        "max_iter": LINEAR_SVC_MAX_ITER,
        "random_state": RANDOM_STATE,
    }

    print(f"Using TF-IDF params: {tfidf_params}")
    print(f"Using LinearSVC params: {clf_params}")

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

    print("\nTraining LinearSVC...")
    train_start = time.perf_counter()
    clf = LinearSVC(**clf_params)
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
