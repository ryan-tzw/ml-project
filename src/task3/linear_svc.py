"""Task 3 LinearSVC search workflow (stage1 halving + stage2 finalist CV)."""

import argparse
import ast
import numbers
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
STAGE1_OUTPUT_PATH = OUTPUT_DIR / "linear_svc_halving_results.csv"
STAGE2_OUTPUT_PATH = OUTPUT_DIR / "linear_svc_finalists_cv5.csv"

# Shared CV/data settings.
RANDOM_STATE = 42
N_JOBS = 2
SCORING = {"macro_f1": "f1_macro", "accuracy": "accuracy"}

# Stage 1 search settings.
STAGE1_CV_SPLITS = 3
STAGE1_FACTOR = 3

# Stage 2 finalist confirmation settings.
STAGE2_CV_SPLITS = 5
STAGE2_TOP_K_DEFAULT = 5


def build_pipeline() -> Pipeline:
    """Build TF-IDF + LinearSVC pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_DEFAULTS)),
            (
                "clf",
                LinearSVC(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _stage1_param_grid() -> dict[str, list[Any]]:
    return {
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__min_df": [2, 5],
        "tfidf__max_features": [75000, 100000, 150000, 200000, 300000],
        "clf__C": [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0],
    }


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


def _load_stage1_top_params(top_k: int) -> list[dict[str, Any]]:
    if not STAGE1_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing stage1 results: {STAGE1_OUTPUT_PATH}. Run --stage1 first."
        )

    df = pd.read_csv(STAGE1_OUTPUT_PATH)
    if df.empty:
        raise ValueError("Stage1 results CSV is empty.")

    sort_col = (
        "rank_test_score" if "rank_test_score" in df.columns else "mean_test_score"
    )
    ascending = sort_col == "rank_test_score"
    top_df = df.sort_values(by=sort_col, ascending=ascending).head(top_k)

    param_cols = [col for col in top_df.columns if col.startswith("param_")]
    if not param_cols:
        raise ValueError("No parameter columns found in stage1 results.")

    params_list: list[dict[str, Any]] = []
    for _, row in top_df.iterrows():
        params: dict[str, Any] = {}
        for col in param_cols:
            key = col.replace("param_", "", 1)
            value = row[col]
            if pd.isna(value):
                continue
            params[key] = _coerce_param_type(key, value)
        params_list.append(params)

    return params_list


def run_stage1_halving_grid() -> pd.DataFrame:
    """Run coarse hyperparameter search with HalvingGridSearchCV."""
    stage_start = time.perf_counter()

    print("Loading cleaned train data for stage 1...")
    train_df, _ = load_train_test_cleaned_data(PROJECT_ROOT)
    X = train_df["cleaned_abstract"].to_numpy(dtype=str)
    y = train_df["label_id"].to_numpy(dtype=np.int64)
    print(f"Train shape: {train_df.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    cv = StratifiedKFold(
        n_splits=STAGE1_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = HalvingGridSearchCV(
        estimator=build_pipeline(),
        param_grid=_stage1_param_grid(),
        cv=cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
        factor=STAGE1_FACTOR,
        error_score="raise",
        refit=True,
    )

    print("Running stage 1 HalvingGridSearchCV...")
    search.fit(X, y)
    stage_elapsed = time.perf_counter() - stage_start

    results_df = pd.DataFrame(search.cv_results_).sort_values(
        by=["rank_test_score", "mean_test_score"],
        ascending=[True, False],
    )
    results_df = results_df.reset_index(drop=True)
    results_df["mean_fit_time_sec"] = results_df["mean_fit_time"].astype(float)
    results_df["mean_score_time_sec"] = results_df["mean_score_time"].astype(float)
    results_df["stage1_total_runtime_sec"] = stage_elapsed

    required_cols = {
        "mean_test_score",
        "mean_fit_time_sec",
        "mean_score_time_sec",
        "stage1_total_runtime_sec",
    }
    missing = sorted(required_cols.difference(results_df.columns))
    if missing:
        raise ValueError(f"Stage1 results missing required columns: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(STAGE1_OUTPUT_PATH, index=False)

    print("\n=== Stage 1 Summary (HalvingGridSearchCV) ===")
    print(
        results_df[
            [
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
                "mean_score_time",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nBest params: {search.best_params_}")
    print(f"Saved stage1 results to: {STAGE1_OUTPUT_PATH}")
    print(f"Stage1 total runtime: {stage_elapsed:.2f}s")
    return results_df


def run_stage2_cv5_finalists(top_k: int = STAGE2_TOP_K_DEFAULT) -> pd.DataFrame:
    """Re-evaluate top-k stage1 candidates with 5-fold CV."""
    stage_start = time.perf_counter()

    print("Loading cleaned train data for stage 2...")
    train_df, _ = load_train_test_cleaned_data(PROJECT_ROOT)
    X = train_df["cleaned_abstract"].to_numpy(dtype=str)
    y = train_df["label_id"].to_numpy(dtype=np.int64)
    print(f"Train shape: {train_df.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    top_params = _load_stage1_top_params(top_k=top_k)
    cv = StratifiedKFold(
        n_splits=STAGE2_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, Any]] = []
    for idx, params in enumerate(top_params, start=1):
        print(f"\n=== Stage 2 candidate {idx}/{len(top_params)} ===")
        pipe = build_pipeline()
        pipe.set_params(**params)

        cv_result = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            cv=cv,
            scoring=SCORING,
            n_jobs=N_JOBS,
            return_train_score=False,
            error_score="raise",
        )

        row: dict[str, Any] = {
            "candidate_rank_stage1": idx,
            "mean_macro_f1": float(np.mean(cv_result["test_macro_f1"])),
            "std_macro_f1": float(np.std(cv_result["test_macro_f1"])),
            "mean_accuracy": float(np.mean(cv_result["test_accuracy"])),
            "std_accuracy": float(np.std(cv_result["test_accuracy"])),
            "mean_fit_time_sec": float(np.mean(cv_result["fit_time"])),
            "mean_score_time_sec": float(np.mean(cv_result["score_time"])),
        }
        for key, value in params.items():
            row[f"param_{key}"] = value
        rows.append(row)
        print(
            f"[candidate {idx}/{len(top_params)}] "
            f"mean_macro_f1={row['mean_macro_f1']:.6f}, "
            f"std_macro_f1={row['std_macro_f1']:.6f}, "
            f"mean_accuracy={row['mean_accuracy']:.6f}, "
            f"std_accuracy={row['std_accuracy']:.6f}, "
            f"mean_fit_time_sec={row['mean_fit_time_sec']:.2f}, "
            f"mean_score_time_sec={row['mean_score_time_sec']:.2f}"
        )

    stage_elapsed = time.perf_counter() - stage_start
    results_df = pd.DataFrame(rows).sort_values(
        by=["mean_macro_f1", "mean_accuracy"],
        ascending=[False, False],
    )
    results_df = results_df.reset_index(drop=True)
    results_df["stage2_total_runtime_sec"] = stage_elapsed

    required_cols = {
        "mean_macro_f1",
        "mean_accuracy",
        "mean_fit_time_sec",
        "mean_score_time_sec",
        "stage2_total_runtime_sec",
    }
    missing = sorted(required_cols.difference(results_df.columns))
    if missing:
        raise ValueError(f"Stage2 results missing required columns: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(STAGE2_OUTPUT_PATH, index=False)

    print("\n=== Stage 2 Summary (Top Finalists, 5-fold CV) ===")
    print(
        results_df[
            [
                "mean_macro_f1",
                "std_macro_f1",
                "mean_accuracy",
                "std_accuracy",
                "mean_fit_time_sec",
                "mean_score_time_sec",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved stage2 results to: {STAGE2_OUTPUT_PATH}")
    print(f"Stage2 total runtime: {stage_elapsed:.2f}s")
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3 LinearSVC search workflow with staged CV."
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Run stage1 HalvingGridSearchCV (3-fold).",
    )
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Run stage2 finalist confirmation (5-fold).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=STAGE2_TOP_K_DEFAULT,
        help=f"Number of stage1 finalists to re-evaluate in stage2 (default: {STAGE2_TOP_K_DEFAULT}).",
    )
    args = parser.parse_args()

    if not args.stage1 and not args.stage2:
        parser.error("Pass at least one flag: --stage1 and/or --stage2")
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer.")

    if args.stage1:
        run_stage1_halving_grid()
    if args.stage2:
        run_stage2_cv5_finalists(top_k=args.top_k)


if __name__ == "__main__":
    main()
