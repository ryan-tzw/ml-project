"""Task 3 XGBoost hyperparameter optimization using Optuna."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
OPTUNA_RESULTS_PATH = OUTPUT_DIR / "xgboost_optuna_best.csv"

def objective(trial, X, y):
    """The function Optuna tries to maximize."""
    # 1. Define the search space for this trial
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.7),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "multi:softmax",
        "tree_method": "hist",
        "random_state": 42,
        # Leave fold-level parallelism to sklearn to avoid nested CPU oversubscription.
        "n_jobs": 1,
    }
    
    # 2. Add TF-IDF tuning (built into the trial)
    max_features = trial.suggest_categorical("max_features", [80000, 120000, 150000])

    # 3. Vectorization
    tfidf_params = {
        **TFIDF_DEFAULTS,
        "max_features": max_features,
        "sublinear_tf": True,
    }
    tfidf = TfidfVectorizer(**tfidf_params)
    X_transformed = tfidf.fit_transform(X)
    
    # 4. Cross-Validation
    clf = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # We use Macro F1 as the metric to maximize
    scores = cross_val_score(clf, X_transformed, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    
    return scores.mean()

def run_optimization(n_trials: int = 20, sample_frac: float = 0.5) -> None:
    print("Loading data...")
    train_df, _ = load_train_test_cleaned_data(PROJECT_ROOT)

    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be between 0 and 1.")

    if sample_frac < 1.0:
        print(f"Sampling {sample_frac:.0%} of the training set for Optuna...")
        train_df = train_df.sample(frac=sample_frac, random_state=42)

    X = train_df["cleaned_abstract"].values.astype(str)
    le = LabelEncoder()
    y = le.fit_transform(train_df["label_id"])

    print(f"Starting Optuna Study ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("\nBest Trial Results:")
    print(f"  Macro F1: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_df = pd.DataFrame([study.best_params])
    best_df.to_csv(OPTUNA_RESULTS_PATH, index=False)
    print(f"Saved best parameters to: {OPTUNA_RESULTS_PATH}")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna tuning for the Task 3 XGBoost model.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to run.")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.5,
        help="Fraction of the training set to use during tuning (0 < value <= 1).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_optimization(n_trials=args.trials, sample_frac=args.sample_frac)