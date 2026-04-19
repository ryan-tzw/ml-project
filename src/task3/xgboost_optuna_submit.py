"""Task 3 XGBoost full-train submission generator.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS

# Path configuration
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
OPTUNA_RESULTS_PATH = OUTPUT_DIR / "xgboost_optuna_best.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "task3_xgboost_optuna_final.csv"

RANDOM_STATE = 42

def _load_best_params() -> dict[str, Any]:
    """Loads the specific winning parameters found in your CSV."""
    if OPTUNA_RESULTS_PATH.exists():
        df = pd.read_csv(OPTUNA_RESULTS_PATH)
        print(f"Loaded Winning Params: {df.iloc[0].to_dict()}")
        return df.iloc[0].to_dict()
    
    raise FileNotFoundError(f"Could not find {OPTUNA_RESULTS_PATH}. Run Optuna first!")

def main() -> None:
    overall_start = time.perf_counter()

    print("Loading cleaned train/test data (100% dataset)...")
    train_df, test_df = load_train_test_cleaned_data(PROJECT_ROOT)
    best_params = _load_best_params()

    # Isolating text from ID to satisfy strict non-ID feature usage
    train_texts = train_df["cleaned_abstract"].values.astype(str)
    test_texts = test_df["cleaned_abstract"].values.astype(str)
    test_ids = test_df["id"].values.astype(np.int64) 

    # XGBoost requires consecutive labels starting at 0
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label_id"])

    # Create a new config dictionary to avoid "multiple values" TypeError
    tfidf_config = TFIDF_DEFAULTS.copy()
    tfidf_config.update({
        "max_features": int(best_params["max_features"]),
        "sublinear_tf": True
    })

    print(f"Vectorizing with max_features={tfidf_config['max_features']}...")
    vectorizer = TfidfVectorizer(**tfidf_config)
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Training using the architecture discovered by search
    print(f"Training final XGBoost model with {int(best_params['n_estimators'])} trees...")
    clf = xgb.XGBClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        learning_rate=float(best_params["learning_rate"]),
        subsample=float(best_params.get("subsample", 0.6387861501068111)),
        colsample_bytree=float(best_params.get("colsample_bytree", 0.5153186462482722)),
        min_child_weight=int(best_params.get("min_child_weight", 2)),
        objective="multi:softmax",
        tree_method="hist", 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    print("Predicting test labels...")
    y_pred_encoded = clf.predict(X_test)
    y_pred_original = le.inverse_transform(y_pred_encoded)

    submission_df = pd.DataFrame({
        "id": test_ids,
        "label_id": y_pred_original.astype(np.int64),
    })

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nSUCCESS! Final Submission saved to: {SUBMISSION_PATH}")
    print(f"Total runtime: {(time.perf_counter() - overall_start)/60:.2f} minutes")

if __name__ == "__main__":
    main()