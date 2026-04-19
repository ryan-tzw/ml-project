"""
Task 3: SVD + Logistic Regression — Hyperparameter Search
----------------------------------------------------------
Method: GridSearchCV with cross-validation over key parameters.
Results are saved to a CSV for reporting in your final submission.

Hyperparameters explored:
  - SVD n_components : [100, 200, 300, 500]
  - LogReg C         : [0.1, 1.0, 5.0, 10.0]
  - LogReg solver    : ['lbfgs', 'saga']  (saga scales better at high dims)

Metric: macro F1 (matches Kaggle evaluation)
CV folds: 3  (increase to 5 if you have time)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2]

TRAIN_ABSTRACTS = DATA_DIR / "data/train_abstracts.csv"
TRAIN_LABELS    = DATA_DIR / "data/train_labels.csv"
TEST_ABSTRACTS  = DATA_DIR / "data/test_abstracts.csv"
TEST_IDS        = DATA_DIR / "data/test_ids.csv"

RESULTS_PATH    = DATA_DIR / "src/task3/svd_logreg_hyperparam_results.csv"
SUBMISSION_PATH = DATA_DIR / "submissions/SVD_LogReg_Prediction.csv"

RANDOM_STATE    = 42


print("Checking paths...")
missing = []
for name, path in [
    ("train_abstracts", TRAIN_ABSTRACTS),
    ("train_labels",    TRAIN_LABELS),
    ("test_abstracts",  TEST_ABSTRACTS),
    ("test_ids",        TEST_IDS),
]:
    status = "OK" if path.exists() else "MISSING"
    print(f"  [{status}] {path}")
    if status == "MISSING":
        missing.append(name)
 
# Check output folders exist (create if not)
for name, path in [
    ("results",    RESULTS_PATH.parent),
    ("submission", SUBMISSION_PATH.parent),
]:
    path.mkdir(parents=True, exist_ok=True)
    print(f"  [OK - dir ready] {path}")
 
if missing:
    raise FileNotFoundError(
        f"\nAborting — these input files were not found: {missing}\n"
        f"Project root detected as: {DATA_DIR}\n"
        "Check that DATA_DIR points to the ml-project root."
    )
 
print("\nAll paths OK — safe to start training.\n")


# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
train_abstracts = pd.read_csv(TRAIN_ABSTRACTS)["abstract"].fillna("").tolist()
train_labels    = pd.read_csv(TRAIN_LABELS)["label_id"].values
test_abstracts  = pd.read_csv(TEST_ABSTRACTS)["abstract"].fillna("").tolist()
test_ids        = pd.read_csv(TEST_IDS)["id"].values

print(f"  Train: {len(train_abstracts)} samples, {len(np.unique(train_labels))} classes")

# ── 2. Build base pipeline ─────────────────────────────────────────────────────
# Fixed: TF-IDF settings are kept constant (already validated in preprocessing)
base_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),
        min_df       = 2,
        max_df       = 0.8,
        sublinear_tf = True,
        stop_words   = "english",
    )),
    ("svd",        TruncatedSVD(random_state=RANDOM_STATE)),
    ("normalizer", Normalizer(copy=False)),
    ("logreg",     LogisticRegression(
        multi_class = "multinomial",
        max_iter    = 1000,
        random_state= RANDOM_STATE,
        n_jobs      = -1,
    )),
])

# ── 3. Hyperparameter grid ─────────────────────────────────────────────────────
# Prefix matches pipeline step names above
param_grid = {
    "svd__n_components": [100, 200, 300, 500],   # LSA dimensionality
    "logreg__C":         [0.1, 1.0, 5.0, 10.0],  # regularisation strength
    "logreg__solver":    ["lbfgs", "saga"],        # optimisation algorithm
}

# ── 4. Cross-validated grid search ────────────────────────────────────────────
macro_f1 = make_scorer(f1_score, average="macro")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator  = base_pipeline,
    param_grid = param_grid,
    scoring    = macro_f1,
    cv         = cv,
    n_jobs     = -1,         # parallelise across folds
    verbose    = 2,          # print progress
    refit      = True,       # refit best model on full training set
    return_train_score = True,
)

print(f"\nStarting grid search ({len(param_grid['svd__n_components']) * len(param_grid['logreg__C']) * len(param_grid['logreg__solver'])} combos × 3 folds)...")
print("Estimated time: 15–40 min depending on hardware.\n")

t0 = time.time()
grid_search.fit(train_abstracts, train_labels)
elapsed = time.time() - t0

print(f"\nGrid search complete in {elapsed/60:.1f} min")
print(f"Best CV Macro F1 : {grid_search.best_score_:.4f}")
print(f"Best params      : {grid_search.best_params_}")

# ── 5. Save all results for reporting ─────────────────────────────────────────
results_df = pd.DataFrame(grid_search.cv_results_)

# Keep only the columns useful for your report
report_cols = [
    "param_svd__n_components",
    "param_logreg__C",
    "param_logreg__solver",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "rank_test_score",
    "mean_fit_time",
]
report_df = results_df[report_cols].copy()
report_df.columns = [
    "svd_components", "C", "solver",
    "val_macro_f1_mean", "val_macro_f1_std",
    "train_macro_f1_mean",
    "rank",
    "mean_fit_time_sec",
]
report_df = report_df.sort_values("rank")
report_df.to_csv(RESULTS_PATH, index=False)

print(f"\nFull results saved → {RESULTS_PATH}")
print("\nTop 5 configurations:")
print(report_df.head(5).to_string(index=False))

# ── 6. Predict on test set using best model ────────────────────────────────────
print("\nPredicting test set with best model...")
test_preds = grid_search.predict(test_abstracts)

submission = pd.DataFrame({"id": test_ids, "label_id": test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved → {SUBMISSION_PATH}")
