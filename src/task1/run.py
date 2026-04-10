from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.data_io import load_train_test_cleaned_data
from src.preprocessing.tfidf import TFIDF_DEFAULTS
from src.task1.logistic_regression import MultiClassLogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 2.0
REG_STRENGTH = 1e-4
OUTPUT_PATH = PROJECT_ROOT / "submissions" / "LogReg_Prediction.csv"


def main() -> None:
    print("Loading data...")
    train_df, test_df = load_train_test_cleaned_data(PROJECT_ROOT)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    X_train_cleaned = train_df["cleaned_abstract"].tolist()
    X_test_cleaned = test_df["cleaned_abstract"].tolist()

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(**TFIDF_DEFAULTS)
    X_train = vectorizer.fit_transform(X_train_cleaned)
    X_test = vectorizer.transform(X_test_cleaned)
    y_train = train_df["label_id"].to_numpy(dtype=np.int64)
    print(f"Train TF-IDF shape: {X_train.shape}")
    print(f"Test TF-IDF shape: {X_test.shape}")

    model = MultiClassLogisticRegression()
    model.train(
        X_train,
        y_train,
        bs=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        reg_strength=REG_STRENGTH,
    )

    print("Predicting test labels...")
    y_pred = model.predict(X_test).astype(np.int64)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(np.int64),
            "label_id": y_pred,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved submission to: {OUTPUT_PATH}")
    print(f"Submission rows: {len(submission_df)}")


if __name__ == "__main__":
    main()
