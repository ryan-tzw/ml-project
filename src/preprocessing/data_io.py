"""Shared data-loading helpers for the competition datasets."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_train_test_data(
    project_root: Path | None = None,
    sep: str = "\t",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSV files and validate required columns."""
    root = project_root or PROJECT_ROOT
    train_path = root / "data" / "train.csv"
    test_path = root / "data" / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing required file: {test_path}")

    train_df = pd.read_csv(train_path, sep=sep)
    test_df = pd.read_csv(test_path, sep=sep)

    required_train_cols = {"id", "abstract", "label_id"}
    required_test_cols = {"id", "abstract"}
    if not required_train_cols.issubset(train_df.columns):
        raise ValueError(
            f"train.csv must contain columns: {sorted(required_train_cols)}"
        )
    if not required_test_cols.issubset(test_df.columns):
        raise ValueError(f"test.csv must contain columns: {sorted(required_test_cols)}")

    return train_df, test_df


__all__ = ["PROJECT_ROOT", "load_train_test_data"]
