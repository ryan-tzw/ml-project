"""Shared data-loading helpers for the competition datasets."""

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.preprocessing.cleaning import CleaningConfig, TextCleaner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_SEP = "\t"


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


def load_train_test_cleaned_data(
    project_root: Path | None = None,
    cleaning_cfg: CleaningConfig | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test data and ensure cleaned_abstract is available.

    If cached cleaned data exists, load it; otherwise clean raw data and cache it.
    Returned/cached frames intentionally drop raw ``abstract``.
    """
    root = project_root or PROJECT_ROOT
    cfg = cleaning_cfg or CleaningConfig()
    cache_dir = root / "data" / "cache"
    train_cache = cache_dir / "train_cleaned.csv"
    test_cache = cache_dir / "test_cleaned.csv"
    cache_meta = cache_dir / "cleaning_cache_meta.json"

    def _read_cache(cache_path: Path) -> pd.DataFrame:
        # Preferred format going forward is tab-separated. Fallback keeps
        # compatibility with older comma-separated cache files.
        try:
            return pd.read_csv(cache_path, sep=CACHE_SEP)
        except Exception:
            return pd.read_csv(cache_path)

    if use_cache and not refresh_cache and train_cache.exists() and test_cache.exists():
        train_df = _read_cache(train_cache)
        test_df = _read_cache(test_cache)
        required_train_cols = {"id", "label_id", "cleaned_abstract"}
        required_test_cols = {"id", "cleaned_abstract"}
        if required_train_cols.issubset(
            train_df.columns
        ) and required_test_cols.issubset(test_df.columns):
            train_df = train_df[["id", "label_id", "cleaned_abstract"]]
            test_df = test_df[["id", "cleaned_abstract"]]
            return train_df, test_df

    train_df, test_df = load_train_test_data(project_root=root)
    cleaner = TextCleaner(cfg)

    train_cleaned = [
        cleaner.preprocess(text)
        for text in tqdm(train_df["abstract"], desc="Cleaning train")
    ]
    test_cleaned = [
        cleaner.preprocess(text)
        for text in tqdm(test_df["abstract"], desc="Cleaning test")
    ]

    train_df = train_df[["id", "label_id"]].copy()
    test_df = test_df[["id"]].copy()
    train_df["cleaned_abstract"] = train_cleaned
    test_df["cleaned_abstract"] = test_cleaned

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_cache, index=False, sep=CACHE_SEP)
        test_df.to_csv(test_cache, index=False, sep=CACHE_SEP)
        cfg_dict = asdict(cfg)
        cfg_dict["extra_stopwords"] = sorted(cfg_dict["extra_stopwords"])
        cache_meta.write_text(
            json.dumps({"cleaning_config": cfg_dict}, indent=2),
            encoding="utf-8",
        )

    return train_df, test_df


__all__ = [
    "PROJECT_ROOT",
    "CACHE_DIR",
    "load_train_test_data",
    "load_train_test_cleaned_data",
]
