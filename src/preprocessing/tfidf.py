"""Defaults for sklearn TfidfVectorizer."""

TFIDF_DEFAULTS = {
    "max_features": 5000,
    "min_df": 2,
    "sublinear_tf": True,
}

__all__ = ["TFIDF_DEFAULTS"]
