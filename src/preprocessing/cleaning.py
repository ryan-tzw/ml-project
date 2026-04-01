"""Text cleaning utilities for preprocessing"""

from dataclasses import dataclass, field
from pathlib import Path
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for text cleaning and token normalization."""

    remove_stopwords: bool = True
    remove_single_char_tokens: bool = True
    use_stemming: bool = False
    use_lemmatization: bool = True
    preserve_hyphen_underscore: bool = True
    extra_stopwords: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.use_stemming and self.use_lemmatization:
            raise ValueError("Choose either stemming or lemmatization, not both.")


class TextCleaner:
    """Deterministic text cleaner for academic abstracts."""

    def __init__(self, cfg: CleaningConfig | None = None) -> None:
        self.cfg = cfg or CleaningConfig()
        self._ensure_required_nltk_data()
        self._stemmer = PorterStemmer() if self.cfg.use_stemming else None
        self._lemmatizer = WordNetLemmatizer() if self.cfg.use_lemmatization else None
        self._stop_words = (
            self._load_stop_words() if self.cfg.remove_stopwords else set()
        )

    def _ensure_required_nltk_data(self) -> None:
        resources: dict[str, str] = {}
        if self.cfg.remove_stopwords:
            resources["stopwords"] = "corpora/stopwords"
        if self.cfg.use_lemmatization:
            resources["wordnet"] = "corpora/wordnet"

        for resource_name, resource_path in resources.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                downloaded = bool(
                    nltk.download(
                        resource_name,
                        download_dir=str(NLTK_DATA_DIR),
                        quiet=True,
                    )
                )
                if not downloaded:
                    raise LookupError(
                        f"NLTK resource '{resource_name}' is missing and could not be downloaded. "
                        f"Expected local data dir: {NLTK_DATA_DIR}"
                    )

    def _load_stop_words(self) -> set[str]:
        try:
            words = set(stopwords.words("english"))
        except LookupError as exc:
            raise LookupError(
                "NLTK stopwords resource not found. "
                f"Expected local data dir: {NLTK_DATA_DIR}"
            ) from exc
        return words | set(self.cfg.extra_stopwords)

    @staticmethod
    def _normalize_non_string(text: object) -> str:
        if isinstance(text, str):
            return text
        return ""

    def clean_text(self, text: object) -> str:
        """Apply regex-based normalization only."""
        text = self._normalize_non_string(text)
        if not text:
            return ""

        text = text.lower()

        # Remove LaTeX blocks and commands.
        text = re.sub(r"\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", " ", text, flags=re.DOTALL)
        text = re.sub(
            r"\\(?:begin|end)\{[^{}]*\}|\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*",
            " ",
            text,
        )

        # Remove URLs.
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove numeric citations
        text = re.sub(r"\[[0-9,\s]+\]", " ", text)

        if self.cfg.preserve_hyphen_underscore:
            text = re.sub(r"[^a-z0-9\-_ ]", " ", text)
        else:
            text = re.sub(r"[^a-z0-9 ]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: object) -> str:
        """Apply full preprocessing pipeline: clean + tokenize + normalize."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""

        tokens = cleaned.split()

        if self.cfg.remove_stopwords:
            tokens = [token for token in tokens if token not in self._stop_words]

        if self.cfg.remove_single_char_tokens:
            tokens = [token for token in tokens if len(token) > 1]

        if self._stemmer is not None:
            tokens = [self._stemmer.stem(token) for token in tokens]
        elif self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(token) for token in tokens]

        return " ".join(tokens)


__all__ = [
    "CleaningConfig",
    "TextCleaner",
]
