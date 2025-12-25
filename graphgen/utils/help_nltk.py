from functools import lru_cache
import os
from typing import Dict, List, Final, Optional
import warnings
import nltk

warnings.filterwarnings(
    "ignore", 
    category=UserWarning,
    module="jieba\._compat"
)


import jieba

class NLTKHelper:
    """
    NLTK helper class
    """

    SUPPORTED_LANGUAGES: Final[Dict[str, str]] = {
        "en": "english",
        "zh": "chinese"
    }
    _NLTK_PACKAGES: Final[Dict[str, str]] = {
        "stopwords": "corpora",
        "punkt_tab": "tokenizers"
    }

    def __init__(self, nltk_data_path: Optional[str] = None):
        self._nltk_path = nltk_data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "resources", 
            "nltk_data"
        )
        nltk.data.path.append(self._nltk_path)
        jieba.initialize()

        self._ensure_nltk_data("stopwords")
        self._ensure_nltk_data("punkt_tab")

    def _ensure_nltk_data(self, package_name: str) -> None:
        """
        ensure nltk data is downloaded
        """
        try:
            nltk.data.find(f"{self._NLTK_PACKAGES[package_name]}/{package_name}")
        except LookupError:
            nltk.download(package_name, download_dir=self._nltk_path, quiet=True)

    @lru_cache(maxsize=2)
    def get_stopwords(self, lang: str) -> List[str]:
        if lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {lang} is not supported.")
        return nltk.corpus.stopwords.words(self.SUPPORTED_LANGUAGES[lang])

    def word_tokenize(self, text: str, lang: str) -> List[str]:
        if lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {lang} is not supported.")
        if lang == "zh":
            return jieba.lcut(text)

        return nltk.word_tokenize(text)
