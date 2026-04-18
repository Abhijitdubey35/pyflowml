"""
TextCleaner — NLP text preprocessing pipeline.
Tokenizes, removes stopwords, and lemmatizes text.
"""

import re
import string
from pyflowml.monitoring.logger import get_logger

logger = get_logger("TextCleaner")


class TextCleaner:
    """
    Text preprocessing pipeline for NLP tasks.

    Steps (configurable):
    1. Lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove punctuation
    5. Remove digits
    6. Remove stopwords (requires nltk)
    7. Lemmatization (requires nltk)

    Example
    -------
    >>> cleaner = TextCleaner(remove_stopwords=True, lemmatize=True)
    >>> clean = cleaner.fit_transform(["Hello World!", "Machine Learning rocks"])
    """

    def __init__(self, lowercase: bool = True, remove_html: bool = True,
                 remove_urls: bool = True, remove_punctuation: bool = True,
                 remove_digits: bool = False, remove_stopwords: bool = True,
                 lemmatize: bool = True, language: str = "english"):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.language = language
        self._stopwords = set()
        self._lemmatizer = None

    def fit(self, texts=None) -> "TextCleaner":
        """Download and load NLTK resources if needed."""
        if self.remove_stopwords or self.lemmatize:
            try:
                import nltk
                if self.remove_stopwords:
                    try:
                        from nltk.corpus import stopwords
                        self._stopwords = set(stopwords.words(self.language))
                    except LookupError:
                        nltk.download("stopwords", quiet=True)
                        from nltk.corpus import stopwords
                        self._stopwords = set(stopwords.words(self.language))

                if self.lemmatize:
                    try:
                        from nltk.stem import WordNetLemmatizer
                        self._lemmatizer = WordNetLemmatizer()
                    except LookupError:
                        nltk.download("wordnet", quiet=True)
                        nltk.download("omw-1.4", quiet=True)
                        from nltk.stem import WordNetLemmatizer
                        self._lemmatizer = WordNetLemmatizer()
            except ImportError:
                logger.warning("nltk not installed. Stopword removal and lemmatization disabled.")
                self.remove_stopwords = False
                self.lemmatize = False
        return self

    def transform(self, texts: list) -> list:
        """Clean a list of text strings."""
        return [self._clean_one(t) for t in texts]

    def fit_transform(self, texts: list) -> list:
        return self.fit().transform(texts)

    def _clean_one(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)

        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+", " ", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.remove_digits:
            text = re.sub(r"\d+", " ", text)

        tokens = text.split()

        if self.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)
