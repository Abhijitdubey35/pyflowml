"""
Vectorizer — TF-IDF and Bag-of-Words text vectorization.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyflowml.monitoring.logger import get_logger

logger = get_logger("Vectorizer")


class Vectorizer:
    """
    Text vectorization wrapper supporting TF-IDF and Bag-of-Words.

    Parameters
    ----------
    method       : 'tfidf' | 'bow'
    max_features : Maximum number of features (vocabulary size)
    ngram_range  : N-gram range tuple, e.g. (1, 2) for unigrams + bigrams

    Example
    -------
    >>> vec = Vectorizer(method="tfidf", max_features=5000)
    >>> X_train = vec.fit_transform(train_texts)
    >>> X_test  = vec.transform(test_texts)
    """

    def __init__(self, method: str = "tfidf", max_features: int = 10_000,
                 ngram_range: tuple = (1, 1), min_df: int = 2, max_df: float = 0.95):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        if method == "tfidf":
            self._vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,
            )
        elif method == "bow":
            self._vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            )
        else:
            raise ValueError(f"method must be 'tfidf' or 'bow', got '{method}'")

    def fit(self, texts: list) -> "Vectorizer":
        self._vectorizer.fit(texts)
        logger.info(f"Vectorizer fitted | method={self.method} | vocab={len(self.vocabulary_)}")
        return self

    def transform(self, texts: list):
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: list):
        result = self._vectorizer.fit_transform(texts)
        logger.info(f"Vectorizer fitted | method={self.method} | vocab={len(self.vocabulary_)}")
        return result

    @property
    def vocabulary_(self) -> dict:
        return self._vectorizer.vocabulary_

    @property
    def feature_names(self) -> list:
        return self._vectorizer.get_feature_names_out().tolist()
