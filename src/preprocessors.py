"""Reusable Scikit-Learn Compatible Preprocessors for Movie Reviews Emotion Classification.

Each component corresponds to an assigned group member:
- Member 1 (IT25102549): TextCleaner (stateless contraction expansion, HTML, URLs, lowercase, non-alpha)
- Member 2 (IT25102550): StopwordFilter (stateless stopword removal using sklearn English + movie domain stops)
- Member 3 (IT25102631): GenreBinarizer (learns genre vocabulary in fit, handles unseen genres gracefully)
- Member 4 (IT25102877): WordCountIQRCapper + RobustScaler & MinMaxScaler for numerical features
- Member 5 (IT25103066): compute_train_class_weights (balanced weights from training labels only)
- Member 6 (IT25103132): TF-IDF feature extraction (unigrams/bigrams, no redundant stop_words) and optional TruncatedSVD

Also provides `build_full_preprocessor()` composing all components into an end-to-end ColumnTransformer.
"""

import ast
import html
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data_prep import WordCountIQRCapper

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'s": " is", "'re": " are", "'ve": " have", "'d": " would",
    "'ll": " will", "'m": " am", "it's": "it is"
}

MOVIE_DOMAIN_STOPS = {
    'movie', 'movies', 'film', 'films', 'watch', 'watching',
    'one', 'like', 'really', 'see', 'saw', 'story', 'time',
    'character', 'characters', 'scene', 'scenes'
}

ALL_STOPS = set(ENGLISH_STOP_WORDS).union(MOVIE_DOMAIN_STOPS)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Member 1 (IT25102549): Stateless text sanitation transformer.
    Expands contractions, strips HTML entities, tags, URLs, lowercase, non-alpha characters.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _clean_single(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        for c, exp in CONTRACTIONS.items():
            text = text.replace(c, exp)
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)
        return s.apply(self._clean_single).to_numpy()


class StopwordFilter(BaseEstimator, TransformerMixin):
    """Member 2 (IT25102550): Stateless stopword removal.
    Filters standard English stopwords plus movie domain stopwords for tokens > 2 characters.
    """

    def __init__(self, stop_words: Optional[set] = None, min_token_len: int = 2):
        self.stop_words = stop_words if stop_words is not None else ALL_STOPS
        self.min_token_len = min_token_len

    def fit(self, X, y=None):
        return self

    def _filter_single(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        tokens = text.split()
        return " ".join([t for t in tokens if t not in self.stop_words and len(t) > self.min_token_len])

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)
        return s.apply(self._filter_single).to_numpy()


class GenreBinarizer(BaseEstimator, TransformerMixin):
    """Member 3 (IT25102631): Categorical multi-label genre binarizer.
    Learns genre vocabulary during fit() on training data.
    Produces zero-filled columns for unseen genres encountered during transform().
    """

    def __init__(self):
        pass

    @staticmethod
    def _parse_genres(g_str: Union[str, list]) -> List[str]:
        if pd.isna(g_str):
            return []
        if isinstance(g_str, list):
            return [str(g).strip().lower().replace(" ", "_").replace("-", "_") for g in g_str if str(g).strip()]
        try:
            parsed = ast.literal_eval(str(g_str))
            if isinstance(parsed, list):
                return [str(g).strip().lower().replace(" ", "_").replace("-", "_") for g in parsed if str(g).strip()]
        except Exception:
            pass
        return [g.strip(" '[]\"").lower().replace(" ", "_").replace("-", "_") for g in str(g_str).split(",") if g.strip(" '[]\"")]

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)

        all_genres = set()
        for row in s:
            for g in self._parse_genres(row):
                if g:
                    all_genres.add(g)

        self.genres_ = sorted(list(all_genres))
        self.genre_to_idx_ = {g: i for i, g in enumerate(self.genres_)}
        return self

    def transform(self, X):
        if not hasattr(self, "genres_"):
            raise NotFittedError("GenreBinarizer instance is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)

        n_samples = len(s)
        n_features = len(self.genres_)
        matrix = np.zeros((n_samples, n_features), dtype=np.int32)

        for row_idx, item in enumerate(s):
            parsed = self._parse_genres(item)
            for g in parsed:
                if g in self.genre_to_idx_:
                    col_idx = self.genre_to_idx_[g]
                    matrix[row_idx, col_idx] = 1
                # Unseen genres are ignored, producing all-zero entries for them

        return matrix

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "genres_"):
            raise NotFittedError("GenreBinarizer instance is not fitted yet.")
        return np.array([f"genre_{g}" for g in self.genres_])


class WordCountExtractor(BaseEstimator, TransformerMixin):
    """Helper transformer to compute word counts from review texts."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(X)
        counts = s.apply(lambda x: len(str(x).split())).to_numpy().reshape(-1, 1)
        return counts

    def get_feature_names_out(self, input_features=None):
        return np.array(["word_count"])


def compute_train_class_weights(y_train: pd.Series) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Member 5 (IT25103066): Compute balanced class weights strictly from training labels.

    Returns:
    --------
    Tuple[Dict[str, float], pd.DataFrame]:
        - class_weights_dict: {class_name: weight_float}
        - summary_df: DataFrame summarizing classes, counts, shares, and loss penalty weights.
    """
    classes = np.array(np.unique(y_train), dtype=str)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.to_numpy())
    class_weights_dict = {cls: round(float(w), 4) for cls, w in zip(classes, weights)}

    counts = y_train.value_counts()
    summary_df = pd.DataFrame({
        "Emotion": counts.index,
        "Train Samples": counts.values,
        "Share (%)": (counts.values / len(y_train) * 100).round(2),
        "Balanced Loss Penalty Multiplier": [class_weights_dict[cls] for cls in counts.index]
    }).reset_index(drop=True)

    return class_weights_dict, summary_df


def build_full_preprocessor(use_svd: bool = False, n_svd_components: int = 100) -> ColumnTransformer:
    """Build ONE end-to-end ColumnTransformer composing all six member preprocessing stages.

    Takes the load_clean() DataFrame containing raw/base columns ('Reviews', 'genres', 'Ratings')
    and returns an engineered numerical feature matrix. Runs end-to-end without manual pre-steps.

    Parameters:
    -----------
    use_svd : bool, default=False
        If True, appends TruncatedSVD dimensionality reduction after TF-IDF vectorization.
        Default is False (TF-IDF only) per group consensus.
    n_svd_components : int, default=100
        Number of latent semantic components if use_svd=True.

    Returns:
    --------
    ColumnTransformer
        Skikit-learn ColumnTransformer composing text, genre, word count, and rating pipelines.
    """
    # 1. Text Semantics Pipeline (Member 1 -> Member 2 -> Member 6)
    text_steps = [
        ("cleaner", TextCleaner()),
        ("stopword_filter", StopwordFilter()),
        ("tfidf", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), sublinear_tf=True, min_df=3))
    ]
    if use_svd:
        text_steps.append(("svd", TruncatedSVD(n_components=n_svd_components, random_state=42)))

    text_pipeline = Pipeline(text_steps)

    # 2. Genre Pipeline (Member 3)
    genre_pipeline = Pipeline([
        ("binarizer", GenreBinarizer())
    ])

    # 3. Word Count Pipeline (Member 4)
    # Cleans review text -> computes word count -> caps outliers using training Tukey fences -> RobustScaler
    word_count_pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("wc_extractor", WordCountExtractor()),
        ("capper", WordCountIQRCapper(factor=1.5)),
        ("scaler", RobustScaler())
    ])

    # 4. Rating Scaler (Member 4)
    rating_pipeline = Pipeline([
        ("scaler", MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text_features", text_pipeline, "Reviews"),
            ("genre_features", genre_pipeline, "genres"),
            ("word_count_features", word_count_pipeline, "Reviews"),
            ("rating_features", rating_pipeline, ["Ratings"])
        ],
        remainder="drop"
    )

    return preprocessor
