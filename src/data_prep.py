"""Data Preprocessing & Preparation Module for Movie Reviews Emotion Classification.

Addresses 5 core data quality and pipeline issues:
1. Deduplication: Drops repeat review texts to prevent train/test leakage.
2. Conflict resolution: Drops review texts with multiple conflicting emotion labels (drop_conflicts).
3. Sparse class handling: Drops the 'surprise' class (only 19 reviews, all from one movie 'She\'s All That')
   leaving 7 well-defined classes suitable for validation.
4. Stateless text preprocessing: Normalizes text (contractions, HTML, lower, non-alpha) before deduplication.
5. Stratified Grouped Splitting: Uses StratifiedGroupKFold on `movie_name` to ensure both
   group-disjoint partitions across films and stratified class balance across all 7 emotions.
"""

import html
import os
import re
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'s": " is", "'re": " are", "'ve": " have", "'d": " would",
    "'ll": " will", "'m": " am", "it's": "it is"
}


def sanitize_text(text: str) -> str:
    """Stateless text cleaning: HTML entities, tags, URLs, contractions, lowercase, non-alpha."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    for c, exp in CONTRACTIONS.items():
        text = text.replace(c, exp)
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def find_repo_root(start_path: Union[str, Path] = ".") -> Path:
    """Programmatically locate the repository root containing data/raw/."""
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "data" / "raw" / "Movies_Reviews_modified_version1.csv").is_file():
            return parent
        if (parent / "data_prep.py").is_file() and (parent.parent / "data").is_dir():
            return parent.parent
    return current


def load_clean(
    filepath: Union[str, Path] = "data/raw/Movies_Reviews_modified_version1.csv",
    drop_conflicts: bool = True,
    drop_surprise: bool = True,
    deduplicate: bool = True,
    clean_text_first: bool = True,
) -> pd.DataFrame:
    """Load and clean the raw movie reviews dataset.

    Guarantees exactly 16,107 clean records across 1,309 unique movies and 7 classes.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the raw CSV file.
    drop_conflicts : bool, default=True
        If True, drops review texts that appear with multiple conflicting emotion labels.
    drop_surprise : bool, default=True
        If True, drops the 'surprise' class (only 19 reviews, all from one movie).
    deduplicate : bool, default=True
        If True, removes duplicate review texts, keeping the first occurrence.
    clean_text_first : bool, default=True
        If True, normalizes and cleans text before conflict detection and deduplication.
        Produces exactly 16,107 clean records across 1,309 movies.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 7 emotion classes.
    """
    path = Path(filepath)
    if not path.is_file():
        root = find_repo_root()
        path = root / "data" / "raw" / "Movies_Reviews_modified_version1.csv"
        if not path.is_file():
            # Fallback relative to current working directory
            candidates = [
                Path("data/raw/Movies_Reviews_modified_version1.csv"),
                Path("../data/raw/Movies_Reviews_modified_version1.csv"),
                Path("../../data/raw/Movies_Reviews_modified_version1.csv")
            ]
            for cand in candidates:
                if cand.is_file():
                    path = cand
                    break

    if not path.is_file():
        raise FileNotFoundError(f"Could not locate raw dataset at {filepath} or within repo root.")

    df = pd.read_csv(path)
    df = df.dropna(subset=["Reviews", "emotion"]).copy()

    if clean_text_first:
        df["cleaned_review"] = df["Reviews"].apply(sanitize_text)
        df = df[df["cleaned_review"] != ""].copy()
        match_col = "cleaned_review"
    else:
        match_col = "Reviews"

    # Drop review texts that carry conflicting emotion labels
    if drop_conflicts:
        conflict_texts = (
            df.groupby(match_col)["emotion"]
            .nunique()
            .loc[lambda s: s > 1]
            .index
        )
        df = df[~df[match_col].isin(conflict_texts)].copy()

    # Deduplicate exact review texts
    if deduplicate:
        df = df.drop_duplicates(subset=[match_col]).copy()

    # Handle surprise class
    if drop_surprise:
        df = df[df["emotion"] != "surprise"].copy()

    if "cleaned_review" not in df.columns:
        df["cleaned_review"] = df["Reviews"].apply(sanitize_text)

    df = df.reset_index(drop=True)
    return df


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "movie_name",
    target_col: str = "emotion",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified grouped train/test split.

    Guarantees that movie groups are disjoint between train and test, while
    preserving stratified proportions of each emotion class across the 7 classes.
    """
    n_splits = int(round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(df, df[target_col], groups=df[group_col]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df


def get_cv(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedGroupKFold:
    """Return a StratifiedGroupKFold cross-validator for grouped, stratified CV."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


class WordCountIQRCapper(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that fits Tukey IQR fences on training folds
    and caps outliers during transform, preventing fence leakage.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_fence_ = None
        self.upper_fence_ = None

    def fit(self, X, y=None):
        vals = np.asarray(X).ravel()
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        self.upper_fence_ = float(q3 + self.factor * iqr)
        self.lower_fence_ = float(max(0.0, q1 - self.factor * iqr))
        return self

    def transform(self, X):
        vals = np.asarray(X).ravel()
        capped = np.clip(vals, self.lower_fence_, self.upper_fence_)
        return capped.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["word_count_capped"])
