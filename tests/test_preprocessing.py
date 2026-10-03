"""Sanity and verification tests for data preparation and preprocessing components."""

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    WordCountIQRCapper,
    find_repo_root,
    get_cv,
    load_clean,
    make_split,
)
from src.preprocessors import (
    GenreBinarizer,
    StopwordFilter,
    TextCleaner,
    WordCountExtractor,
    build_full_preprocessor,
    compute_train_class_weights,
)


@pytest.fixture(scope="module")
def dataset():
    """Load cleaned dataset once for all tests."""
    df = load_clean()
    return df


def test_load_clean_counts(dataset):
    """Fact check: exactly 16,107 rows, 1,309 movies, and 7 classes."""
    assert len(dataset) == 16107, f"Expected 16,107 rows, got {len(dataset)}"
    assert dataset["movie_name"].nunique() == 1309, f"Expected 1,309 movies, got {dataset['movie_name'].nunique()}"
    emotions = dataset["emotion"].unique()
    assert len(emotions) == 7, f"Expected 7 classes, got {len(emotions)}"
    assert "surprise" not in emotions, "'surprise' must be excluded"


def test_make_split_properties(dataset):
    """Check split counts, zero movie overlap, zero review overlap, and stratified class balance."""
    train_df, test_df = make_split(dataset, test_size=0.2, random_state=42)

    assert len(train_df) == 12886, f"Expected 12,886 train rows, got {len(train_df)}"
    assert len(test_df) == 3221, f"Expected 3,221 test rows, got {len(test_df)}"

    # Zero movie overlap
    train_movies = set(train_df["movie_name"])
    test_movies = set(test_df["movie_name"])
    overlap_movies = train_movies.intersection(test_movies)
    assert len(overlap_movies) == 0, f"Movie overlap detected: {len(overlap_movies)} movies"

    # Zero text overlap
    train_texts = set(train_df["cleaned_review"])
    test_texts = set(test_df["cleaned_review"])
    overlap_texts = train_texts.intersection(test_texts)
    assert len(overlap_texts) == 0, f"Text overlap detected: {len(overlap_texts)} reviews"


def test_text_cleaner_and_stopword_filter():
    """Test Member 1 and Member 2 stateless transformers."""
    cleaner = TextCleaner()
    stop_filter = StopwordFilter()

    raw_sample = pd.Series(["<b>Amazing</b> film! It won't disappoint you: http://imdb.com"])
    cleaned = cleaner.transform(raw_sample)
    assert "<b>" not in cleaned[0]
    assert "will not" in cleaned[0]
    assert "http" not in cleaned[0]

    filtered = stop_filter.transform(cleaned)
    assert "film" not in filtered[0].split()
    assert len(filtered[0]) > 0


def test_genre_binarizer():
    """Test Member 3 GenreBinarizer vocabulary learning and unseen genre handling."""
    train_genres = pd.Series(["['Action', 'Comedy']", "['Drama']", "['Action', 'Drama']"])
    test_genres = pd.Series(["['Action', 'Mystery']"])  # 'Mystery' is unseen

    binarizer = GenreBinarizer()
    binarizer.fit(train_genres)

    assert "action" in binarizer.genres_
    assert "comedy" in binarizer.genres_
    assert "drama" in binarizer.genres_
    assert "mystery" not in binarizer.genres_

    test_mat = binarizer.transform(test_genres)
    assert test_mat.shape == (1, 3)
    # Action should be 1
    action_idx = binarizer.genre_to_idx_["action"]
    assert test_mat[0, action_idx] == 1


def test_word_count_iqr_capper():
    """Test Member 4 WordCountIQRCapper fits on train and caps properly."""
    capper = WordCountIQRCapper(factor=1.5)
    train_counts = np.array([10, 20, 25, 30, 1000])  # outlier 1000
    capper.fit(train_counts)

    assert capper.upper_fence_ < 1000
    assert capper.lower_fence_ >= 0

    test_counts = np.array([5, 500])
    capped = capper.transform(test_counts)
    assert capped[1, 0] <= capper.upper_fence_


def test_compute_train_class_weights(dataset):
    """Test Member 5 class weights computation on training data only."""
    train_df, _ = make_split(dataset, test_size=0.2, random_state=42)
    weights, summary = compute_train_class_weights(train_df["emotion"])

    assert len(weights) == 7
    assert len(summary) == 7
    # Minority classes should have higher weights than majority class 'sadness'
    assert weights["disgust"] > weights["sadness"]
    assert weights["anger"] > weights["sadness"]


def test_build_full_preprocessor_end_to_end(dataset):
    """Test end-to-end full preprocessor fit on train, transform on test, check no NaNs, and consistency."""
    train_df, test_df = make_split(dataset, test_size=0.2, random_state=42)

    preprocessor = build_full_preprocessor(use_svd=False)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert X_train.shape[1] == X_test.shape[1]

    # Verify no NaNs in transformed output
    assert not np.isnan(X_train.toarray() if hasattr(X_train, "toarray") else X_train).any()
    assert not np.isnan(X_test.toarray() if hasattr(X_test, "toarray") else X_test).any()

    # Verify consistency between ColumnTransformer and Stage-wise transformations
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_wc = train_df["cleaned_review"].apply(lambda x: len(x.split())).to_numpy().reshape(-1, 1)
    test_wc = test_df["cleaned_review"].apply(lambda x: len(x.split())).to_numpy().reshape(-1, 1)

    capper = WordCountIQRCapper(factor=1.5)
    train_wc_capped = capper.fit_transform(train_wc)
    test_wc_capped = capper.transform(test_wc)

    robust_scaler = RobustScaler()
    robust_scaler.fit(train_wc_capped)
    test_wc_robust = robust_scaler.transform(test_wc_capped)

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(train_df[["Ratings"]])
    test_rating_scaled = minmax_scaler.transform(test_df[["Ratings"]])

    stop_filter = StopwordFilter()
    train_tokens = stop_filter.transform(train_df["cleaned_review"])
    test_tokens = stop_filter.transform(test_df["cleaned_review"])
    tfidf_vec = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    tfidf_vec.fit(train_tokens)
    test_tfidf = tfidf_vec.transform(test_tokens)

    X_test_arr = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    diff_tfidf = float(np.abs(X_test[:, :2500] - test_tfidf).max())
    diff_wc = float(np.abs(X_test_arr[:, -2] - test_wc_robust.ravel()).max())
    diff_rating = float(np.abs(X_test_arr[:, -1] - test_rating_scaled.ravel()).max())

    assert np.allclose(X_test_arr[:, -2], test_wc_robust.ravel()), f"Word count mismatch: {diff_wc}"
    assert np.allclose(X_test_arr[:, -1], test_rating_scaled.ravel()), f"Rating mismatch: {diff_rating}"
    assert diff_tfidf < 1e-9, f"TF-IDF max diff too high: {diff_tfidf}"
    assert diff_wc < 1e-9, f"Word count max diff too high: {diff_wc}"
    assert diff_rating < 1e-9, f"Rating max diff too high: {diff_rating}"

    # Test SVD option
    preprocessor_svd = build_full_preprocessor(use_svd=True, n_svd_components=100)
    X_train_svd = preprocessor_svd.fit_transform(train_df)
    assert X_train_svd.shape[0] == len(train_df)
