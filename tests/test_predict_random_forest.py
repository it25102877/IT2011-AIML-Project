"""Tests for the reusable Random Forest prediction interface."""

import numpy as np
import pytest

from scripts.predict_random_forest import make_input, normalize_genres, predict_review


class _Classifier:
    classes_ = np.array(["anger", "joy", "sadness"])


class _Predictor:
    named_steps = {"clf": _Classifier()}

    def predict(self, frame):
        assert list(frame.columns) == ["Reviews", "genres", "Ratings"]
        return np.array(["joy"])

    def predict_proba(self, frame):
        return np.array([[0.1, 0.7, 0.2]])


def test_normalize_genres_accepts_common_input_forms():
    assert normalize_genres("Drama, Romance") == "['Drama', 'Romance']"
    assert normalize_genres("['Horror', 'Thriller']") == "['Horror', 'Thriller']"
    assert normalize_genres(["Comedy"]) == "['Comedy']"


def test_make_input_validates_required_values():
    frame = make_input("A hopeful ending", "Drama", 8)
    assert frame.iloc[0].to_dict() == {
        "Reviews": "A hopeful ending",
        "genres": "['Drama']",
        "Ratings": 8.0,
    }
    with pytest.raises(ValueError, match="must not be empty"):
        make_input("", "Drama", 8)
    with pytest.raises(ValueError, match="between 0 and 10"):
        make_input("Review", "Drama", 11)


def test_predict_review_returns_ranked_probabilities():
    prediction, ranked = predict_review(_Predictor(), "A joyful ending", "Comedy", 9)
    assert prediction == "joy"
    assert ranked == [("joy", 0.7), ("sadness", 0.2), ("anger", 0.1)]
