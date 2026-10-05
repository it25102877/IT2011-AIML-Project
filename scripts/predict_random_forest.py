"""Predict movie-review emotions with Member 4's exported Random Forest."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import project classes before unpickling the preprocessing pipeline.
import src.data_prep  # noqa: E402,F401
import src.preprocessors  # noqa: E402,F401


DEFAULT_MODEL_PATH = REPO_ROOT / "results" / "phase2" / "random_forest_model.joblib"


def normalize_genres(genres: str | Iterable[str]) -> str:
    """Return genres in the list-like representation used during training."""
    if isinstance(genres, str):
        value = genres.strip()
        if not value:
            return "[]"
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parsed = list(genres)
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError("genres must be a comma-separated string or a list of genre names")
    return repr([str(item).strip() for item in parsed if str(item).strip()])


def make_input(review: str, genres: str | Iterable[str], rating: float) -> pd.DataFrame:
    """Validate one request and return the model's three required columns."""
    review = str(review).strip()
    if not review:
        raise ValueError("review must not be empty")
    rating = float(rating)
    if not 0 <= rating <= 10:
        raise ValueError("rating must be between 0 and 10")
    return pd.DataFrame({
        "Reviews": [review],
        "genres": [normalize_genres(genres)],
        "Ratings": [rating],
    })


def predict_review(model, review: str, genres: str | Iterable[str], rating: float):
    """Return the selected emotion and probabilities ranked from high to low."""
    model_input = make_input(review, genres, rating)
    prediction = str(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]
    classes = model.named_steps["clf"].classes_
    ranked = sorted(
        ((str(label), float(probability)) for label, probability in zip(classes, probabilities)),
        key=lambda item: item[1],
        reverse=True,
    )
    return prediction, ranked


def print_prediction(prediction: str, ranked, top: int = 7) -> None:
    """Print a readable prediction and ranked class vote distribution."""
    print(f"\nPredicted emotion: {prediction}")
    print("Class vote distribution:")
    for label, probability in ranked[:top]:
        print(f"  {label:<13} {probability:>7.2%}")
    print("Note: these Random Forest probabilities are not calibrated confidence scores.")


def load_model(path: Path):
    """Load the exported model or explain how to create it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python scripts/train_random_forest_predictor.py"
        )
    return joblib.load(path)


def interactive(model, top: int) -> None:
    """Run a prompt loop until the user enters quit."""
    print("Random Forest movie-review emotion predictor")
    print("Type quit as the review to stop.")
    while True:
        review = input("\nReview: ").strip()
        if review.lower() in {"quit", "exit"}:
            print("Prediction session ended.")
            return
        genres = input("Genres (for example Drama, Romance): ").strip()
        rating = input("Rating from 0 to 10: ").strip()
        try:
            prediction, ranked = predict_review(model, review, genres, float(rating))
            print_prediction(prediction, ranked, top)
        except ValueError as error:
            print(f"Invalid input: {error}")


def main() -> None:
    """Run a one-off prediction or start interactive mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--review", help="Movie-review text; omit for interactive mode")
    parser.add_argument("--genres", default="", help="Comma-separated genres")
    parser.add_argument("--rating", type=float, help="Numerical rating from 0 to 10")
    parser.add_argument("--top", type=int, default=7, choices=range(1, 8))
    args = parser.parse_args()

    model = load_model(args.model)
    if args.review is None:
        interactive(model, args.top)
        return
    if args.rating is None:
        parser.error("--rating is required when --review is supplied")
    prediction, ranked = predict_review(model, args.review, args.genres, args.rating)
    print_prediction(prediction, ranked, args.top)


if __name__ == "__main__":
    main()
