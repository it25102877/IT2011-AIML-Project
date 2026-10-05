"""Train and export Member 4's selected Random Forest prediction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.protocol import standard_setup  # noqa: E402


DEFAULT_MODEL_PATH = REPO_ROOT / "results" / "phase2" / "random_forest_model.joblib"
DEFAULT_METADATA_PATH = REPO_ROOT / "results" / "phase2" / "random_forest_model_metadata.json"

BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


def train_pipeline():
    """Fit the selected pipeline on the canonical training partition only."""
    ctx = standard_setup(verbose=True)
    pipeline = Pipeline([
        ("prep", ctx.build_preprocessor()),
        ("clf", RandomForestClassifier(**BEST_PARAMS)),
    ])
    print("Training the selected Random Forest on 12,886 training reviews...")
    pipeline.fit(ctx.X_train, ctx.y_train)
    return pipeline, ctx


def main() -> None:
    """Train the model and save its pipeline and provenance metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    pipeline, ctx = train_pipeline()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path, compress=3)

    metadata = {
        "model": "RandomForestClassifier",
        "member_id": "IT25102877",
        "role": "primary",
        "training_rows": len(ctx.X_train),
        "training_movies": int(ctx.groups_train.nunique()),
        "labels": list(ctx.LABELS),
        "features": ["Reviews", "genres", "Ratings"],
        "selected_parameters": BEST_PARAMS,
        "test_set_used_for_export": False,
        "warning": "Class probabilities are uncalibrated and are not suitable for high-stakes use.",
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model: {args.model_path.relative_to(REPO_ROOT)}")
    print(f"Saved metadata: {args.metadata_path.relative_to(REPO_ROOT)}")
    print("The protected test partition was not used during this export fit.")


if __name__ == "__main__":
    main()
