"""Tests for the minimal shared Phase 2 training protocol."""

import json
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data_prep import find_repo_root, get_cv
from src.evaluation import LABELS, load_model_results, validate_results
from src.protocol import (
    SCORING,
    cv_scores_from_folds,
    finalize_model,
    fold_scores_for,
    standard_setup,
)


@pytest.fixture(scope="module")
def ctx():
    """Reuse the expensive canonical setup across protocol tests."""
    return standard_setup(verbose=False)


def _manual_scores():
    return cv_scores_from_folds(
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.20, 0.21, 0.22, 0.23, 0.24],
        [0.15, 0.16, 0.17, 0.18, 0.19],
    )


def _manual_finalize(ctx, tmp_path, **overrides):
    args = {
        "model_key": "logistic_regression",
        "member_id": "IT25102550",
        "model_label": "Manual Test",
        "algorithm": "ExternalModel",
        "preprocessing": "External fold-safe preprocessing",
        "y_pred": np.full(len(ctx.X_test), "sadness"),
        "cv_scores": _manual_scores(),
        "n_configs": 1,
        "tuning_method": "ManualSearch",
        "elapsed_seconds": 0.1,
        "best_params": {},
        "out_dir": tmp_path,
    }
    args.update(overrides)
    return finalize_model(ctx, **args)


def test_standard_setup_invariants_and_folds(ctx):
    assert len(ctx.df) == 16107
    assert ctx.df["movie_name"].nunique() == 1309
    assert ctx.df["emotion"].nunique() == 7
    assert "surprise" not in set(ctx.df["emotion"])
    assert len(ctx.train_df) == 12886
    assert len(ctx.test_df) == 3221
    expected = list(get_cv(5).split(ctx.X_train, ctx.y_train, groups=ctx.groups_train))
    assert len(ctx.cv_splits) == len(expected) == 5
    for actual_fold, expected_fold in zip(ctx.cv_splits, expected):
        np.testing.assert_array_equal(actual_fold[0], expected_fold[0])
        np.testing.assert_array_equal(actual_fold[1], expected_fold[1])
    assert ctx.build_preprocessor() is not ctx.build_preprocessor()


def test_search_quick_path_writes_valid_results(ctx, tmp_path):
    ctx.test_used = False
    pipe = Pipeline([("prep", ctx.build_preprocessor()), ("clf", DummyClassifier())])
    start = time.perf_counter()
    search = GridSearchCV(pipe, {"clf__strategy": ["most_frequent"]}, scoring=SCORING, refit="macro_f1", cv=ctx.cv_splits)
    search.fit(ctx.X_train, ctx.y_train)
    json_path, csv_path = finalize_model(ctx, model_key="logistic_regression", member_id="IT25102550", model_label="Quick Path Test", algorithm="DummyClassifier", preprocessing="Shared full pipeline", search=search, elapsed_seconds=time.perf_counter() - start, out_dir=tmp_path)

    assert json_path.parent == tmp_path
    assert csv_path.parent == tmp_path
    assert json_path.is_file() and csv_path.is_file()
    loaded = load_model_results(json_path)
    validate_results(loaded)
    predictions = pd.read_csv(csv_path)
    recomputed = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=LABELS)
    np.testing.assert_array_equal(recomputed, np.asarray(loaded["test"]["confusion_matrix"]))
    saved_text = json.dumps(loaded)
    assert str(tmp_path) not in saved_text
    assert str(find_repo_root()) not in saved_text


def test_manual_predictions_and_custom_fold_helpers(ctx, tmp_path):
    ctx.test_used = False
    json_path, csv_path = _manual_finalize(ctx, tmp_path)
    assert json_path.is_file() and csv_path.is_file()
    loaded = load_model_results(json_path)
    assert loaded["cv"]["macro_f1"]["mean"] == pytest.approx(0.12)
    assert loaded["cv"]["macro_f1"]["std"] == pytest.approx(np.std([0.10, 0.11, 0.12, 0.13, 0.14], ddof=0))

    macro, accuracy, weighted = fold_scores_for(
        ctx,
        lambda train_idx, val_idx: np.full(len(val_idx), "sadness"),
    )
    scores = cv_scores_from_folds(macro, accuracy, weighted)
    assert scores["n_splits"] == 5
    assert all(len(scores[name]["folds"]) == 5 for name in SCORING)


def test_finalize_rejects_invalid_inputs(ctx, tmp_path):
    predictions = np.full(len(ctx.X_test), "sadness")
    numeric_x = ctx.X_train[["Ratings"]]

    wrong_cv = GridSearchCV(
        DummyClassifier(), {"strategy": ["most_frequent"]},
        scoring=SCORING, refit="macro_f1", cv=5,
    ).fit(numeric_x, ctx.y_train)
    ctx.test_used = False
    with pytest.raises(ValueError, match="pass cv=ctx.cv_splits"):
        _manual_finalize(ctx, tmp_path, y_pred=None, search=wrong_cv)

    wrong_refit = GridSearchCV(
        DummyClassifier(), {"strategy": ["most_frequent"]},
        scoring=SCORING, refit="accuracy", cv=ctx.cv_splits,
    ).fit(numeric_x, ctx.y_train)
    ctx.test_used = False
    with pytest.raises(ValueError, match="search.refit"):
        _manual_finalize(ctx, tmp_path, y_pred=None, search=wrong_refit)

    ctx.test_used = False
    with pytest.raises(ValueError, match="model_key"):
        _manual_finalize(ctx, tmp_path, model_key="unknown_model")
    with pytest.raises(ValueError, match="member_id"):
        _manual_finalize(ctx, tmp_path, member_id="25102550")
    with pytest.raises(ValueError, match="exactly one prediction source"):
        _manual_finalize(ctx, tmp_path, model=DummyClassifier())
    with pytest.raises(ValueError, match="exactly 3221"):
        _manual_finalize(ctx, tmp_path, y_pred=predictions[:-1])
    bad_labels = predictions.copy()
    bad_labels[0] = "unknown"
    with pytest.raises(ValueError, match="unknown labels"):
        _manual_finalize(ctx, tmp_path, y_pred=bad_labels)


def test_finalize_allows_test_evaluation_only_once(ctx, tmp_path):
    ctx.test_used = False
    _manual_finalize(ctx, tmp_path)
    with pytest.raises(RuntimeError, match="test set was already used"):
        _manual_finalize(ctx, tmp_path)
