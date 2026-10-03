"""Tests for the Phase 2 shared evaluation module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score

from src.data_prep import load_clean, make_split
from src.evaluation import (
    LABELS,
    SCHEMA_VERSION,
    bootstrap_macro_f1,
    compute_baselines,
    compute_review_id,
    confusion_matrix_normalized,
    cv_fold_scores,
    evaluate_on_test,
    fold_fingerprints,
    load_model_results,
    macro_f1_from_confusion,
    make_bootstrap_indices,
    movie_confusion_tensor,
    paired_bootstrap_difference,
    percentile_ci,
    save_model_results,
    sha256_of_names,
    split_fingerprint,
    validate_results,
)


def _make_dummy_test_df(n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic test DataFrame for unit testing without loading raw data."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Reviews": [f"Synthetic review text number {i} for emotion classification" for i in range(n_samples)],
        "movie_name": [f"movie_{i % 5}" for i in range(n_samples)],
        "emotion": rng.choice(LABELS, size=n_samples),
    })


def _build_valid_payload() -> dict:
    """Build a minimal valid results dictionary matching schema expectations."""
    cm = [
        [10, 1, 0, 0, 0, 0, 0],
        [1, 8, 1, 0, 0, 0, 0],
        [0, 1, 9, 0, 0, 0, 0],
        [0, 0, 0, 10, 0, 0, 0],
        [0, 0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0, 0, 10, 0],
        [0, 0, 0, 0, 0, 0, 10],
    ]
    cm_arr = np.array(cm)
    total_test = int(cm_arr.sum())
    macro_f1 = macro_f1_from_confusion(cm_arr)

    per_class = {}
    for idx, lbl in enumerate(LABELS):
        per_class[lbl] = {
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "support": int(cm_arr[idx].sum()),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_key": "test_model",
        "labels": list(LABELS),
        "created_with": {
            "python": "3.14.0",
            "sklearn": "1.6.0",
            "numpy": "2.0.0",
            "pandas": "2.2.0",
        },
        "meta": {
            "model_label": "Test Classifier",
            "member_id": "IT25102877",
            "algorithm": "DecisionTreeClassifier",
            "preprocessing": "TF-IDF + MultiHot",
            "tuning": {
                "method": "GridSearchCV",
                "n_configs": 10,
                "n_folds": 5,
                "scoring": "f1_macro",
                "elapsed_seconds": 12.5,
                "best_params": {"max_depth": 5},
            },
        },
        "split": {
            "random_state": 42,
            "test_size": 0.2,
            "n_train": 100,
            "n_test": total_test,
            "n_train_movies": 20,
            "n_test_movies": 5,
            "train_movies_sha256": "abc",
            "test_movies_sha256": "def",
            "train_review_ids_sha256": "123",
            "test_review_ids_sha256": "456",
        },
        "cv": {
            "n_splits": 5,
            "macro_f1": {"folds": [0.8, 0.82, 0.81, 0.79, 0.83], "mean": 0.81, "std": 0.01414},
            "accuracy": {"folds": [0.85, 0.86, 0.84, 0.85, 0.85], "mean": 0.85, "std": 0.00632},
            "weighted_f1": {"folds": [0.83, 0.84, 0.82, 0.83, 0.83], "mean": 0.83, "std": 0.00632},
        },
        "cv_fingerprints": [
            {"fold": 1, "n_train": 80, "n_val": 20, "val_movies_sha256": "f1"},
            {"fold": 2, "n_train": 80, "n_val": 20, "val_movies_sha256": "f2"},
            {"fold": 3, "n_train": 80, "n_val": 20, "val_movies_sha256": "f3"},
            {"fold": 4, "n_train": 80, "n_val": 20, "val_movies_sha256": "f4"},
            {"fold": 5, "n_train": 80, "n_val": 20, "val_movies_sha256": "f5"},
        ],
        "test": {
            "accuracy": 0.9,
            "macro_f1": macro_f1,
            "weighted_f1": 0.9,
            "per_class": per_class,
            "confusion_matrix": cm,
        },
        "baselines": {
            "majority": {"class": "sadness", "accuracy": 0.4, "macro_f1": 0.08, "weighted_f1": 0.23},
            "stratified_random": {"accuracy_mean": 0.24, "accuracy_std": 0.01, "macro_f1_mean": 0.14, "macro_f1_std": 0.01},
            "uniform_random": {"accuracy_mean": 0.14, "accuracy_std": 0.01, "macro_f1_mean": 0.12, "macro_f1_std": 0.01},
        },
    }


# ==============================================================================
# SYNTHETIC TESTS (Zero Data Loading)
# ==============================================================================

def test_macro_f1_from_confusion_matches_sklearn():
    """Verify macro_f1_from_confusion matches sklearn f1_score on random distributions."""
    rng = np.random.default_rng(123)
    for _ in range(5):
        y_true = rng.choice(LABELS, size=300)
        y_pred = rng.choice(LABELS, size=300)
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)

        sklearn_score = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        cm_score = macro_f1_from_confusion(cm)

        assert abs(sklearn_score - cm_score) < 1e-12


def test_confusion_matrix_normalized():
    """Verify confusion_matrix_normalized computes correct row-normalized fractions."""
    cm = np.array([
        [10, 10, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],  # zero support row
        [5, 5, 5, 5, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 8],
    ])
    norm = confusion_matrix_normalized(cm)

    assert norm.shape == (7, 7)
    assert np.allclose(norm[0], [0.5, 0.5, 0, 0, 0, 0, 0])
    assert np.allclose(norm[1], np.zeros(7))
    assert np.isclose(norm[0].sum(), 1.0)
    assert np.isclose(norm[2].sum(), 1.0)


def test_evaluate_on_test_dummy_model():
    """Verify evaluate_on_test calls predict once and produces correct output formats."""
    df_test = _make_dummy_test_df(n_samples=40, seed=7)
    y_test = df_test["emotion"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(["sadness"] * len(df_test))

    metrics, preds_df = evaluate_on_test(mock_model, df_test, y_test)

    # Assert model.predict called exactly once
    assert mock_model.predict.call_count == 1

    # Assert predictions DataFrame structure
    expected_cols = ["row_id", "review_id", "movie_name", "y_true", "y_pred"]
    assert list(preds_df.columns) == expected_cols
    assert len(preds_df) == len(df_test)
    assert list(preds_df["row_id"]) == list(range(len(df_test)))
    assert preds_df["review_id"].nunique() == len(df_test)

    # Assert metrics structure
    assert set(metrics.keys()) == {"accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"}
    assert len(metrics["confusion_matrix"]) == 7
    assert len(metrics["per_class"]) == 7
    for lbl in LABELS:
        assert set(metrics["per_class"][lbl].keys()) == {"precision", "recall", "f1", "support"}


def test_fold_fingerprints_detects_leakage():
    """Verify fold_fingerprints raises ValueError when a movie leaks across folds."""
    movies = ["m1", "m1", "m2", "m3", "m4"]
    # Fold 1 has movie leakage (m1 in both train and val)
    cv_splits = [
        (np.array([0, 2]), np.array([1, 3, 4])),  # m1 is at index 0 and 1!
    ]
    with pytest.raises(ValueError, match="Movie leakage detected"):
        fold_fingerprints(cv_splits, movies)


def test_cv_fold_scores_extraction_and_assertion():
    """Verify cv_fold_scores extracts per-fold scores and verifies standard deviations."""
    mock_search = MagicMock()
    mock_search.best_index_ = 0
    mock_search.cv_results_ = {
        "split0_test_macro_f1": [0.15],
        "split1_test_macro_f1": [0.17],
        "split2_test_macro_f1": [0.16],
        "mean_test_macro_f1": [0.16],
        "std_test_macro_f1": [float(np.std([0.15, 0.17, 0.16], ddof=0))],
        "split0_test_accuracy": [0.30],
        "split1_test_accuracy": [0.32],
        "split2_test_accuracy": [0.34],
        "mean_test_accuracy": [0.32],
        "std_test_accuracy": [float(np.std([0.30, 0.32, 0.34], ddof=0))],
        "split0_test_weighted_f1": [0.25],
        "split1_test_weighted_f1": [0.26],
        "split2_test_weighted_f1": [0.27],
        "mean_test_weighted_f1": [0.26],
        "std_test_weighted_f1": [float(np.std([0.25, 0.26, 0.27], ddof=0))],
    }

    scores = cv_fold_scores(mock_search)
    assert scores["n_splits"] == 3
    assert np.isclose(scores["macro_f1"]["mean"], 0.16)
    assert len(scores["macro_f1"]["folds"]) == 3

    # Missing scorer raises ValueError
    bad_search = MagicMock()
    bad_search.best_index_ = 0
    bad_search.cv_results_ = {"split0_test_macro_f1": [0.15]}
    with pytest.raises(ValueError, match="missing required metric"):
        cv_fold_scores(bad_search)


def test_save_and_load_round_trip(tmp_path: Path):
    """Verify save_model_results and load_model_results execute clean serialization round trip."""
    valid_payload = _build_valid_payload()
    cm_arr = np.asarray(valid_payload["test"]["confusion_matrix"])
    y_true_list = []
    y_pred_list = []
    for i, t_lbl in enumerate(LABELS):
        for j, p_lbl in enumerate(LABELS):
            count = cm_arr[i, j]
            y_true_list.extend([t_lbl] * count)
            y_pred_list.extend([p_lbl] * count)

    preds_df = pd.DataFrame({
        "row_id": list(range(len(y_true_list))),
        "review_id": [f"rev_{i:04d}" for i in range(len(y_true_list))],
        "movie_name": [f"movie_{i % 5}" for i in range(len(y_true_list))],
        "y_true": y_true_list,
        "y_pred": y_pred_list,
    })

    json_path, csv_path = save_model_results(
        out_dir=tmp_path,
        model_key="test_model",
        meta=valid_payload["meta"],
        split=valid_payload["split"],
        cv=valid_payload["cv"],
        cv_fingerprints=valid_payload["cv_fingerprints"],
        test=valid_payload["test"],
        baselines=valid_payload["baselines"],
        predictions_df=preds_df,
    )

    assert json_path.is_file()
    assert csv_path.is_file()

    loaded = load_model_results(json_path)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["model_key"] == "test_model"
    assert loaded["labels"] == LABELS


def test_save_model_results_predictions_df_validations(tmp_path: Path):
    """Verify save_model_results validates predictions_df and writes nothing on failure."""
    valid_payload = _build_valid_payload()
    cm_arr = np.asarray(valid_payload["test"]["confusion_matrix"])
    y_true_list = []
    y_pred_list = []
    for i, t_lbl in enumerate(LABELS):
        for j, p_lbl in enumerate(LABELS):
            count = cm_arr[i, j]
            y_true_list.extend([t_lbl] * count)
            y_pred_list.extend([p_lbl] * count)

    def _make_valid_preds_df():
        return pd.DataFrame({
            "row_id": list(range(len(y_true_list))),
            "review_id": [f"rev_{i:04d}" for i in range(len(y_true_list))],
            "movie_name": [f"movie_{i % 5}" for i in range(len(y_true_list))],
            "y_true": list(y_true_list),
            "y_pred": list(y_pred_list),
        })

    def _try_save(df, subfolder_name):
        out_sub = tmp_path / subfolder_name
        save_model_results(
            out_dir=out_sub,
            model_key="test_model",
            meta=valid_payload["meta"],
            split=valid_payload["split"],
            cv=valid_payload["cv"],
            cv_fingerprints=valid_payload["cv_fingerprints"],
            test=valid_payload["test"],
            baselines=valid_payload["baselines"],
            predictions_df=df,
        )

    # 1. Wrong column order
    df_wrong_cols = _make_valid_preds_df()[["review_id", "row_id", "movie_name", "y_true", "y_pred"]]
    with pytest.raises(ValueError, match="columns mismatch"):
        _try_save(df_wrong_cols, "fail_cols")
    assert not (tmp_path / "fail_cols").exists() or len(list((tmp_path / "fail_cols").iterdir())) == 0

    # 2. Missing row
    df_missing_row = _make_valid_preds_df().iloc[:-1].copy()
    with pytest.raises(ValueError, match="row count .* does not match split.n_test"):
        _try_save(df_missing_row, "fail_rows")
    assert not (tmp_path / "fail_rows").exists() or len(list((tmp_path / "fail_rows").iterdir())) == 0

    # 3. Duplicate review_id
    df_dup_rev = _make_valid_preds_df()
    df_dup_rev.loc[1, "review_id"] = df_dup_rev.loc[0, "review_id"]
    with pytest.raises(ValueError, match="duplicate review_id"):
        _try_save(df_dup_rev, "fail_dup")
    assert not (tmp_path / "fail_dup").exists() or len(list((tmp_path / "fail_dup").iterdir())) == 0

    # 4. Unknown label in y_true or y_pred
    df_bad_label = _make_valid_preds_df()
    df_bad_label.loc[0, "y_true"] = "unknown_emotion"
    with pytest.raises(ValueError, match="contains invalid labels"):
        _try_save(df_bad_label, "fail_label")
    assert not (tmp_path / "fail_label").exists() or len(list((tmp_path / "fail_label").iterdir())) == 0

    # 5. Changed prediction that no longer matches confusion matrix
    df_changed_pred = _make_valid_preds_df()
    cur_pred = df_changed_pred.loc[0, "y_pred"]
    alt_pred = [l for l in LABELS if l != cur_pred][0]
    df_changed_pred.loc[0, "y_pred"] = alt_pred
    with pytest.raises(ValueError, match="Recomputed 7x7 confusion matrix .* does not match"):
        _try_save(df_changed_pred, "fail_cm")
    assert not (tmp_path / "fail_cm").exists() or len(list((tmp_path / "fail_cm").iterdir())) == 0


def test_validate_results_error_conditions():
    """Verify validate_results raises descriptive ValueError across all specified invalid states."""
    # 1. Schema version mismatch
    p = _build_valid_payload()
    p["schema_version"] = 999
    with pytest.raises(ValueError, match="Schema version mismatch"):
        validate_results(p)

    # 2. Missing top-level key
    p = _build_valid_payload()
    del p["meta"]
    with pytest.raises(ValueError, match="Missing required top-level key: 'meta'"):
        validate_results(p)

    # 3. Labels mismatch
    p = _build_valid_payload()
    p["labels"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="Labels mismatch"):
        validate_results(p)

    # 4. Missing review ID hash in split
    p = _build_valid_payload()
    del p["split"]["train_review_ids_sha256"]
    with pytest.raises(ValueError, match="missing required hash key"):
        validate_results(p)

    # 5. Confusion matrix shape mismatch
    p = _build_valid_payload()
    p["test"]["confusion_matrix"] = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="Confusion matrix shape mismatch"):
        validate_results(p)

    # 6. Confusion matrix total count mismatch
    p = _build_valid_payload()
    p["split"]["n_test"] = 9999
    with pytest.raises(ValueError, match="Confusion matrix sum .* does not match split.n_test"):
        validate_results(p)

    # 7. Class support mismatch
    p = _build_valid_payload()
    p["test"]["per_class"]["anger"]["support"] = 9999
    with pytest.raises(ValueError, match="Class 'anger' support mismatch"):
        validate_results(p)

    # 8. Recomputed macro-F1 mismatch
    p = _build_valid_payload()
    p["test"]["macro_f1"] = 0.001
    with pytest.raises(ValueError, match="Recomputed Macro-F1 .* differs from recorded"):
        validate_results(p)

    # 9. CV fold count mismatch
    p = _build_valid_payload()
    p["cv"]["macro_f1"]["folds"] = [0.1, 0.2]  # cv.n_splits is 5
    with pytest.raises(ValueError, match="folds count .* does not match cv.n_splits"):
        validate_results(p)


def test_bootstrapping_reproducibility_and_paired_difference():
    """Verify movie tensor bootstrapping reproducibility, point estimate proximity, and self-paired 0 difference."""
    rng = np.random.default_rng(42)
    n_movies = 15
    tensor = rng.integers(0, 10, size=(n_movies, 7, 7))

    # Point estimate from overall confusion matrix
    cm_overall = tensor.sum(axis=0)
    point_f1 = macro_f1_from_confusion(cm_overall)

    # Bootstrap indices
    idx1 = make_bootstrap_indices(n_movies=n_movies, n_boot=200, seed=42)
    idx2 = make_bootstrap_indices(n_movies=n_movies, n_boot=200, seed=42)
    np.testing.assert_array_equal(idx1, idx2)

    draws1 = bootstrap_macro_f1(tensor, idx1)
    draws2 = bootstrap_macro_f1(tensor, idx2)
    np.testing.assert_array_almost_equal(draws1, draws2)

    # Bootstrap mean is close to point estimate
    assert abs(np.mean(draws1) - point_f1) < 0.05

    # Paired difference of a model with itself is exactly 0
    diff = paired_bootstrap_difference(tensor, tensor, idx1)
    np.testing.assert_array_almost_equal(diff, np.zeros_like(diff))

    # Percentile CI contains point estimate
    low, high = percentile_ci(draws1, level=0.95)
    assert low < high
    assert low <= point_f1 <= high


# ==============================================================================
# REAL DATA INVARIANT TESTS
# ==============================================================================

def test_real_data_evaluation_and_baselines():
    """Verify evaluation invariants, review IDs, and baselines against clean real dataset."""
    df = load_clean()
    train_df, test_df = make_split(df, test_size=0.2, random_state=42)

    # 1. Fingerprint verification
    fp = split_fingerprint(train_df, test_df, random_state=42, test_size=0.2)
    assert fp["n_train"] == 12886
    assert fp["n_test"] == 3221
    assert fp["n_train_movies"] == 1048
    assert fp["n_test_movies"] == 261

    # 2. Review ID uniqueness and disjointness
    train_review_ids = [compute_review_id(t) for t in train_df["Reviews"]]
    test_review_ids = [compute_review_id(t) for t in test_df["Reviews"]]

    assert len(set(train_review_ids)) == len(train_df)
    assert len(set(test_review_ids)) == len(test_df)
    assert len(set(train_review_ids).intersection(set(test_review_ids))) == 0

    # 3. evaluate_on_test produces matching review_ids sequence
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(["sadness"] * len(test_df))
    metrics, preds_df = evaluate_on_test(mock_model, test_df, test_df["emotion"])

    assert list(preds_df["review_id"]) == test_review_ids
    assert len(preds_df) == 3221

    # 4. Baselines reproduction
    baselines = compute_baselines(train_df["emotion"], test_df["emotion"], n_seeds=200)
    assert round(baselines["majority"]["macro_f1"], 4) == 0.0827
    assert round(baselines["majority"]["accuracy"], 4) == 0.4070
    assert 0.140 <= baselines["stratified_random"]["macro_f1_mean"] <= 0.146
    assert 0.220 <= baselines["stratified_random"]["accuracy_mean"] <= 0.260
