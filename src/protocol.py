"""Minimal shared training and result-export protocol for Phase 2 models."""
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.data_prep import find_repo_root, get_cv, load_clean, make_split
from src.evaluation import (
    LABELS as EVALUATION_LABELS,
    compute_baselines,
    cv_fold_scores,
    evaluate_on_test,
    fold_fingerprints,
    load_model_results,
    save_model_results,
    split_fingerprint,
    validate_results,
)
from src.preprocessors import build_full_preprocessor

ALLOWED_MODEL_KEYS = [
    "naive_bayes", "logistic_regression", "linear_svc", "decision_tree",
    "random_forest", "hist_gradient_boosting", "mlp",
]

SCORING = {"macro_f1": "f1_macro", "accuracy": "accuracy", "weighted_f1": "f1_weighted"}

_SHARED_FOLDS_MESSAGE = (
    "pass cv=ctx.cv_splits to the search so every model uses the same 5 movie-grouped folds")


@dataclass
class Context:
    """Hold the shared data, folds, metadata, and one-use test-set guard."""

    df: pd.DataFrame = field(repr=False)
    train_df: pd.DataFrame = field(repr=False)
    test_df: pd.DataFrame = field(repr=False)
    X_train: pd.DataFrame = field(repr=False)
    y_train: pd.Series = field(repr=False)
    groups_train: pd.Series = field(repr=False)
    X_test: pd.DataFrame = field(repr=False)
    y_test: pd.Series = field(repr=False)
    LABELS: List[str]
    SCORING: Dict[str, str]
    cv_splits: List[Tuple[np.ndarray, np.ndarray]] = field(repr=False)
    SPLIT_INFO: Dict[str, Any]
    CV_FINGERPRINTS: List[Dict[str, Any]]
    BASELINES: Dict[str, Any]
    test_used: bool = field(default=False, repr=False)

    def build_preprocessor(self):
        """Return a fresh, unfitted copy of the shared full preprocessor."""
        return build_full_preprocessor()


def standard_setup(csv_path=None, verbose=True) -> Context:
    """Build and validate the canonical dataset, split, folds, and baselines."""
    repo_root = find_repo_root()
    data_path = Path(csv_path) if csv_path is not None else Path(
        "data/raw/Movies_Reviews_modified_version1.csv")
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    df = load_clean(filepath=data_path)
    n_rows = len(df)
    n_movies = df["movie_name"].nunique()
    n_classes = df["emotion"].nunique()
    assert n_rows == 16107, f"Expected 16,107 rows, got {n_rows}"
    assert n_movies == 1309, f"Expected 1,309 movies, got {n_movies}"
    assert n_classes == 7, f"Expected 7 classes, got {n_classes}"
    assert "surprise" not in set(df["emotion"]), "Found 'surprise' in emotion"

    train_df, test_df = make_split(df, test_size=0.2, random_state=42)
    assert len(train_df) == 12886, f"Expected 12,886 train rows, got {len(train_df)}"
    assert len(test_df) == 3221, f"Expected 3,221 test rows, got {len(test_df)}"
    movie_overlap = set(train_df["movie_name"]) & set(test_df["movie_name"])
    text_overlap = set(train_df["cleaned_review"]) & set(test_df["cleaned_review"])
    assert not movie_overlap, f"Expected zero movie overlap, got {len(movie_overlap)}"
    assert not text_overlap, f"Expected zero review-text overlap, got {len(text_overlap)}"

    X_train = train_df
    y_train = train_df["emotion"]
    groups_train = train_df["movie_name"]
    X_test = test_df
    y_test = test_df["emotion"]
    split_info = split_fingerprint(train_df, test_df, random_state=42, test_size=0.2)
    cv_splits = list(get_cv(5).split(X_train, y_train, groups=groups_train))
    cv_fingerprints = fold_fingerprints(cv_splits, groups_train)
    baselines = compute_baselines(y_train, y_test)

    ctx = Context(
        df=df,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_test=X_test,
        y_test=y_test,
        LABELS=list(EVALUATION_LABELS),
        SCORING=dict(SCORING),
        cv_splits=cv_splits,
        SPLIT_INFO=split_info,
        CV_FINGERPRINTS=cv_fingerprints,
        BASELINES=baselines,
    )

    if verbose:
        fold_sizes = [len(val_idx) for _, val_idx in cv_splits]
        print("Invariant             | Value")
        print(f"Rows                  | {n_rows:,}")
        print(f"Movies                | {n_movies:,}")
        print(f"Classes               | {n_classes}")
        print(f"Train / test rows     | {len(train_df):,} / {len(test_df):,}")
        print(f"Validation fold rows  | {fold_sizes}")
        print(
            "Baseline macro-F1     | "
            f"majority={baselines['majority']['macro_f1']:.4f}, "
            f"stratified={baselines['stratified_random']['macro_f1_mean']:.4f}, "
            f"uniform={baselines['uniform_random']['macro_f1_mean']:.4f}"
        )
    return ctx


def cv_scores_from_folds(macro_f1, accuracy, weighted_f1) -> Dict[str, Any]:
    """Convert three five-fold score lists to the shared CV result structure."""
    result: Dict[str, Any] = {"n_splits": 5}
    for name, values in (
        ("macro_f1", macro_f1),
        ("accuracy", accuracy),
        ("weighted_f1", weighted_f1),
    ):
        scores = [float(value) for value in values]
        if len(scores) != 5:
            raise ValueError(f"{name} must contain exactly 5 fold scores")
        result[name] = {
            "folds": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=0)),
        }
    return result


def fold_scores_for(ctx: Context, fit_predict: Callable) -> Tuple[List[float], List[float], List[float]]:
    """Run a custom fold trainer and return macro-F1, accuracy, and weighted-F1 lists."""
    macro_scores: List[float] = []
    accuracy_scores: List[float] = []
    weighted_scores: List[float] = []
    for train_idx, val_idx in ctx.cv_splits:
        predicted = np.asarray(fit_predict(train_idx, val_idx)).reshape(-1)
        if len(predicted) != len(val_idx):
            raise ValueError("fit_predict must return one prediction per validation row")
        unknown = set(predicted) - set(ctx.LABELS)
        if unknown:
            raise ValueError(f"fit_predict returned unknown labels: {sorted(unknown)}")
        truth = ctx.y_train.iloc[val_idx]
        macro_scores.append(float(f1_score(
            truth, predicted, labels=ctx.LABELS, average="macro", zero_division=0)))
        accuracy_scores.append(float(accuracy_score(truth, predicted)))
        weighted_scores.append(float(f1_score(
            truth, predicted, labels=ctx.LABELS, average="weighted", zero_division=0)))
    return macro_scores, accuracy_scores, weighted_scores


class _PredictionModel:
    """Adapt a fixed prediction array to the evaluator's predict interface."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        """Return the fixed predictions after checking the requested row count."""
        if len(X) != len(self.predictions):
            raise ValueError("prediction length does not match the evaluation data")
        return self.predictions


def _validate_search_folds(search, expected_splits) -> None:
    """Require the exact shared validation indices in every search fold."""
    if not isinstance(getattr(search, "cv", None), list):
        raise ValueError(_SHARED_FOLDS_MESSAGE)
    if len(search.cv) != len(expected_splits):
        raise ValueError(_SHARED_FOLDS_MESSAGE)
    for supplied, expected in zip(search.cv, expected_splits):
        if len(supplied) != 2 or not np.array_equal(np.asarray(supplied[1]), expected[1]):
            raise ValueError(_SHARED_FOLDS_MESSAGE)


def finalize_model(
    ctx: Context,
    *,
    model_key,
    member_id,
    model_label,
    algorithm,
    preprocessing,
    search=None,
    model=None,
    y_pred=None,
    cv_scores=None,
    n_configs=None,
    tuning_method=None,
    elapsed_seconds,
    best_params=None,
    out_dir=None,
) -> Tuple[Path, Path]:
    """Evaluate the test set once and save one model in the shared result format."""
    if model_key not in ALLOWED_MODEL_KEYS:
        raise ValueError(f"model_key must be one of {ALLOWED_MODEL_KEYS}")
    if re.fullmatch(r"IT\d{8}", str(member_id)) is None:
        raise ValueError("member_id must match ^IT\\d{8}$")
    if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be >= 0")
    if sum(source is not None for source in (search, model, y_pred)) != 1:
        raise ValueError("provide exactly one prediction source: search, model, or y_pred")

    predictor = model
    if search is not None:
        if not isinstance(search, (GridSearchCV, RandomizedSearchCV)):
            raise ValueError("search must be a fitted GridSearchCV or RandomizedSearchCV")
        if getattr(search, "refit", None) != "macro_f1":
            raise ValueError('search.refit must equal "macro_f1"')
        _validate_search_folds(search, ctx.cv_splits)
        if not hasattr(search, "best_estimator_") or not hasattr(search, "cv_results_"):
            raise ValueError("search must be fitted before finalize_model")
        predictor = search.best_estimator_
        cv_scores = cv_fold_scores(search)
        n_configs = len(search.cv_results_["params"])
        tuning_method = type(search).__name__
        if best_params is None:
            best_params = search.best_params_
    else:
        missing = [
            name
            for name, value in (
                ("cv_scores", cv_scores),
                ("n_configs", n_configs),
                ("tuning_method", tuning_method),
                ("best_params", best_params),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"search=None requires: {', '.join(missing)}")

    if y_pred is not None:
        predictions = np.asarray(y_pred).reshape(-1)
        if len(predictions) != len(ctx.X_test):
            raise ValueError(f"y_pred must contain exactly {len(ctx.X_test)} predictions")
        unknown = set(predictions) - set(ctx.LABELS)
        if unknown:
            raise ValueError(f"y_pred contains unknown labels: {sorted(unknown)}")
        predictor = _PredictionModel(predictions)
    elif not callable(getattr(predictor, "predict", None)):
        raise ValueError("model must be a fitted estimator or pipeline with predict(X)")

    if ctx.test_used:
        raise RuntimeError("the test set was already used; do not tune after seeing test results")
    ctx.test_used = True
    test_metrics, predictions_df = evaluate_on_test(predictor, ctx.X_test, ctx.y_test)

    meta = {
        "model_label": str(model_label),
        "member_id": str(member_id),
        "algorithm": str(algorithm),
        "preprocessing": str(preprocessing),
        "tuning": {
            "method": str(tuning_method),
            "n_configs": int(n_configs),
            "n_folds": int(cv_scores["n_splits"]),
            "scoring": list(SCORING.keys()),
            "elapsed_seconds": float(elapsed_seconds),
            "best_params": dict(best_params),
        },
    }
    destination = Path(out_dir) if out_dir is not None else find_repo_root() / "results/phase2"
    json_path, csv_path = save_model_results(
        out_dir=destination,
        model_key=model_key,
        meta=meta,
        split=ctx.SPLIT_INFO,
        cv=cv_scores,
        cv_fingerprints=ctx.CV_FINGERPRINTS,
        test=test_metrics,
        baselines=ctx.BASELINES,
        predictions_df=predictions_df,
    )
    loaded = load_model_results(json_path)
    validate_results(loaded)
    print(
        f"valid: {json_path.name}, {csv_path.name}, "
        f"test macro-F1 = {test_metrics['macro_f1']:.4f} "
        f"(CV mean = {cv_scores['macro_f1']['mean']:.4f})"
    )
    return json_path, csv_path
