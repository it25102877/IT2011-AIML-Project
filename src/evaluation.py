"""Shared evaluation, metrics, serialization, and bootstrapping utilities for Phase 2."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS: List[str] = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "optimism",
    "sadness",
]

SCHEMA_VERSION: int = 1


def sha256_of_names(names: Iterable[str]) -> str:
    """Return the SHA-256 hex digest of sorted unique names joined by newlines."""
    return hashlib.sha256("\n".join(sorted(set(names))).encode("utf-8")).hexdigest()


def compute_review_id(text: str) -> str:
    """Compute 16-character SHA-1 hexadecimal prefix for UTF-8 review text."""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def split_fingerprint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Compute reproducibility fingerprint and verify zero overlap between train and test splits."""
    train_movies = set(train_df["movie_name"])
    test_movies = set(test_df["movie_name"])
    overlap_movies = train_movies.intersection(test_movies)
    if len(overlap_movies) > 0:
        raise ValueError(f"Movie overlap detected between train and test splits: {len(overlap_movies)} movies")

    train_review_ids = [compute_review_id(t) for t in train_df["Reviews"]]
    test_review_ids = [compute_review_id(t) for t in test_df["Reviews"]]

    if len(set(train_review_ids)) != len(train_review_ids):
        raise ValueError("Duplicate review IDs found within train_df.")
    if len(set(test_review_ids)) != len(test_review_ids):
        raise ValueError("Duplicate review IDs found within test_df.")

    overlap_reviews = set(train_review_ids).intersection(set(test_review_ids))
    if len(overlap_reviews) > 0:
        raise ValueError(f"Review overlap detected between train and test splits: {len(overlap_reviews)} reviews")

    return {
        "random_state": int(random_state),
        "test_size": float(test_size),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_movies": int(len(train_movies)),
        "n_test_movies": int(len(test_movies)),
        "train_movies_sha256": sha256_of_names(train_movies),
        "test_movies_sha256": sha256_of_names(test_movies),
        "train_review_ids_sha256": sha256_of_names(train_review_ids),
        "test_review_ids_sha256": sha256_of_names(test_review_ids),
    }


def fold_fingerprints(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    movie_names: Iterable[str],
) -> List[Dict[str, Any]]:
    """Compute per-fold size and validation movie hash fingerprints while validating zero fold leakage."""
    movie_arr = np.asarray(movie_names)
    records = []
    for fold_idx, (trn_idx, val_idx) in enumerate(cv_splits, start=1):
        trn_movies = set(movie_arr[trn_idx])
        val_movies = set(movie_arr[val_idx])
        overlap = trn_movies.intersection(val_movies)
        if len(overlap) > 0:
            raise ValueError(f"Movie leakage detected in fold {fold_idx}: {len(overlap)} overlapping movies.")

        records.append({
            "fold": int(fold_idx),
            "n_train": int(len(trn_idx)),
            "n_val": int(len(val_idx)),
            "val_movies_sha256": sha256_of_names(val_movies),
        })
    return records


def cv_fold_scores(search: Any) -> Dict[str, Any]:
    """Extract per-fold scores, mean, and population std for macro_f1, accuracy, and weighted_f1."""
    required_metrics = ["macro_f1", "accuracy", "weighted_f1"]
    cv_results = search.cv_results_
    best_idx = search.best_index_

    # Determine number of CV folds from split0_test_macro_f1
    n_splits = 0
    while f"split{n_splits}_test_macro_f1" in cv_results:
        n_splits += 1

    if n_splits == 0:
        raise ValueError("Search results do not contain split{i}_test_macro_f1 entries.")

    results: Dict[str, Any] = {"n_splits": int(n_splits)}
    for metric in required_metrics:
        expected_key = f"mean_test_{metric}"
        if expected_key not in cv_results:
            raise ValueError(f"Search results missing required metric '{metric}'. Expected '{expected_key}'.")

        scores = [
            float(cv_results[f"split{i}_test_{metric}"][best_idx])
            for i in range(n_splits)
        ]
        calc_mean = float(np.mean(scores))
        calc_std = float(np.std(scores, ddof=0))
        expected_std = float(cv_results[f"std_test_{metric}"][best_idx])

        if abs(calc_std - expected_std) > 1e-9:
            raise AssertionError(
                f"Calculated std {calc_std} differs from sklearn std {expected_std} for metric '{metric}'"
            )

        results[metric] = {
            "folds": scores,
            "mean": calc_mean,
            "std": calc_std,
        }

    return results


def compute_baselines(
    y_train: pd.Series,
    y_test: pd.Series,
    n_seeds: int = 200,
) -> Dict[str, Any]:
    """Calculate majority-class, stratified-random, and uniform-random baseline benchmarks."""
    maj_class = str(y_train.value_counts().idxmax())
    y_pred_maj = np.full(shape=len(y_test), fill_value=maj_class)
    maj_acc = float(accuracy_score(y_test, y_pred_maj))
    maj_macro = float(f1_score(y_test, y_pred_maj, labels=LABELS, average="macro", zero_division=0))
    maj_weighted = float(f1_score(y_test, y_pred_maj, labels=LABELS, average="weighted", zero_division=0))

    X_dummy_train = np.zeros((len(y_train), 1))
    X_dummy_test = np.zeros((len(y_test), 1))

    strat_accs, strat_f1s = [], []
    unif_accs, unif_f1s = [], []

    for seed in range(n_seeds):
        clf_strat = DummyClassifier(strategy="stratified", random_state=seed)
        clf_strat.fit(X_dummy_train, y_train)
        pred_strat = clf_strat.predict(X_dummy_test)
        strat_accs.append(accuracy_score(y_test, pred_strat))
        strat_f1s.append(f1_score(y_test, pred_strat, labels=LABELS, average="macro", zero_division=0))

        clf_unif = DummyClassifier(strategy="uniform", random_state=seed)
        clf_unif.fit(X_dummy_train, y_train)
        pred_unif = clf_unif.predict(X_dummy_test)
        unif_accs.append(accuracy_score(y_test, pred_unif))
        unif_f1s.append(f1_score(y_test, pred_unif, labels=LABELS, average="macro", zero_division=0))

    return {
        "majority": {
            "class": maj_class,
            "accuracy": maj_acc,
            "macro_f1": maj_macro,
            "weighted_f1": maj_weighted,
        },
        "stratified_random": {
            "accuracy_mean": float(np.mean(strat_accs)),
            "accuracy_std": float(np.std(strat_accs)),
            "macro_f1_mean": float(np.mean(strat_f1s)),
            "macro_f1_std": float(np.std(strat_f1s)),
        },
        "uniform_random": {
            "accuracy_mean": float(np.mean(unif_accs)),
            "accuracy_std": float(np.std(unif_accs)),
            "macro_f1_mean": float(np.mean(unif_f1s)),
            "macro_f1_std": float(np.std(unif_f1s)),
        },
    }


def evaluate_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute single predict call on test set, returning comprehensive metrics and predictions DataFrame."""
    y_pred = model.predict(X_test)

    review_ids = [compute_review_id(t) for t in X_test["Reviews"]]
    if len(set(review_ids)) != len(review_ids):
        raise ValueError("Non-unique review_id encountered across test set.")

    predictions_df = pd.DataFrame({
        "row_id": list(range(len(X_test))),
        "review_id": review_ids,
        "movie_name": list(X_test["movie_name"]),
        "y_true": list(y_test),
        "y_pred": list(y_pred),
    })

    acc = float(accuracy_score(y_test, y_pred))
    macro = float(f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0))
    weighted = float(f1_score(y_test, y_pred, labels=LABELS, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=LABELS).tolist()

    rep = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    per_class = {
        lbl: {
            "precision": float(rep[lbl]["precision"]),
            "recall": float(rep[lbl]["recall"]),
            "f1": float(rep[lbl]["f1-score"]),
            "support": int(rep[lbl]["support"]),
        }
        for lbl in LABELS
    }

    metrics = {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
    }

    return metrics, predictions_df


def confusion_matrix_normalized(cm: Any) -> np.ndarray:
    """Return row-normalized confusion matrix where rows with zero support remain zero."""
    cm_arr = np.asarray(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    return np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)


def macro_f1_from_confusion(cm: Any) -> float:
    """Compute Macro-F1 score directly from 7x7 confusion matrix matching sklearn definitions."""
    cm_arr = np.asarray(cm, dtype=float)
    tp = np.diag(cm_arr)
    row_sum = cm_arr.sum(axis=1)
    col_sum = cm_arr.sum(axis=0)
    denom = row_sum + col_sum
    f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return float(np.mean(f1))


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and non-string keys for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def save_model_results(
    out_dir: Path | str,
    model_key: str,
    meta: Dict[str, Any],
    split: Dict[str, Any],
    cv: Dict[str, Any],
    cv_fingerprints: List[Dict[str, Any]],
    test: Dict[str, Any],
    baselines: Dict[str, Any],
    predictions_df: pd.DataFrame,
) -> Tuple[Path, Path]:
    """Validate and persist model evaluation metrics JSON and test predictions CSV."""
    required_meta_keys = ["model_label", "member_id", "algorithm", "preprocessing", "tuning"]
    for k in required_meta_keys:
        if k not in meta:
            raise ValueError(f"meta dictionary missing required key: '{k}'")

    required_tuning_keys = ["method", "n_configs", "n_folds", "scoring", "elapsed_seconds", "best_params"]
    for k in required_tuning_keys:
        if k not in meta["tuning"]:
            raise ValueError(f"meta['tuning'] dictionary missing required key: '{k}'")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_key": str(model_key),
        "labels": list(LABELS),
        "created_with": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "meta": meta,
        "split": split,
        "cv": cv,
        "cv_fingerprints": cv_fingerprints,
        "test": test,
        "baselines": baselines,
    }

    validate_results(payload)

    # Validate predictions_df BEFORE writing any files
    expected_cols = ["row_id", "review_id", "movie_name", "y_true", "y_pred"]
    if list(predictions_df.columns) != expected_cols:
        raise ValueError(
            f"predictions_df columns mismatch: expected exactly {expected_cols}, got {list(predictions_df.columns)}"
        )

    if len(predictions_df) != split["n_test"]:
        raise ValueError(
            f"predictions_df row count ({len(predictions_df)}) does not match split.n_test ({split['n_test']})"
        )

    if predictions_df["review_id"].nunique() != len(predictions_df):
        raise ValueError("predictions_df contains duplicate review_id values.")

    invalid_true = set(predictions_df["y_true"]) - set(LABELS)
    if invalid_true:
        raise ValueError(f"predictions_df['y_true'] contains invalid labels not in LABELS: {invalid_true}")

    invalid_pred = set(predictions_df["y_pred"]) - set(LABELS)
    if invalid_pred:
        raise ValueError(f"predictions_df['y_pred'] contains invalid labels not in LABELS: {invalid_pred}")

    recomputed_cm = confusion_matrix(predictions_df["y_true"], predictions_df["y_pred"], labels=LABELS)
    expected_cm = np.asarray(test["confusion_matrix"])
    if not np.array_equal(recomputed_cm, expected_cm):
        raise ValueError(
            "Recomputed 7x7 confusion matrix from predictions_df does not match test['confusion_matrix'] exactly."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / f"{model_key}_results.json"
    csv_file = out_path / f"{model_key}_test_predictions.csv"

    predictions_df.to_csv(csv_file, index=False)

    sanitized = _sanitize_for_json(payload)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    return json_file, csv_file


def validate_results(d: Dict[str, Any]) -> None:
    """Validate results schema, consistency of confusion matrix, class supports, and score invariants."""
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {SCHEMA_VERSION}, got {d.get('schema_version')}"
        )

    required_top_keys = [
        "schema_version",
        "model_key",
        "labels",
        "created_with",
        "meta",
        "split",
        "cv",
        "cv_fingerprints",
        "test",
        "baselines",
    ]
    for k in required_top_keys:
        if k not in d:
            raise ValueError(f"Missing required top-level key: '{k}'")

    if d["labels"] != LABELS:
        raise ValueError(f"Labels mismatch: expected {LABELS}, got {d.get('labels')}")

    split = d["split"]
    for req_hash in ["train_review_ids_sha256", "test_review_ids_sha256"]:
        if req_hash not in split:
            raise ValueError(f"split dictionary missing required hash key '{req_hash}'")

    cm = np.asarray(d["test"]["confusion_matrix"])
    if cm.shape != (7, 7):
        raise ValueError(f"Confusion matrix shape mismatch: expected (7, 7), got {cm.shape}")

    total_support = int(cm.sum())
    if total_support != split["n_test"]:
        raise ValueError(
            f"Confusion matrix sum ({total_support}) does not match split.n_test ({split['n_test']})"
        )

    per_class = d["test"]["per_class"]
    for idx, lbl in enumerate(LABELS):
        expected_support = int(cm[idx].sum())
        recorded_support = per_class[lbl]["support"]
        if recorded_support != expected_support:
            raise ValueError(
                f"Class '{lbl}' support mismatch: confusion matrix row sum is {expected_support}, "
                f"but per_class recorded {recorded_support}."
            )

    recomputed_f1 = macro_f1_from_confusion(cm)
    recorded_f1 = d["test"]["macro_f1"]
    if abs(recomputed_f1 - recorded_f1) > 1e-9:
        raise ValueError(
            f"Recomputed Macro-F1 ({recomputed_f1}) differs from recorded test.macro_f1 ({recorded_f1}) by > 1e-9."
        )

    cv_info = d["cv"]
    n_splits = cv_info["n_splits"]
    for metric in ["macro_f1", "accuracy", "weighted_f1"]:
        if metric not in cv_info:
            raise ValueError(f"cv dictionary missing metric: '{metric}'")
        folds_len = len(cv_info[metric]["folds"])
        if folds_len != n_splits:
            raise ValueError(
                f"CV metric '{metric}' folds count ({folds_len}) does not match cv.n_splits ({n_splits})"
            )


def load_model_results(path: Path | str) -> Dict[str, Any]:
    """Load, validate, and return model results dictionary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_results(data)
    return data


def movie_confusion_tensor(predictions_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Compute 3D confusion tensor of shape (n_movies, 7, 7) indexing per-movie confusion matrices."""
    movies = sorted(predictions_df["movie_name"].unique())
    n_movies = len(movies)
    tensor = np.zeros((n_movies, 7, 7), dtype=np.int64)

    movie_to_idx = {m: i for i, m in enumerate(movies)}
    label_to_idx = {l: i for i, l in enumerate(LABELS)}

    # Map labels to indices for fast counting
    y_true_indices = [label_to_idx[l] for l in predictions_df["y_true"]]
    y_pred_indices = [label_to_idx[l] for l in predictions_df["y_pred"]]
    movie_indices = [movie_to_idx[m] for m in predictions_df["movie_name"]]

    for m_idx, t_idx, p_idx in zip(movie_indices, y_true_indices, y_pred_indices):
        tensor[m_idx, t_idx, p_idx] += 1

    return movies, tensor


def make_bootstrap_indices(
    n_movies: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Generate (n_boot, n_movies) array of bootstrap movie sample indices drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_movies, size=(n_boot, n_movies), dtype=np.int64)


def bootstrap_macro_f1(tensor: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Compute vectorised Macro-F1 bootstrap draws from movie confusion tensor and sample indices."""
    cms = tensor[idx].sum(axis=1)  # shape: (n_boot, 7, 7)
    diag = np.diagonal(cms, axis1=1, axis2=2)  # shape: (n_boot, 7)
    row_sum = cms.sum(axis=2)
    col_sum = cms.sum(axis=1)
    denom = row_sum + col_sum
    f1 = np.where(denom > 0, 2.0 * diag / denom, 0.0)
    return f1.mean(axis=1)


def paired_bootstrap_difference(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Compute paired differences in Macro-F1 draws between model A and model B on identical samples."""
    draws_a = bootstrap_macro_f1(tensor_a, idx)
    draws_b = bootstrap_macro_f1(tensor_b, idx)
    return draws_a - draws_b


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """Compute empirical percentile confidence interval bounds (low, high) at the specified level."""
    alpha = (1.0 - level) / 2.0
    low = float(np.percentile(draws, 100.0 * alpha))
    high = float(np.percentile(draws, 100.0 * (1.0 - alpha)))
    return low, high
