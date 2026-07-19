"""Pure-Python/NumPy utilities shared by training, evaluation, export, and FL.

This module deliberately does not import TensorFlow. The live FL server can use
it to evaluate an exported head without loading the training framework.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:  # package import (FL server/tests)
    from .config import (
        FEATURE_DIM,
        FEATURE_STD_EPSILON,
        HIDDEN_SIZE,
        LABELS,
        LABEL_TO_IDX,
        NUM_CLASSES,
    )
except ImportError:  # direct script execution from server/retrain
    from config import (
        FEATURE_DIM,
        FEATURE_STD_EPSILON,
        HIDDEN_SIZE,
        LABELS,
        LABEL_TO_IDX,
        NUM_CLASSES,
    )


class PipelineError(RuntimeError):
    """Raised for invalid data or incompatible model artifacts."""


@dataclass(frozen=True)
class ManifestItem:
    path: Path
    label: int


def _resolve_image_path(raw_path: str, dataset_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()


def load_manifest(manifest_path: Path | str, dataset_dir: Path | str) -> list[ManifestItem]:
    """Load a single-label CSV manifest and validate every row.

    New manifests use ``path,label,class_index``. ``image_path`` is accepted for
    compatibility with already prepared datasets.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not manifest_path.is_file():
        raise PipelineError(f"Manifest not found: {manifest_path}")

    items: list[ManifestItem] = []
    seen: set[Path] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        path_field = "path" if "path" in fields else "image_path" if "image_path" in fields else None
        if path_field is None or "label" not in fields:
            raise PipelineError(
                f"{manifest_path} must contain path (or image_path) and label columns"
            )

        for line_number, row in enumerate(reader, start=2):
            label_name = (row.get("label") or "").strip().lower()
            if label_name not in LABEL_TO_IDX:
                raise PipelineError(
                    f"{manifest_path}:{line_number}: unknown label {label_name!r}; "
                    f"expected one of {list(LABELS)}"
                )
            raw_path = (row.get(path_field) or "").strip()
            if not raw_path:
                raise PipelineError(f"{manifest_path}:{line_number}: empty image path")
            image_path = _resolve_image_path(raw_path, dataset_dir)
            if not image_path.is_file():
                raise PipelineError(
                    f"{manifest_path}:{line_number}: image not found: {image_path}"
                )
            if image_path in seen:
                raise PipelineError(
                    f"{manifest_path}:{line_number}: duplicate image path: {image_path}"
                )
            seen.add(image_path)

            label = LABEL_TO_IDX[label_name]
            raw_index = (row.get("class_index") or "").strip()
            if raw_index and int(raw_index) != label:
                raise PipelineError(
                    f"{manifest_path}:{line_number}: class_index {raw_index} does not match {label_name}"
                )
            items.append(ManifestItem(image_path, label))

    if not items:
        raise PipelineError(f"Manifest is empty: {manifest_path}")
    return items


def validate_split_manifests(splits: Mapping[str, Sequence[ManifestItem]]) -> None:
    """Fail on leakage or a missing class in any split."""

    owner: dict[Path, str] = {}
    for split_name, items in splits.items():
        labels = np.asarray([item.label for item in items], dtype=np.int64)
        missing = [LABELS[i] for i in range(NUM_CLASSES) if not np.any(labels == i)]
        if missing:
            raise PipelineError(f"{split_name} split has no samples for: {', '.join(missing)}")
        for item in items:
            previous = owner.get(item.path)
            if previous is not None:
                raise PipelineError(
                    f"Data leakage: {item.path} appears in both {previous} and {split_name}"
                )
            owner[item.path] = split_name


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        missing = [LABELS[i] for i, count in enumerate(counts) if count == 0]
        raise PipelineError(f"Training split has no samples for: {', '.join(missing)}")
    weights = labels.size / (NUM_CLASSES * counts.astype(np.float64))
    return {index: float(weight) for index, weight in enumerate(weights)}


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise PipelineError(f"Expected features shaped (N, {FEATURE_DIM}), got {features.shape}")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, FEATURE_STD_EPSILON).astype(np.float32)
    return mean, std


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)


def fold_feature_standardization(
    w1: np.ndarray,
    b1: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``(x - mean) / std`` into the first dense layer.

    Android and the FL server consume raw backbone features and only understand
    four head tensors. Folding makes their raw-feature forward pass exactly
    equivalent to the standardized training graph.
    """

    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if w1.shape != (FEATURE_DIM, HIDDEN_SIZE):
        raise PipelineError(f"w1 has shape {w1.shape}, expected {(FEATURE_DIM, HIDDEN_SIZE)}")
    if b1.shape != (HIDDEN_SIZE,) or mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
        raise PipelineError("Invalid b1/feature normalization shape")
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise PipelineError("Feature standard deviations must be finite and positive")

    folded_w1 = w1 / std[:, None]
    folded_b1 = b1 - (mean / std) @ w1
    return folded_w1.astype(np.float32), folded_b1.astype(np.float32)


def validate_head_arrays(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
    expected = {
        "w1": (FEATURE_DIM, HIDDEN_SIZE),
        "b1": (HIDDEN_SIZE,),
        "w2": (HIDDEN_SIZE, NUM_CLASSES),
        "b2": (NUM_CLASSES,),
    }
    for name, value in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
        array = np.asarray(value)
        if array.shape != expected[name]:
            raise PipelineError(f"{name} has shape {array.shape}, expected {expected[name]}")
        if array.dtype.kind != "f" or not np.all(np.isfinite(array)):
            raise PipelineError(f"{name} must contain finite floating-point values")


def forward_logits(
    features: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    validate_head_arrays(w1, b1, w2, b2)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise PipelineError(f"Expected features shaped (N, {FEATURE_DIM}), got {features.shape}")
    hidden = np.maximum(features @ w1 + b1, 0.0)
    return hidden @ w2 + b2


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (y_true.size, NUM_CLASSES):
        raise PipelineError(
            f"Probabilities have shape {probabilities.shape}, expected {(y_true.size, NUM_CLASSES)}"
        )
    predictions = probabilities.argmax(axis=1)
    class_ids = np.arange(NUM_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=class_ids,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=class_ids).tolist(),
        "num_samples": int(y_true.size),
    }


def cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    selected = probabilities[np.arange(y_true.size), y_true]
    return float(-np.log(np.clip(selected, 1e-8, 1.0)).mean())


def load_checkpoint(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"Checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ("w1", "b1", "w2", "b2", "feature_mean", "feature_std")
        missing = [key for key in required if key not in data]
        if missing:
            raise PipelineError(
                f"Checkpoint {path} is incompatible; missing {', '.join(missing)}. Retrain it."
            )
        result = {key: np.asarray(data[key], dtype=np.float32) for key in required}
        if "labels" in data:
            labels = tuple(str(value) for value in data["labels"].tolist())
            if labels != LABELS:
                raise PipelineError(f"Checkpoint labels {labels} do not match {LABELS}")
    validate_head_arrays(result["w1"], result["b1"], result["w2"], result["b2"])
    if result["feature_mean"].shape != (FEATURE_DIM,) or result["feature_std"].shape != (FEATURE_DIM,):
        raise PipelineError("Checkpoint feature normalization has an invalid shape")
    return result


def load_production_head(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in ("w1", "b1", "w2", "b2") if key not in data]
        if missing:
            raise PipelineError(f"Production head is missing: {', '.join(missing)}")
        result = {key: np.asarray(data[key], dtype=np.float32) for key in ("w1", "b1", "w2", "b2")}
    validate_head_arrays(**result)
    return result


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npz(path: Path | str, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temp_name = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path | str, payload: Mapping | Sequence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def default_threshold_payload() -> dict:
    return {
        "schema_version": 1,
        "task": "single_label_multiclass",
        "labels": list(LABELS),
        "decision_rule": "argmax",
        "confidence_gate_semantics": "optional abstention after argmax; never used to create multi-label predictions",
        "confidence_thresholds": {label: 0.0 for label in LABELS},
    }
