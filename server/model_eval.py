"""Pure NumPy evaluation for the deployed seven-parent GalleryFL head."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FEATURE_DIM = 1024
HIDDEN_SIZE = 256
LABELS = ("people", "places", "activities", "objects", "documents", "nature", "events")
NUM_CLASSES = len(LABELS)


def _validate_head(weights: Sequence[np.ndarray]) -> None:
    expected = ((FEATURE_DIM, HIDDEN_SIZE), (HIDDEN_SIZE,), (HIDDEN_SIZE, NUM_CLASSES), (NUM_CLASSES,))
    if len(weights) != len(expected):
        raise ValueError(f"Expected four head tensors, got {len(weights)}")
    for value, shape in zip(weights, expected):
        if np.asarray(value).shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid head tensor; expected finite {shape}")


def probabilities(features: np.ndarray, weights: Sequence[np.ndarray]) -> np.ndarray:
    _validate_head(weights)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected features shaped (N,{FEATURE_DIM}), got {features.shape}")
    w1, b1, w2, b2 = weights
    hidden = np.maximum(features @ w1 + b1, 0.0)
    logits = hidden @ w2 + b2
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)


def multiclass_report(labels: np.ndarray, probs: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(probs).argmax(axis=1)
    per_class = {}
    precisions = []
    recalls = []
    f1_values = []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    for class_id, label in enumerate(LABELS):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        support = int(confusion[class_id, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    selected = probs[np.arange(labels.size), labels]
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "loss": float(-np.log(np.clip(selected, 1e-8, 1.0)).mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "num_samples": int(labels.size),
    }


def evaluate(features: np.ndarray, labels: np.ndarray, weights: Sequence[np.ndarray]) -> dict:
    return multiclass_report(labels, probabilities(features, weights))


def atomic_write_json(path: str | Path, payload: Mapping | Sequence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
