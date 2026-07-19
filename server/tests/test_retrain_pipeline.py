import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrain"))

from config import FEATURE_DIM, HIDDEN_SIZE, NUM_CLASSES
from pipeline import (
    fold_feature_standardization,
    forward_logits,
    multiclass_metrics,
    softmax,
    standardize,
)


def test_feature_normalization_folding_matches_runtime_head():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, FEATURE_DIM)).astype(np.float32)
    mean = rng.normal(scale=0.2, size=FEATURE_DIM).astype(np.float32)
    std = rng.uniform(0.2, 2.0, size=FEATURE_DIM).astype(np.float32)
    w1 = rng.normal(scale=0.03, size=(FEATURE_DIM, HIDDEN_SIZE)).astype(np.float32)
    b1 = rng.normal(scale=0.02, size=HIDDEN_SIZE).astype(np.float32)
    w2 = rng.normal(scale=0.03, size=(HIDDEN_SIZE, NUM_CLASSES)).astype(np.float32)
    b2 = rng.normal(scale=0.02, size=NUM_CLASSES).astype(np.float32)

    expected = forward_logits(standardize(x, mean, std), w1, b1, w2, b2)
    folded_w1, folded_b1 = fold_feature_standardization(w1, b1, mean, std)
    actual = forward_logits(x, folded_w1, folded_b1, w2, b2)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)


def test_argmax_metrics_are_single_label_and_include_all_classes():
    labels = np.arange(NUM_CLASSES, dtype=np.int64)
    logits = np.full((NUM_CLASSES, NUM_CLASSES), -5.0, dtype=np.float32)
    logits[np.arange(NUM_CLASSES), labels] = 5.0
    probabilities = softmax(logits)
    report = multiclass_metrics(labels, probabilities)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert set(report["per_class"]) == {
        "people", "places", "activities", "objects", "documents", "nature", "events"
    }


def test_softmax_is_stable_and_normalized():
    probabilities = softmax(np.asarray([[10000.0] + [-10000.0] * (NUM_CLASSES - 1)]))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.argmax(axis=1).item() == 0
