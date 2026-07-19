#!/usr/bin/env python3
"""Evaluate a checkpoint on a held-out single-label test manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backbone import FrozenBackbone
from config import (
    EVAL_FEATURES_PATH,
    HISTORICAL_MACRO_F1,
    LABELS,
    MIN_ACCEPTABLE_TEST_MACRO_F1,
    NUM_CLASSES,
    OUTPUT_DIR,
    PROJECT_DIR,
    TEST_METRICS_PATH,
    THRESHOLDS_PATH,
    ensure_output_dirs,
)
from pipeline import (
    PipelineError,
    atomic_save_npz,
    atomic_write_json,
    cross_entropy,
    default_threshold_payload,
    file_sha256,
    fold_feature_standardization,
    forward_logits,
    load_checkpoint,
    load_manifest,
    multiclass_metrics,
    softmax,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-macro-f1",
        type=float,
        default=MIN_ACCEPTABLE_TEST_MACRO_F1,
        help="write the report but exit non-zero if the held-out score is below this value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    items = load_manifest(args.test_manifest, dataset_dir)
    labels = np.asarray([item.label for item in items], dtype=np.int64)
    missing = [LABELS[index] for index in range(NUM_CLASSES) if not np.any(labels == index)]
    if missing:
        raise PipelineError(f"Test manifest has no samples for: {', '.join(missing)}")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    backbone = FrozenBackbone()
    raw_features = backbone.extract([item.path for item in items], "test")
    normalized_features = standardize(
        raw_features,
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
    )
    normalized_logits = forward_logits(
        normalized_features,
        checkpoint["w1"],
        checkpoint["b1"],
        checkpoint["w2"],
        checkpoint["b2"],
    )
    probabilities = softmax(normalized_logits)

    # Prove that the four exported tensors produce the same result on raw
    # Android backbone features after normalization is folded into layer one.
    folded_w1, folded_b1 = fold_feature_standardization(
        checkpoint["w1"],
        checkpoint["b1"],
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
    )
    deployment_logits = forward_logits(
        raw_features,
        folded_w1,
        folded_b1,
        checkpoint["w2"],
        checkpoint["b2"],
    )
    max_export_error = float(np.max(np.abs(normalized_logits - deployment_logits)))
    if max_export_error > 2e-3:
        raise PipelineError(
            f"Normalization folding changed logits by {max_export_error}; refusing evaluation"
        )

    report = multiclass_metrics(labels, probabilities)
    report["cross_entropy"] = cross_entropy(labels, probabilities)
    payload = {
        "schema_version": 1,
        "task": "single_label_multiclass",
        "decision_rule": "argmax",
        "labels": list(LABELS),
        **report,
        "checkpoint": (
            checkpoint_path.relative_to(PROJECT_DIR).as_posix()
            if checkpoint_path.is_relative_to(PROJECT_DIR)
            else checkpoint_path.name
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_manifest": "$DATASET_DIR/manifests/test.csv",
        "dataset": {
            "name": "COCO Minitrain 10K" if (dataset_dir / "labels" / "train2017").is_dir() else "external seven-parent gallery dataset",
            "source": "https://www.kaggle.com/datasets/banuprasadb/coco-minitrain-10k" if (dataset_dir / "labels" / "train2017").is_dir() else None,
        },
        "backbone_sha256": backbone.sha256,
        "deployment_equivalence_max_abs_logit_error": max_export_error,
        "comparison": {
            "old_test_macro_f1": HISTORICAL_MACRO_F1,
            "old_test_accuracy": None,
            "new_test_macro_f1": report["macro_f1"],
            "new_test_accuracy": report["accuracy"],
            "macro_f1_delta": report["macro_f1"] - HISTORICAL_MACRO_F1,
            "note": "Historical accuracy was not present in the repository, so it is not fabricated.",
        },
    }
    atomic_write_json(TEST_METRICS_PATH, payload)
    atomic_write_json(OUTPUT_DIR.parent / "bootstrap_eval_report.json", payload)
    atomic_write_json(THRESHOLDS_PATH, default_threshold_payload())
    atomic_save_npz(
        EVAL_FEATURES_PATH,
        features=raw_features.astype(np.float32),
        labels=labels.astype(np.int64),
        label_names=np.asarray(LABELS),
        backbone_sha256=np.asarray(backbone.sha256),
    )

    print(f"test macro F1: {report['macro_f1']:.4f}")
    print(f"test accuracy: {report['accuracy']:.4f}")
    print("per-class metrics:")
    for label in LABELS:
        metrics = report["per_class"][label]
        print(
            f"  {label:12s} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"support={metrics['support']}"
        )
    print(f"report: {TEST_METRICS_PATH}")

    if report["macro_f1"] < args.minimum_macro_f1:
        print(
            f"QUALITY GATE FAILED: macro F1 {report['macro_f1']:.4f} is below "
            f"{args.minimum_macro_f1:.4f}. Do not export this checkpoint."
        )
        return 2
    print("quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
