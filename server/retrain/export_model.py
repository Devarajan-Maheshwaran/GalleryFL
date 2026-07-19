#!/usr/bin/env python3
"""Atomically promote a validated checkpoint to GalleryFL runtime artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import (
    BACKBONE_PATH,
    BEST_CHECKPOINT,
    DEPLOY_THRESHOLDS_PATH,
    FEATURE_DIM,
    FINAL_MODEL_INFO_PATH,
    HEAD_WEIGHTS_PATH,
    HIDDEN_SIZE,
    LABELS,
    MIN_ACCEPTABLE_TEST_MACRO_F1,
    MODEL_SCHEMA_PATH,
    MODEL_VERSION_FILE,
    NUM_CLASSES,
    TEST_METRICS_PATH,
    THRESHOLDS_PATH,
    ensure_output_dirs,
)
from pipeline import (
    PipelineError,
    atomic_save_npz,
    atomic_write_json,
    default_threshold_payload,
    file_sha256,
    fold_feature_standardization,
    forward_logits,
    load_checkpoint,
    load_production_head,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BEST_CHECKPOINT)
    return parser.parse_args()


def next_version() -> int:
    if not MODEL_VERSION_FILE.is_file():
        return 1
    try:
        return int(MODEL_VERSION_FILE.read_text(encoding="utf-8").strip()) + 1
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Invalid model version file: {MODEL_VERSION_FILE}") from exc


def validated_test_report(checkpoint_path: Path) -> dict:
    if not TEST_METRICS_PATH.is_file():
        raise PipelineError(
            f"Held-out report not found: {TEST_METRICS_PATH}. Run evaluate.py before export."
        )
    try:
        report = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Invalid held-out report: {TEST_METRICS_PATH}") from exc
    if report.get("task") != "single_label_multiclass" or tuple(report.get("labels", [])) != LABELS:
        raise PipelineError("Held-out report does not match the seven-parent task")
    reported_sha256 = report.get("checkpoint_sha256")
    actual_sha256 = file_sha256(checkpoint_path)
    if reported_sha256 != actual_sha256:
        raise PipelineError(
            "Held-out report was produced from a different checkpoint "
            f"({reported_sha256!r} != {actual_sha256!r})"
        )
    macro_f1 = float(report.get("macro_f1", -1.0))
    if macro_f1 < MIN_ACCEPTABLE_TEST_MACRO_F1:
        raise PipelineError(
            f"Refusing export: held-out macro F1 {macro_f1:.4f} is below "
            f"{MIN_ACCEPTABLE_TEST_MACRO_F1:.4f}"
        )
    return report


def validated_thresholds() -> dict:
    if not THRESHOLDS_PATH.is_file():
        return default_threshold_payload()
    try:
        payload = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Invalid thresholds file: {THRESHOLDS_PATH}") from exc
    if (
        payload.get("task") != "single_label_multiclass"
        or tuple(payload.get("labels", [])) != LABELS
        or payload.get("decision_rule") != "argmax"
    ):
        raise PipelineError("Threshold metadata is not compatible with the seven-parent argmax head")
    return payload


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    test_report = validated_test_report(checkpoint_path)
    version = next_version()
    thresholds = validated_thresholds()
    folded_w1, folded_b1 = fold_feature_standardization(
        checkpoint["w1"],
        checkpoint["b1"],
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
    )

    # Numerical deployment-equivalence test before touching production files.
    rng = np.random.default_rng(1234)
    probe = (
        checkpoint["feature_mean"]
        + rng.normal(size=(8, FEATURE_DIM)).astype(np.float32) * checkpoint["feature_std"]
    ).astype(np.float32)
    training_logits = forward_logits(
        standardize(probe, checkpoint["feature_mean"], checkpoint["feature_std"]),
        checkpoint["w1"],
        checkpoint["b1"],
        checkpoint["w2"],
        checkpoint["b2"],
    )
    deployment_logits = forward_logits(
        probe,
        folded_w1,
        folded_b1,
        checkpoint["w2"],
        checkpoint["b2"],
    )
    equivalence_error = float(np.max(np.abs(training_logits - deployment_logits)))
    if equivalence_error > 2e-3:
        raise PipelineError(
            f"Refusing export: folded head logit error {equivalence_error} is too large"
        )

    # The production NPZ intentionally has exactly the four tensors understood
    # by ModelManager, WeightSerializer, and Android ClassificationHead.
    atomic_save_npz(
        HEAD_WEIGHTS_PATH,
        w1=folded_w1,
        b1=folded_b1,
        w2=checkpoint["w2"].astype(np.float32),
        b2=checkpoint["b2"].astype(np.float32),
    )
    load_production_head(HEAD_WEIGHTS_PATH)  # read-back validation

    schema = {
        "schema_version": 1,
        "task": "single_label_multiclass",
        "decision_rule": "argmax",
        "num_classes": NUM_CLASSES,
        "labels": list(LABELS),
        "layers": [
            {"name": "w1", "shape": [FEATURE_DIM, HIDDEN_SIZE], "dtype": "float32"},
            {"name": "b1", "shape": [HIDDEN_SIZE], "dtype": "float32"},
            {"name": "w2", "shape": [HIDDEN_SIZE, NUM_CLASSES], "dtype": "float32"},
            {"name": "b2", "shape": [NUM_CLASSES], "dtype": "float32"},
        ],
        "feature_normalization": "folded_into_w1_b1",
    }
    atomic_write_json(MODEL_SCHEMA_PATH, schema)

    atomic_write_json(THRESHOLDS_PATH, thresholds)
    atomic_write_json(DEPLOY_THRESHOLDS_PATH, thresholds)

    MODEL_VERSION_FILE.write_text(f"{version}\n", encoding="utf-8")
    model_info = {
        "model_version": version,
        "head": "server/models/head_weights.npz",
        "head_sha256": file_sha256(HEAD_WEIGHTS_PATH),
        "checkpoint": test_report["checkpoint"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "backbone": "server/models/base_model.tflite",
        "backbone_sha256": file_sha256(BACKBONE_PATH),
        "num_classes": NUM_CLASSES,
        "held_out_macro_f1": float(test_report["macro_f1"]),
        "held_out_accuracy": float(test_report["accuracy"]),
        "labels": list(LABELS),
        "architecture": f"{FEATURE_DIM} -> {HIDDEN_SIZE} ReLU -> {NUM_CLASSES} logits",
        "feature_normalization": "folded into the deployed first dense layer",
        "deployment_equivalence_max_abs_logit_error": equivalence_error,
    }
    atomic_write_json(FINAL_MODEL_INFO_PATH, model_info)

    print(f"deployed head: {HEAD_WEIGHTS_PATH}")
    print(f"model schema: {MODEL_SCHEMA_PATH}")
    print(f"threshold metadata: {DEPLOY_THRESHOLDS_PATH}")
    print(f"model version: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
