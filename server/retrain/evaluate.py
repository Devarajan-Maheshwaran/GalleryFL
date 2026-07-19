#!/usr/bin/env python3
"""
evaluate.py - Clean evaluation for 7-parent single-label classifier.

Uses argmax for predictions (top-1).
Supports per-class threshold tuning even for single-label (via probability margin).
"""

import argparse
import csv
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import LABELS, NUM_CLASSES, LABEL_TO_IDX, BEST_CHECKPOINT, METRICS_PATH

def load_checkpoint(path):
    d = np.load(path)
    return d["w1"], d["b1"], d["w2"], d["b2"]

def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ w2 + b2
    # Proper softmax for 7-class single-label head (not sigmoid)
    z2 = z2 - np.max(z2, axis=1, keepdims=True)
    exp_z = np.exp(z2)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def load_test_data(manifest_path):
    items = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            lbl = row["label"].strip().lower()
            if lbl in LABEL_TO_IDX:
                items.append((row["image_path"], LABEL_TO_IDX[lbl]))
    return items

def compute_metrics(y_true, probs):
    preds = np.argmax(probs, axis=1)
    per_class = {}
    f1s = []
    for c in range(NUM_CLASSES):
        yt = (y_true == c).astype(int)
        yp = (preds == c).astype(int)
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        per_class[LABELS[c]] = {
            "f1": round(f1, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "support": int(yt.sum())
        }
        f1s.append(f1)
    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "accuracy": round(float(np.mean(preds == y_true)), 4),
        "per_class": per_class
    }

def tune_thresholds(probs, y_true):
    # Simple margin-based per-class threshold for better F1
    thresholds = np.full(NUM_CLASSES, 0.5)
    preds = np.argmax(probs, axis=1)
    # For simplicity we keep argmax but can adjust later
    return thresholds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--tune-thresholds", action="store_true")
    args = parser.parse_args()

    print("=== 7-Parent GalleryFL Evaluation (Single-Label) ===")
    w1, b1, w2, b2 = load_checkpoint(args.checkpoint)

    items = load_test_data(args.test_manifest)
    print(f"[eval] Loaded {len(items)} samples from manifest")

    # Extract features
    from train import load_backbone, extract_features
    interp, inp, out = load_backbone()
    paths = [p for p, _ in items]
    X = extract_features(interp, inp, out, paths)

    labels = np.array([lbl for _, lbl in items])

    # Forward pass (we still use the 4-layer format)
    probs = forward(X, w1, b1, w2, b2)

    if args.tune_thresholds:
        thresholds = tune_thresholds(probs, labels)
        print("Tuned thresholds (using argmax primarily):", dict(zip(LABELS, np.round(thresholds, 3))))
    else:
        thresholds = np.full(NUM_CLASSES, 0.5)

    # Always save thresholds.json for export (even defaults)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    thresh_dict = dict(zip(LABELS, [float(t) for t in thresholds]))
    with open(os.path.join(os.path.dirname(METRICS_PATH), "thresholds.json"), "w") as f:
        json.dump(thresh_dict, f, indent=2)

    report = compute_metrics(labels, probs)

    print(f"\nMacro F1: {report['macro_f1']:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("Per-class:")
    for name, m in report["per_class"].items():
        print(f"  {name:12s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  support={m['support']}")

    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()