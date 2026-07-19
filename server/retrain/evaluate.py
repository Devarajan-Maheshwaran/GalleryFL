#!/usr/bin/env python3
"""
evaluate.py - Clean single-label evaluation for 7-parent GalleryFL head.

- Uses argmax (top-1) for predictions
- Consistent with training (softmax)
- Supports test manifest
- Saves metrics + default thresholds
"""

import argparse
import csv
import json
import os
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import LABELS, NUM_CLASSES, LABEL_TO_IDX, BEST_CHECKPOINT, METRICS_PATH, THRESHOLDS_PATH

def load_checkpoint(path):
    d = np.load(path)
    return d["w1"], d["b1"], d["w2"], d["b2"]

def forward_softmax(X, w1, b1, w2, b2):
    """Pure numpy forward pass matching the trained head (1024→256→7 softmax)"""
    z1 = X @ w1 + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ w2 + b2
    # numerically stable softmax
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

def compute_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = float(np.mean(y_pred == y_true))

    per_class = {}
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)

    for c in range(NUM_CLASSES):
        per_class[LABELS[c]] = {
            "f1": round(float(f1[c]), 4),
            "precision": round(float(p[c]), 4),
            "recall": round(float(r[c]), 4),
            "support": int(support[c])
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "per_class": per_class
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    print("=== 7-Parent GalleryFL Evaluation (Single-Label, argmax) ===")

    w1, b1, w2, b2 = load_checkpoint(args.checkpoint)

    items = load_test_data(args.test_manifest)
    print(f"[eval] Loaded {len(items)} samples from manifest")

    # Extract features using train's extractor (keeps backbone consistent)
    from train import load_backbone, extract_features
    interp, inp, out = load_backbone()
    paths = [p for p, _ in items]
    X = extract_features(interp, inp, out, paths)

    labels = np.array([lbl for _, lbl in items])

    probs = forward_softmax(X, w1, b1, w2, b2)
    preds = np.argmax(probs, axis=1)

    report = compute_metrics(labels, preds)

    print(f"\nMacro F1: {report['macro_f1']:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("Per-class:")
    for name, m in report["per_class"].items():
        print(f"  {name:12s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  support={m['support']}")

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {METRICS_PATH}")

    # Always write sensible default thresholds for 7-class (argmax primary)
    default_thresh = {lbl: 0.5 for lbl in LABELS}
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(default_thresh, f, indent=2)
    print(f"Thresholds written to {THRESHOLDS_PATH}")

if __name__ == "__main__":
    main()