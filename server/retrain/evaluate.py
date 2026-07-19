#!/usr/bin/env python3
"""
evaluate.py
Full evaluation for the 7-parent GalleryFL classifier.

Computes:
- macro F1 (primary)
- per-class F1, precision, recall, support
- accuracy
- simple confusion summary (top confusions)

Usage:
    python evaluate.py --checkpoint server/output/retrain/best_checkpoint.npz
    python evaluate.py --checkpoint server/output/retrain/best_checkpoint.npz --test-manifest manifests/test.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LABELS, NUM_CLASSES, LABEL_TO_IDX,
    BEST_CHECKPOINT, METRICS_REPORT
)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ w2 + b2
    return sigmoid(z2)

def load_checkpoint(path: str):
    d = np.load(path)
    return d["w1"], d["b1"], d["w2"], d["b2"]

def load_test_data(manifest: str = None, dataset_dir: str = None):
    """Load test set. Prefer manifest, else scan folders (last 15%)."""
    items = []
    if manifest and os.path.exists(manifest):
        with open(manifest) as f:
            for row in csv.DictReader(f):
                lbl = row["label"].strip().lower()
                if lbl in LABEL_TO_IDX:
                    items.append((row["image_path"], LABEL_TO_IDX[lbl]))
        print(f"[eval] Loaded {len(items)} samples from manifest")
        return items

    # Fallback scan (very simple)
    print("[eval] No test manifest — using folder scan (best-effort)")
    from prepare_dataset import collect_dataset
    raw = collect_dataset(dataset_dir or ".")
    # crude: take ~15% per class as "test"
    test_items = []
    for lbl, paths in raw.items():
        n = max(1, int(len(paths) * 0.15))
        test_items.extend([(p, LABEL_TO_IDX[lbl]) for p in paths[-n:]])
    return test_items

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray = None):
    if thresholds is None:
        thresholds = np.full(NUM_CLASSES, 0.5)

    y_pred = (probs > thresholds).astype(np.float32)

    per_class = {}
    f1_list = []
    for c in range(NUM_CLASSES):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        support = int(yt.sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[LABELS[c]] = {
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "support": support,
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    accuracy = float(np.mean((y_true == y_pred).all(axis=1)))

    # Simple confusion summary (most confused pairs)
    confusions = []
    for c in range(NUM_CLASSES):
        for c2 in range(NUM_CLASSES):
            if c == c2: continue
            false_as = int(np.sum((y_true[:, c] == 1) & (y_pred[:, c2] == 1)))
            if false_as > 0:
                confusions.append((LABELS[c], LABELS[c2], false_as))

    confusions.sort(key=lambda x: -x[2])
    top_confusions = confusions[:5]

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "top_confusions": top_confusions,
        "num_samples": int(y_true.shape[0]),
    }

def tune_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Simple per-class threshold search for best F1."""
    thresholds = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            yp = (probs[:, c] > t).astype(np.float32)
            yt = y_true[:, c]
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--tune-thresholds", action="store_true")
    args = parser.parse_args()

    print("=== 7-Tag GalleryFL Evaluation ===")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    w1, b1, w2, b2 = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    items = load_test_data(args.test_manifest, args.dataset_dir)
    if not items:
        print("No test data found.")
        sys.exit(1)

    # Extract features for eval (reuse train code logic)
    from train import load_backbone, extract_feature
    interp, inp, proj = load_backbone()

    X, Y = [], []
    for path, lbl in items:
        if not os.path.exists(path): continue
        try:
            feat = extract_feature(interp, inp, proj, path)
            X.append(feat)
            y = np.zeros(NUM_CLASSES, np.float32); y[lbl] = 1.0
            Y.append(y)
        except Exception as e:
            print(f"  [skip] {path}: {e}")

    X = np.array(X)
    Y = np.array(Y)
    print(f"Eval samples: {X.shape[0]}")

    probs = forward(X, w1, b1, w2, b2)

    if args.tune_thresholds:
        thresholds = tune_thresholds(probs, Y)
        print("Tuned thresholds:", dict(zip(LABELS, np.round(thresholds, 3))))
    else:
        thresholds = None

    report = compute_metrics(Y, probs, thresholds)

    print(f"\n=== RESULTS ===")
    print(f"Macro F1: {report['macro_f1']:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("Per-class:")
    for name, m in report["per_class"].items():
        print(f"  {name:12s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  support={m['support']}")

    if report["top_confusions"]:
        print("Top confusions (true -> predicted):")
        for a, b, cnt in report["top_confusions"]:
            print(f"  {a} -> {b}: {cnt}")

    os.makedirs(os.path.dirname(METRICS_REPORT), exist_ok=True)
    with open(METRICS_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {METRICS_REPORT}")

if __name__ == "__main__":
    main()