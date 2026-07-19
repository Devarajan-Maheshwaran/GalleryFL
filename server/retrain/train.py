#!/usr/bin/env python3
"""
train.py
Clean 7-parent GalleryFL head training pipeline.

- Uses frozen base_model.tflite (1024-d projection) for bit-compatibility.
- Trains tiny head on top: 1024 -> 256 ReLU -> 7 sigmoid.
- Primary metric: macro F1 (best checkpoint selection).
- Class weighting + optional per-class threshold tuning.
- Proper train/val/test from manifests or folder structure.

Usage:
    export DATASET_DIR=/path/to/7tag/data
    python train.py --dataset-dir $DATASET_DIR --epochs 30

Outputs:
    server/output/retrain/best_checkpoint.npz
    server/output/retrain/retrain_metrics.json
    (also copies best to models/head_weights.npz via export later)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # for config
from retrain.config import (
    RetrainConfig, get_config, LABELS, LABEL_TO_IDX, NUM_CLASSES,
    BACKBONE_PATH, FEATURE_DIM, IMAGE_SIZE,
    BEST_CHECKPOINT, METRICS_REPORT, HEAD_WEIGHTS_PATH, MODEL_VERSION_FILE
)

# Try to import TF only when needed for feature extraction
def load_backbone():
    import tensorflow as tf
    if not os.path.exists(BACKBONE_PATH):
        raise FileNotFoundError(f"Backbone not found: {BACKBONE_PATH}. "
                                "Place models/base_model.tflite in server/models/")
    interp = tf.lite.Interpreter(model_path=BACKBONE_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    # Find the 1024-d projection output
    out_details = interp.get_output_details()
    proj = None
    for o in out_details:
        if o["shape"][-1] == FEATURE_DIM:
            proj = o
            break
    if proj is None:
        proj = out_details[0]
    return interp, inp, proj

def preprocess_image(path: str):
    import tensorflow as tf
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(img, 0)

def extract_feature(interp, inp, proj, path: str) -> np.ndarray:
    import tensorflow as tf
    img = preprocess_image(path)
    interp.set_tensor(inp["index"], img.numpy())
    interp.invoke()
    feat = interp.get_tensor(proj["index"])[0]
    return feat.astype(np.float32)

def load_manifest(manifest_path: str) -> List[Tuple[str, int]]:
    """Return list of (image_path, label_idx)"""
    items = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image_path"]
            label = row["label"].strip().lower()
            if label in LABEL_TO_IDX:
                items.append((img, LABEL_TO_IDX[label]))
    return items

def load_dataset_from_manifests(dataset_dir: str, manifests_dir: str = None):
    """Prefer manifests if present, otherwise scan folders."""
    if manifests_dir is None:
        manifests_dir = os.path.join(dataset_dir, "manifests")

    train_path = os.path.join(manifests_dir, "train.csv")
    val_path = os.path.join(manifests_dir, "val.csv")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("[data] Using manifests from", manifests_dir)
        train = load_manifest(train_path)
        val = load_manifest(val_path)
        return train, val

    # Fallback: scan folders (simple)
    print("[data] No manifests — scanning folders directly")
    from prepare_dataset import collect_dataset, stratified_split
    raw = collect_dataset(dataset_dir)
    train_d, val_d, _ = stratified_split(raw)
    train = [(p, LABEL_TO_IDX[l]) for l, paths in train_d.items() for p in paths]
    val = [(p, LABEL_TO_IDX[l]) for l, paths in val_d.items() for p in paths]
    return train, val

def compute_class_weights(labels: List[int]) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * np.maximum(counts, 1))
    return weights.astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def relu(x):
    return np.maximum(0.0, x)

def forward(X, w1, b1, w2, b2):
    return sigmoid(relu(X @ w1 + b1) @ w2 + b2)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in range(NUM_CLASSES):
        tp = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1))
        fp = np.sum((y_true[:, c] == 0) & (y_pred[:, c] == 1))
        fn = np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def train_head(Xtr, Ytr, Xval, Yval, cfg: RetrainConfig):
    """Simple numpy Adam + macro-F1 checkpointing."""
    rng = np.random.default_rng(42)
    dim = Xtr.shape[1]
    hidden = cfg.hidden

    # Glorot init
    def glorot(shape):
        return (rng.standard_normal(shape).astype(np.float32) *
                np.sqrt(6.0 / (shape[0] + shape[1])))

    w1 = glorot((dim, hidden))
    b1 = np.zeros(hidden, np.float32)
    w2 = glorot((hidden, NUM_CLASSES))
    b2 = np.zeros(NUM_CLASSES, np.float32)

    # Adam state
    m = {k: np.zeros_like(v) for k, v in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)]}
    v = {k: np.zeros_like(v) for k, v in m.items()}
    t = 0
    lr = cfg.lr
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    class_weights = compute_class_weights(Ytr.argmax(1) if Ytr.ndim > 1 else Ytr) if cfg.use_class_weights else np.ones(NUM_CLASSES, np.float32)

    best_f1 = -1.0
    best_state = None
    history = []

    n = Xtr.shape[0]
    batch = cfg.batch_size

    print(f"[train] samples={n}  val={Xval.shape[0]}  classes={NUM_CLASSES}  lr={lr}")

    for epoch in range(1, cfg.epochs + 1):
        # Shuffle
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb = Xtr[idx]
            yb = Ytr[idx]

            # Forward
            z1 = xb @ w1 + b1
            a1 = relu(z1)
            logits = a1 @ w2 + b2
            probs = sigmoid(logits)

            # Weighted BCE grad (simplified focal-like)
            eps_ = 1e-7
            pc = np.clip(probs, eps_, 1 - eps_)
            loss_grad = (yb - pc) * class_weights   # simple weighted

            # Backprop
            dlogits = loss_grad
            dw2 = a1.T @ dlogits
            db2 = dlogits.sum(0)
            da1 = dlogits @ w2.T
            dz1 = da1 * (z1 > 0)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(0)

            # Adam update
            t += 1
            for name, g in [("w1", dw1), ("b1", db1), ("w2", dw2), ("b2", db2)]:
                m[name] = beta1 * m[name] + (1 - beta1) * g
                v[name] = beta2 * v[name] + (1 - beta2) * (g * g)
                mhat = m[name] / (1 - beta1 ** t)
                vhat = v[name] / (1 - beta2 ** t)
                if name == "w1": w1 -= lr * mhat / (np.sqrt(vhat) + eps)
                elif name == "b1": b1 -= lr * mhat / (np.sqrt(vhat) + eps)
                elif name == "w2": w2 -= lr * mhat / (np.sqrt(vhat) + eps)
                elif name == "b2": b2 -= lr * mhat / (np.sqrt(vhat) + eps)

        # Eval on val
        Pval = forward(Xval, w1, b1, w2, b2)
        yhat = (Pval > cfg.default_threshold).astype(np.float32)
        f1 = macro_f1(Yval, yhat)
        history.append({"epoch": epoch, "macro_f1": round(f1, 4)})

        if f1 > best_f1:
            best_f1 = f1
            best_state = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
            print(f"  epoch {epoch:3d}: val macroF1={f1:.4f}  *** BEST ***")
        elif epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: val macroF1={f1:.4f}")

    print(f"[train] Best val macro F1: {best_f1:.4f}")
    return best_state, history, best_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--manifests", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config({
        "dataset_dir": args.dataset_dir or None,
        "epochs": args.epochs or None,
        "lr": args.lr or None,
        "batch_size": args.batch_size or None,
    })

    print("=== 7-Tag GalleryFL Retraining (TRAIN) ===")
    print(f"Dataset: {cfg.dataset_dir}")
    print(f"Labels: {LABELS}")

    # Load data
    train_items, val_items = load_dataset_from_manifests(cfg.dataset_dir, args.manifests)

    if len(train_items) < 10 or len(val_items) < 5:
        print("ERROR: Not enough data. Need at least a few dozen images total.")
        sys.exit(1)

    # Extract features (or load cached)
    print("[features] Extracting from frozen backbone...")
    interp, inp, proj = load_backbone()

    def extract_batch(items):
        X, Y = [], []
        for path, lbl in items:
            if not os.path.exists(path):
                print(f"  [warn] missing {path}")
                continue
            try:
                feat = extract_feature(interp, inp, proj, path)
                X.append(feat)
                y = np.zeros(NUM_CLASSES, np.float32)
                y[lbl] = 1.0
                Y.append(y)
            except Exception as e:
                print(f"  [warn] failed {os.path.basename(path)}: {e}")
        return np.array(X), np.array(Y)

    Xtr, Ytr = extract_batch(train_items)
    Xval, Yval = extract_batch(val_items)

    print(f"[data] train={Xtr.shape}  val={Xval.shape}")

    best_w, history, best_f1 = train_head(Xtr, Ytr, Xval, Yval, cfg)

    # Save best checkpoint
    os.makedirs(cfg.output_dir, exist_ok=True)
    w1, b1, w2, b2 = best_w
    np.savez(BEST_CHECKPOINT, w1=w1, b1=b1, w2=w2, b2=b2)

    # Also save a copy as active head (will be finalized by export)
    np.savez(HEAD_WEIGHTS_PATH, w1=w1, b1=b1, w2=w2, b2=b2)

    # Write version
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write("7")

    # Metrics
    metrics = {
        "best_macro_f1": round(best_f1, 4),
        "num_classes": NUM_CLASSES,
        "labels": LABELS,
        "history": history[-10:],   # last 10
        "best_epoch": len(history),
        "checkpoint": BEST_CHECKPOINT,
    }
    with open(METRICS_REPORT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[done] Best checkpoint saved to {BEST_CHECKPOINT}")
    print(f"[done] head_weights.npz updated (7-class)")
    print(f"[done] Metrics: {METRICS_REPORT}")

if __name__ == "__main__":
    main()