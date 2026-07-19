#!/usr/bin/env python3
"""
train.py - Clean 7-parent GalleryFL head training

Trains a small classification head on top of the frozen base_model.tflite
(1024-d projection) for the 7 core GalleryFL categories.

Architecture:
    1024 -> Dense(256, relu) -> Dense(7, sigmoid)

Key fixes vs previous broken version:
- Proper feature standardization (mean/std from train set)
- Correct BCE gradient / loss
- Class-weighted training
- Macro-F1 driven checkpointing with per-class threshold tuning
- Clean TF/Keras head training for numerical stability
- Proper one-hot labels (from converter or manifests)
- Best model selected on *tuned* macro F1, not fixed 0.5
- Saves thresholds.json together with checkpoint

Usage:
    export DATASET_DIR=/path/to/gallery_7tag
    python train.py --dataset-dir $DATASET_DIR --epochs 25
"""

import argparse
import json
import os
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np

# TensorFlow for stable head training (TFLite only for feature extraction)
import tensorflow as tf

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RetrainConfig, get_config, LABELS, NUM_CLASSES,
    LABEL_TO_IDX, BACKBONE_PATH, FEATURE_DIM,
    BEST_CHECKPOINT, METRICS_REPORT, HEAD_WEIGHTS_PATH,
    MODEL_VERSION_FILE, RETRAIN_OUTPUT_DIR
)

# ------------------------------------------------------------------
# Feature extraction (frozen TFLite backbone - unchanged contract)
# ------------------------------------------------------------------

def load_backbone():
    if not os.path.exists(BACKBONE_PATH):
        raise FileNotFoundError(
            f"Backbone not found: {BACKBONE_PATH}. "
            "Place models/base_model.tflite in server/models/"
        )
    interpreter = tf.lite.Interpreter(model_path=BACKBONE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Find 1024-d projection
    proj = None
    for o in output_details:
        if o["shape"][-1] == FEATURE_DIM:
            proj = o
            break
    if proj is None:
        proj = output_details[0]
    return interpreter, input_details, proj


def preprocess_image(path: str) -> np.ndarray:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224], method="bilinear")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(img, 0)


def extract_features(
    interpreter,
    input_details,
    proj,
    image_paths: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """Extract 1024-d features from frozen backbone."""
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            if os.path.exists(p):
                try:
                    img = preprocess_image(p)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"  [warn] failed to load {p}: {e}")
                    batch_imgs.append(None)
            else:
                batch_imgs.append(None)

        # Run inference for valid images
        for j, img in enumerate(batch_imgs):
            if img is None:
                features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                continue
            interpreter.set_tensor(input_details["index"], img.numpy())
            interpreter.invoke()
            feat = interpreter.get_tensor(proj["index"])[0]
            features.append(feat.astype(np.float32))

        if (i // batch_size) % 10 == 0:
            print(f"  extracted {len(features)} / {len(image_paths)}")

    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------------
# Data loading (manifests preferred, folder fallback)
# ------------------------------------------------------------------

def load_manifest(manifest_path: str) -> List[Tuple[str, int]]:
    items = []
    if not os.path.exists(manifest_path):
        return items
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"]
            label = row["label"].strip().lower()
            if label in LABEL_TO_IDX:
                items.append((img_path, LABEL_TO_IDX[label]))
    return items


def load_dataset(dataset_dir: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Load train/val. Prefers manifests/, falls back to folder scan."""
    manifests_dir = os.path.join(dataset_dir, "manifests")
    train_path = os.path.join(manifests_dir, "train.csv")
    val_path = os.path.join(manifests_dir, "val.csv")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("[data] Using manifests from", manifests_dir)
        train = load_manifest(train_path)
        val = load_manifest(val_path)
    else:
        print("[data] No manifests found — scanning folders (slower)")
        from prepare_dataset import collect_dataset, stratified_split
        raw = collect_dataset(dataset_dir)
        train_d, val_d, _ = stratified_split(raw)
        train = [(p, LABEL_TO_IDX[l]) for l, paths in train_d.items() for p in paths]
        val = [(p, LABEL_TO_IDX[l]) for l, paths in val_d.items() for p in paths]

    # Validate
    train_counts = defaultdict(int)
    for _, lbl in train:
        train_counts[lbl] += 1

    print(f"[data] Train samples: {len(train)}")
    print(f"[data] Val samples:   {len(val)}")
    for i, name in enumerate(LABELS):
        cnt = train_counts.get(i, 0)
        print(f"  {name:12s}: {cnt}")

    if any(train_counts.get(i, 0) == 0 for i in range(NUM_CLASSES)):
        missing = [LABELS[i] for i in range(NUM_CLASSES) if train_counts.get(i, 0) == 0]
        raise ValueError(f"CRITICAL: These classes have ZERO training samples: {missing}")

    if len(val) < 20:
        print("[warn] Very small validation set — metrics may be noisy")

    return train, val


def to_one_hot(labels: List[int]) -> np.ndarray:
    """Convert list of class indices to one-hot (multi-class style)."""
    y = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    for i, lbl in enumerate(labels):
        y[i, lbl] = 1.0
    return y


# ------------------------------------------------------------------
# Threshold tuning (critical for macro F1)
# ------------------------------------------------------------------

def tune_thresholds(probs: np.ndarray, y_true: np.ndarray, num_steps: int = 21) -> np.ndarray:
    """Search per-class thresholds to maximize macro F1 on validation."""
    thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    for c in range(NUM_CLASSES):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, num_steps):
            yp = (probs[:, c] > t).astype(np.float32)
            yt = y_true[:, c]
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[c] = best_t
    return thresholds


def compute_macro_f1(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> float:
    """Compute macro F1 using per-class thresholds."""
    y_pred = np.zeros_like(probs, dtype=np.float32)
    for c in range(NUM_CLASSES):
        y_pred[:, c] = (probs[:, c] > thresholds[c]).astype(np.float32)

    f1s = []
    for c in range(NUM_CLASSES):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


# ------------------------------------------------------------------
# Main training
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train 7-parent GalleryFL head")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = get_config({
        "dataset_dir": args.dataset_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    })

    print("=" * 60)
    print("7-PARENT GALLERYFL HEAD TRAINING (CLEAN)")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset_dir}")
    print(f"Labels : {LABELS}")
    print(f"Epochs : {cfg.epochs}")
    print(f"LR     : {cfg.lr}")

    # 1. Load data
    train_items, val_items = load_dataset(cfg.dataset_dir)

    # 2. Extract features
    print("\n[1/4] Extracting features from frozen backbone...")
    interpreter, input_details, proj = load_backbone()

    train_paths, train_labels = zip(*train_items)
    val_paths, val_labels = zip(*val_items)

    X_train = extract_features(interpreter, input_details, proj, list(train_paths))
    X_val = extract_features(interpreter, input_details, proj, list(val_paths))

    y_train = to_one_hot(list(train_labels))
    y_val = to_one_hot(list(val_labels))

    # 3. Feature standardization (VERY IMPORTANT)
    print("\n[2/4] Standardizing features...")
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Save normalization stats for inference compatibility later
    norm_stats = {"mean": mean.tolist()[0], "std": std.tolist()[0]}
    with open(os.path.join(RETRAIN_OUTPUT_DIR, "feature_norm.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    # 4. Build small Keras head (stable) - keep sigmoid for GalleryFL compatibility
    print("\n[3/4] Building and training head (sigmoid + binary/focal)...")

    inputs = tf.keras.Input(shape=(FEATURE_DIM,))
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    # Class weights (inverse frequency)
    class_counts = np.sum(y_train, axis=0)
    total = len(y_train)
    class_weights = (total / (NUM_CLASSES * np.maximum(class_counts, 1.0))).astype(np.float32)
    print("Class weights:", dict(zip(LABELS, np.round(class_weights, 3))))

    # Focal loss (binary) - good for imbalance on rare classes like documents/events
    def binary_focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            pt = tf.exp(-bce)
            focal = alpha * tf.pow(1.0 - pt, gamma) * bce
            return tf.reduce_mean(focal)
        return loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr, weight_decay=1e-4),
        loss=binary_focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )

    # Custom callback for macro-F1 + threshold tuning
    best_f1 = -1.0
    best_weights = None
    best_thresholds = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # Train one epoch
        model.fit(
            X_train, y_train,
            batch_size=cfg.batch_size,
            epochs=1,
            verbose=0,
            class_weight={i: float(w) for i, w in enumerate(class_weights)}
        )

        # Predict on val
        val_probs = model.predict(X_val, batch_size=64, verbose=0)

        # Tune thresholds on val for this epoch
        current_thresholds = tune_thresholds(val_probs, y_val)
        epoch_f1 = compute_macro_f1(y_val, val_probs, current_thresholds)

        # Also compute with fixed 0.5 for comparison
        fixed_f1 = compute_macro_f1(y_val, val_probs, np.full(NUM_CLASSES, 0.5))

        history.append({
            "epoch": epoch,
            "tuned_macro_f1": round(epoch_f1, 4),
            "fixed_0.5_macro_f1": round(fixed_f1, 4),
        })

        print(f"Epoch {epoch:2d}/{cfg.epochs} | "
              f"tuned macroF1={epoch_f1:.4f} | fixed0.5={fixed_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_weights = model.get_weights()
            best_thresholds = current_thresholds.copy()
            print(f"  *** New best tuned macro F1: {best_f1:.4f} ***")

    print(f"\n[4/4] Best tuned macro F1 on val: {best_f1:.4f}")

    # Restore best weights
    model.set_weights(best_weights)

    # 5. Save artifacts
    os.makedirs(RETRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HEAD_WEIGHTS_PATH), exist_ok=True)

    # Save Keras weights temporarily, then convert to npz format expected by GalleryFL
    w1, b1 = model.layers[1].get_weights()   # Dense 256
    w2, b2 = model.layers[3].get_weights()   # Dense 7

    # Save in GalleryFL expected format (w1, b1, w2, b2)
    np.savez(
        BEST_CHECKPOINT,
        w1=w1.astype(np.float32),
        b1=b1.astype(np.float32),
        w2=w2.astype(np.float32),
        b2=b2.astype(np.float32)
    )
    np.savez(
        HEAD_WEIGHTS_PATH,
        w1=w1.astype(np.float32),
        b1=b1.astype(np.float32),
        w2=w2.astype(np.float32),
        b2=b2.astype(np.float32)
    )

    # Save thresholds
    thresholds_path = os.path.join(RETRAIN_OUTPUT_DIR, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump({
            "thresholds": best_thresholds.tolist(),
            "labels": LABELS,
            "best_macro_f1": float(best_f1)
        }, f, indent=2)

    # Save metrics
    metrics = {
        "best_macro_f1": round(best_f1, 4),
        "num_classes": NUM_CLASSES,
        "labels": LABELS,
        "thresholds": best_thresholds.tolist(),
        "history": history[-10:],  # last 10 epochs
        "feature_dim": FEATURE_DIM,
        "head_architecture": "1024->256->7 (sigmoid)",
    }
    with open(METRICS_REPORT, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    hist_path = os.path.join(RETRAIN_OUTPUT_DIR, "training_history.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tuned_macro_f1", "fixed_0.5_macro_f1"])
        for h in history:
            writer.writerow([h["epoch"], h["tuned_macro_f1"], h["fixed_0.5_macro_f1"]])

    # Bump version
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write("7")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best checkpoint : {BEST_CHECKPOINT}")
    print(f"Production head : {HEAD_WEIGHTS_PATH}")
    print(f"Thresholds      : {thresholds_path}")
    print(f"Metrics         : {METRICS_REPORT}")
    print(f"History         : {hist_path}")
    print(f"Model version   : 7")
    print("\nNext step: python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds")


if __name__ == "__main__":
    main()