#!/usr/bin/env python3
"""
train.py - Proper 7-parent GalleryFL head training (single-label focus)

Trains a small classification head on top of the frozen base_model.tflite
for the 7 core GalleryFL parent categories.

This version is optimized for single-label classification (one photo = one primary parent)
while keeping the output format compatible with GalleryFL (4-layer npz).

Key improvements for higher macro F1:
- Softmax + SparseCategoricalCrossentropy (much better for 7 mutually exclusive parents)
- Label smoothing
- Strong class weighting + focal-like behavior via class weights
- Proper integer labels
- Macro F1 computed with argmax (top-1) for single-label
- Early stopping on val macro F1
- Feature standardization
- Better regularization
"""

import argparse
import json
import os
import csv
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    get_config, LABELS, NUM_CLASSES, LABEL_TO_IDX,
    BACKBONE_PATH, FEATURE_DIM,
    BEST_CHECKPOINT, METRICS_REPORT, HEAD_WEIGHTS_PATH,
    MODEL_VERSION_FILE, RETRAIN_OUTPUT_DIR
)

# ------------------------------------------------------------------
# Feature extraction (frozen TFLite)
# ------------------------------------------------------------------

def load_backbone():
    if not os.path.exists(BACKBONE_PATH):
        raise FileNotFoundError(f"Backbone not found: {BACKBONE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=BACKBONE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    proj = next((o for o in output_details if o["shape"][-1] == FEATURE_DIM), output_details[0])
    return interpreter, input_details, proj


def preprocess_image(path: str):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224], method="bilinear")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(img, 0)


def extract_features(interpreter, input_details, proj, image_paths, batch_size=32):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for p in batch:
            if not os.path.exists(p):
                features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                continue
            try:
                img = preprocess_image(p)
                interpreter.set_tensor(input_details["index"], img.numpy())
                interpreter.invoke()
                feat = interpreter.get_tensor(proj["index"])[0]
                features.append(feat.astype(np.float32))
            except Exception:
                features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_manifest(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lbl = row["label"].strip().lower()
            if lbl in LABEL_TO_IDX:
                items.append((row["image_path"], LABEL_TO_IDX[lbl]))
    return items


def load_dataset(dataset_dir):
    mdir = os.path.join(dataset_dir, "manifests")
    train_p = os.path.join(mdir, "train.csv")
    val_p = os.path.join(mdir, "val.csv")

    if os.path.exists(train_p) and os.path.exists(val_p):
        train = load_manifest(train_p)
        val = load_manifest(val_p)
    else:
        from prepare_dataset import collect_dataset, stratified_split
        raw = collect_dataset(dataset_dir)
        tr, va, _ = stratified_split(raw)
        train = [(p, LABEL_TO_IDX[l]) for l, ps in tr.items() for p in ps]
        val = [(p, LABEL_TO_IDX[l]) for l, ps in va.items() for p in ps]

    train_counts = defaultdict(int)
    for _, l in train:
        train_counts[l] += 1

    print(f"[data] Train: {len(train)}  Val: {len(val)}")
    for i, name in enumerate(LABELS):
        print(f"  {name:12s}: {train_counts.get(i, 0)}")

    if any(train_counts.get(i, 0) == 0 for i in range(NUM_CLASSES)):
        missing = [LABELS[i] for i in range(NUM_CLASSES) if train_counts.get(i, 0) == 0]
        raise ValueError(f"CRITICAL: Zero training samples for: {missing}")

    return train, val


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00015)
    args = parser.parse_args()

    cfg = get_config({
        "dataset_dir": args.dataset_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    })

    print("=" * 60)
    print("7-PARENT GALLERYFL HEAD TRAINING (SINGLE-LABEL OPTIMIZED)")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset_dir}")
    print(f"Labels:  {LABELS}")

    train_items, val_items = load_dataset(cfg.dataset_dir)

    print("\n[1/4] Extracting features...")
    interp, inp, proj = load_backbone()

    train_paths, train_lbls = zip(*train_items)
    val_paths, val_lbls = zip(*val_items)

    X_train = extract_features(interp, inp, proj, list(train_paths))
    X_val = extract_features(interp, inp, proj, list(val_paths))

    y_train = np.array(train_lbls, dtype=np.int32)
    y_val = np.array(val_lbls, dtype=np.int32)

    print("\n[2/4] Standardizing features...")
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    os.makedirs(RETRAIN_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(RETRAIN_OUTPUT_DIR, "feature_norm.json"), "w") as f:
        json.dump({"mean": mean[0].tolist(), "std": std[0].tolist()}, f, indent=2)

    print("\n[3/4] Building head (softmax + categorical + label smoothing)...")

    inputs = tf.keras.Input(shape=(FEATURE_DIM,))
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Class weights
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    total = len(y_train)
    class_weights = total / (NUM_CLASSES * np.maximum(class_counts, 1.0))
    print("Class weights:", dict(zip(LABELS, np.round(class_weights, 3))))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    best_f1 = -1.0
    best_weights = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.fit(
            X_train, y_train,
            batch_size=cfg.batch_size,
            epochs=1,
            verbose=0,
            class_weight={i: float(w) for i, w in enumerate(class_weights)}
        )

        # Validation with argmax (proper for single-label)
        val_probs = model.predict(X_val, batch_size=64, verbose=0)
        y_pred = np.argmax(val_probs, axis=1)

        # Per-class F1 using top-1 prediction (treated as one-hot for macro F1)
        f1s = []
        for c in range(NUM_CLASSES):
            yt = (y_val == c).astype(np.float32)
            yp = (y_pred == c).astype(np.float32)
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))

        epoch_f1 = float(np.mean(f1s))
        acc = float(np.mean(y_val == y_pred))

        history.append({"epoch": epoch, "macro_f1": round(epoch_f1, 4), "acc": round(acc, 4)})
        print(f"Epoch {epoch:2d}/{cfg.epochs} | macroF1={epoch_f1:.4f} | acc={acc:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_weights = model.get_weights()
            print(f"  *** New best macro F1: {best_f1:.4f} ***")

    print(f"\n[4/4] Best macro F1 on val: {best_f1:.4f}")
    model.set_weights(best_weights)

    # Save in GalleryFL 4-layer format
    w1, b1 = model.layers[1].get_weights()
    w2, b2 = model.layers[-1].get_weights()

    np.savez(BEST_CHECKPOINT, w1=w1.astype(np.float32), b1=b1.astype(np.float32),
             w2=w2.astype(np.float32), b2=b2.astype(np.float32))
    np.savez(HEAD_WEIGHTS_PATH, w1=w1.astype(np.float32), b1=b1.astype(np.float32),
             w2=w2.astype(np.float32), b2=b2.astype(np.float32))

    with open(MODEL_VERSION_FILE, "w") as f:
        f.write("7")

    metrics = {
        "best_macro_f1": round(best_f1, 4),
        "labels": LABELS,
        "history": history[-15:],
        "head_architecture": "1024->256->7 (softmax)"
    }
    with open(METRICS_REPORT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Best macro F1: {best_f1:.4f}")
    print(f"Production head: {HEAD_WEIGHTS_PATH}")
    print("Next: python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --test-manifest ... --tune-thresholds")


if __name__ == "__main__":
    main()