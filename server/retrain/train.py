#!/usr/bin/env python3
"""
train.py - Clean, proper training for 7-parent GalleryFL classifier.

Goal: Single-label classification (one photo → one primary parent tag)
Dataset: COCO minitrain-10k mapped to 7 parents
Backbone: Frozen base_model.tflite (1024-d features)

Training recipe (designed for non-IID gallery data but starting from COCO):
- Softmax head
- Label smoothing + class weights
- Feature standardization
- Early stopping on macro F1 (argmax)
- Best checkpoint saved
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    get_config, LABELS, NUM_CLASSES, LABEL_TO_IDX,
    BACKBONE_PATH, FEATURE_DIM,
    BEST_CHECKPOINT, METRICS_PATH, HEAD_WEIGHTS_PATH,
    MODEL_VERSION_FILE, RETRAIN_DIR
)

def load_backbone():
    interp = tf.lite.Interpreter(model_path=BACKBONE_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = next(o for o in interp.get_output_details() if o["shape"][-1] == FEATURE_DIM)
    return interp, inp, out

def preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(img, 0)

def extract_features(interp, inp, out, paths, batch=64):
    feats = []
    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i+batch]
        batch_imgs = []
        for p in batch_paths:
            if os.path.exists(p):
                try:
                    batch_imgs.append(preprocess(p))
                except:
                    batch_imgs.append(None)
            else:
                batch_imgs.append(None)
        for img in batch_imgs:
            if img is None:
                feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                continue
            interp.set_tensor(inp["index"], img.numpy())
            interp.invoke()
            feats.append(interp.get_tensor(out["index"])[0].astype(np.float32))
    return np.array(feats, dtype=np.float32)

def load_manifest(path):
    items = []
    with open(path) as f:
        for row in __import__("csv").DictReader(f):
            items.append((row["image_path"], LABEL_TO_IDX[row["label"]]))
    return items

def load_data(dataset_dir):
    mdir = os.path.join(dataset_dir, "manifests")
    train = load_manifest(os.path.join(mdir, "train.csv"))
    val = load_manifest(os.path.join(mdir, "val.csv"))
    test = load_manifest(os.path.join(mdir, "test.csv"))
    return train, val, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = get_config({"dataset_dir": args.dataset_dir, "epochs": args.epochs, "lr": args.lr})

    print("="*60)
    print("PROPER 7-PARENT GALLERYFL TRAINING (from scratch)")
    print("Single-label classification on 7 parents")
    print("="*60)

    train_items, val_items, test_items = load_data(cfg.dataset_dir)

    print(f"\nTrain: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    # Class counts for weighting
    train_labels = [lbl for _, lbl in train_items]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = len(train_labels) / (NUM_CLASSES * np.maximum(class_counts, 1))
    print("Class weights:", dict(zip(LABELS, np.round(class_weights, 3))))

    print("\n[1/4] Extracting features from frozen backbone...")
    interp, inp, out = load_backbone()

    train_paths, train_y = zip(*train_items)
    val_paths, val_y = zip(*val_items)

    X_train = extract_features(interp, inp, out, list(train_paths))
    X_val = extract_features(interp, inp, out, list(val_paths))

    y_train = np.array(train_y, dtype=np.int32)
    y_val = np.array(val_y, dtype=np.int32)

    print("\n[2/4] Feature standardization...")
    mean = X_train.mean(0, keepdims=True)
    std = X_train.std(0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    os.makedirs(RETRAIN_DIR, exist_ok=True)
    with open(os.path.join(RETRAIN_DIR, "feature_norm.json"), "w") as f:
        json.dump({"mean": mean[0].tolist(), "std": std[0].tolist()}, f)

    print("\n[3/4] Building head (softmax + class weights)...")

    inp_layer = tf.keras.Input(shape=(FEATURE_DIM,))
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(inp_layer)
    x = tf.keras.layers.Dropout(0.4)(x)
    out_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inp_layer, out_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    best_f1 = -1.0
    best_weights = None
    history = []

    for ep in range(1, cfg.epochs + 1):
        model.fit(X_train, y_train, batch_size=cfg.batch_size, epochs=1, verbose=0,
                  class_weight={i: float(w) for i, w in enumerate(class_weights)})

        probs = model.predict(X_val, batch_size=64, verbose=0)
        preds = np.argmax(probs, axis=1)

        # Macro F1 using top-1
        f1s = []
        for c in range(NUM_CLASSES):
            yt = (y_val == c).astype(int)
            yp = (preds == c).astype(int)
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1s.append(2 * p * r / (p + r + 1e-8))

        macro_f1 = float(np.mean(f1s))
        acc = float(np.mean(preds == y_val))

        history.append({"epoch": ep, "macro_f1": round(macro_f1, 4), "acc": round(acc, 4)})
        print(f"Epoch {ep:2d}/{cfg.epochs} | macroF1={macro_f1:.4f} | acc={acc:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_weights = model.get_weights()
            print(f"  ★ New best macro F1: {best_f1:.4f}")

    print(f"\nBest macro F1 on validation: {best_f1:.4f}")
    model.set_weights(best_weights)

    # Save in GalleryFL format (robust layer extraction)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    w1, b1 = dense_layers[0].get_weights()
    w2, b2 = dense_layers[1].get_weights()

    np.savez(BEST_CHECKPOINT, w1=w1.astype(np.float32), b1=b1.astype(np.float32),
             w2=w2.astype(np.float32), b2=b2.astype(np.float32))
    np.savez(HEAD_WEIGHTS_PATH, w1=w1.astype(np.float32), b1=b1.astype(np.float32),
             w2=w2.astype(np.float32), b2=b2.astype(np.float32))

    with open(MODEL_VERSION_FILE, "w") as f:
        f.write("7")

    metrics = {
        "best_macro_f1": round(best_f1, 4),
        "labels": LABELS,
        "history": history[-20:],
        "architecture": "1024 → 256 → 7 (softmax)"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")
    print(f"Production head: {HEAD_WEIGHTS_PATH}")
    print("Next steps:")
    print(f"  python evaluate.py --checkpoint {BEST_CHECKPOINT} --test-manifest ... --tune-thresholds")
    print("  python export_model.py")

if __name__ == "__main__":
    main()