#!/usr/bin/env python3
"""Train the seven-parent GalleryFL head with TensorFlow/Keras."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np

from backbone import FrozenBackbone
from config import (
    BEST_CHECKPOINT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    FEATURE_DIM,
    HIDDEN_SIZE,
    LABELS,
    NUM_CLASSES,
    RANDOM_SEED,
    TRAIN_METRICS_PATH,
    TrainConfig,
    ensure_output_dirs,
)
from pipeline import (
    PipelineError,
    atomic_save_npz,
    atomic_write_json,
    balanced_class_weights,
    feature_statistics,
    load_checkpoint,
    load_manifest,
    multiclass_metrics,
    softmax,
    standardize,
    validate_split_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def set_reproducible_seed(tf, seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(tf, config: TrainConfig):
    inputs = tf.keras.Input(shape=(FEATURE_DIM,), name="standardized_features")
    hidden = tf.keras.layers.Dense(
        config.hidden_size,
        activation="relu",
        kernel_initializer="he_normal",
        name="hidden",
    )(inputs)
    hidden = tf.keras.layers.Dropout(config.dropout, name="dropout")(hidden)
    logits = tf.keras.layers.Dense(NUM_CLASSES, name="logits")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="galleryfl_parent_head")
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        clipnorm=5.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    config.validate()
    ensure_output_dirs()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise PipelineError(
            "TensorFlow is required. Install dependencies with "
            "python -m pip install -r server/requirements.txt"
        ) from exc
    set_reproducible_seed(tf, config.seed)

    manifest_dir = config.dataset_dir / "manifests"
    splits = {
        name: load_manifest(manifest_dir / f"{name}.csv", config.dataset_dir)
        for name in ("train", "val", "test")
    }
    validate_split_manifests(splits)
    train_items = splits["train"]
    val_items = splits["val"]
    train_labels = np.asarray([item.label for item in train_items], dtype=np.int64)
    val_labels = np.asarray([item.label for item in val_items], dtype=np.int64)
    class_weights = balanced_class_weights(train_labels)

    print("GalleryFL seven-parent single-label training")
    print(
        f"train={len(train_items)} val={len(val_items)} test={len(splits['test'])} "
        "(test is not read during training)"
    )
    print(
        "class weights:",
        {LABELS[index]: round(value, 4) for index, value in class_weights.items()},
    )

    backbone = FrozenBackbone()
    train_features_raw = backbone.extract([item.path for item in train_items], "train")
    val_features_raw = backbone.extract([item.path for item in val_items], "val")
    feature_mean, feature_std = feature_statistics(train_features_raw)
    train_features = standardize(train_features_raw, feature_mean, feature_std)
    val_features = standardize(val_features_raw, feature_mean, feature_std)

    model = build_model(tf, config)

    class MacroF1Checkpoint(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best_f1 = -np.inf
            self.best_epoch = 0
            self.wait = 0
            self.records: list[dict] = []

        def on_epoch_end(self, epoch, logs=None) -> None:
            logs = logs or {}
            logits = self.model.predict(val_features, batch_size=256, verbose=0)
            probabilities = softmax(logits)
            report = multiclass_metrics(val_labels, probabilities)
            macro_f1 = report["macro_f1"]
            logs["val_macro_f1"] = macro_f1
            record = {
                "epoch": int(epoch + 1),
                "loss": float(logs.get("loss", 0.0)),
                "accuracy": float(logs.get("accuracy", 0.0)),
                "val_loss": float(logs.get("val_loss", 0.0)),
                "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                "val_macro_f1": float(macro_f1),
            }
            self.records.append(record)
            print(
                f"epoch={epoch + 1:03d} val_macro_f1={macro_f1:.4f} "
                f"val_accuracy={report['accuracy']:.4f}"
            )

            if macro_f1 > self.best_f1 + EARLY_STOP_MIN_DELTA:
                self.best_f1 = macro_f1
                self.best_epoch = epoch + 1
                self.wait = 0
                w1, b1 = self.model.get_layer("hidden").get_weights()
                w2, b2 = self.model.get_layer("logits").get_weights()
                atomic_save_npz(
                    BEST_CHECKPOINT,
                    w1=w1.astype(np.float32),
                    b1=b1.astype(np.float32),
                    w2=w2.astype(np.float32),
                    b2=b2.astype(np.float32),
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    labels=np.asarray(LABELS),
                    best_epoch=np.asarray(self.best_epoch, dtype=np.int64),
                    best_val_macro_f1=np.asarray(self.best_f1, dtype=np.float32),
                )
                print(f"saved new best checkpoint: {BEST_CHECKPOINT}")
            else:
                self.wait += 1
                if self.wait >= EARLY_STOP_PATIENCE:
                    print(
                        f"early stopping: val macro F1 did not improve by "
                        f"{EARLY_STOP_MIN_DELTA} for {EARLY_STOP_PATIENCE} epochs"
                    )
                    self.model.stop_training = True

    checkpoint_callback = MacroF1Checkpoint()
    model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        class_weight=class_weights,
        callbacks=[checkpoint_callback],
        verbose=0,
    )

    best = load_checkpoint(BEST_CHECKPOINT)
    best_model = build_model(tf, config)
    best_model.get_layer("hidden").set_weights([best["w1"], best["b1"]])
    best_model.get_layer("logits").set_weights([best["w2"], best["b2"]])
    val_probabilities = softmax(best_model.predict(val_features, batch_size=256, verbose=0))
    val_report = multiclass_metrics(val_labels, val_probabilities)

    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    metrics = {
        "schema_version": 1,
        "task": "single_label_multiclass",
        "labels": list(LABELS),
        "architecture": f"frozen TFLite {FEATURE_DIM} -> Dense({HIDDEN_SIZE}, ReLU) -> Dense({NUM_CLASSES})",
        "loss": "class-weighted sparse categorical cross-entropy from logits",
        "selection_metric": "validation macro F1 using argmax",
        "best_epoch": checkpoint_callback.best_epoch,
        "best_validation": val_report,
        "class_counts": {LABELS[i]: int(counts[i]) for i in range(NUM_CLASSES)},
        "class_weights": {LABELS[i]: class_weights[i] for i in range(NUM_CLASSES)},
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "test_split_used_during_training": False,
        "backbone": {"path": "server/models/base_model.tflite", "sha256": backbone.sha256},
        "feature_normalization": "train mean/std; folded into w1/b1 during export",
        "history": checkpoint_callback.records,
    }
    atomic_write_json(TRAIN_METRICS_PATH, metrics)

    print(f"best validation macro F1: {val_report['macro_f1']:.4f}")
    print(f"best checkpoint: {BEST_CHECKPOINT}")
    print("Training did not overwrite the deployed head. Run evaluate.py, then export_model.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
