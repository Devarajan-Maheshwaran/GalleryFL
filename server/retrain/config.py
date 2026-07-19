"""
Central configuration for the clean 7-parent GalleryFL retraining pipeline.

Usage:
    export DATASET_DIR=/path/to/your/7tag/gallery
    python -m server.retrain.train --config server/retrain/config.py

Or override via CLI / env.
"""

import os
from dataclasses import dataclass
from typing import List

# === SINGLE SOURCE OF TRUTH FOR DATASET ===
# Set this via environment or CLI. Example:
#   export DATASET_DIR=/home/user/my_photos/7tag
#   DATASET_DIR must contain 7 subfolders named exactly after the labels below.
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "gallery_7tag"))

# 7 CORE PARENT CATEGORIES (the only labels we train on)
LABELS: List[str] = [
    "people",
    "places",
    "activities",
    "objects",
    "documents",
    "nature",
    "events",
]
NUM_CLASSES: int = len(LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# Splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training hyperparams (tuned for macro-F1 on small/medium gallery sets)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
DROPOUT = 0.3
HIDDEN_SIZE = 256

# Class balancing
USE_CLASS_WEIGHTS = True

# Threshold tuning (per-class or global)
TUNE_THRESHOLDS = True
DEFAULT_THRESHOLD = 0.5

# Paths (relative to server/)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
RETRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "retrain")

# Final artifacts
HEAD_WEIGHTS_PATH = os.path.join(MODELS_DIR, "head_weights.npz")
BEST_CHECKPOINT = os.path.join(RETRAIN_OUTPUT_DIR, "best_checkpoint.npz")
METRICS_REPORT = os.path.join(RETRAIN_OUTPUT_DIR, "retrain_metrics.json")
MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "model_version.txt")

# Backbone (must exist)
BACKBONE_PATH = os.path.join(MODELS_DIR, "base_model.tflite")
IMAGE_SIZE = 224
FEATURE_DIM = 1024

@dataclass
class RetrainConfig:
    dataset_dir: str = DATASET_DIR
    labels: List[str] = None
    num_classes: int = NUM_CLASSES
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr: float = LEARNING_RATE
    hidden: int = HIDDEN_SIZE
    use_class_weights: bool = USE_CLASS_WEIGHTS
    tune_thresholds: bool = TUNE_THRESHOLDS
    output_dir: str = RETRAIN_OUTPUT_DIR

    def __post_init__(self):
        if self.labels is None:
            self.labels = LABELS[:]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

def get_config(overrides: dict = None) -> RetrainConfig:
    cfg = RetrainConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg

# For easy import in scripts
if __name__ == "__main__":
    print("7-tag labels:", LABELS)
    print("DATASET_DIR (current):", DATASET_DIR)
    print("Expected structure:")
    for lbl in LABELS:
        print(f"  {DATASET_DIR}/{lbl}/  (images)")
