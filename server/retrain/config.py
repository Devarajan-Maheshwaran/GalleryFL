"""
config.py - Clean configuration for 7-parent GalleryFL retraining.

Single source of truth for dataset and training.
"""

import os
from dataclasses import dataclass
from typing import List

# === DATASET CONFIG (set via env or CLI) ===
# Expected structure after conversion:
# DATASET_DIR/
#   people/ *.jpg
#   places/
#   activities/
#   objects/
#   documents/
#   nature/
#   events/
DATASET_DIR = os.environ.get(
    "DATASET_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "gallery_7tag")
)

# 7 PARENT CATEGORIES (single-label classification target)
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
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_LABEL = {i: name for i, name in enumerate(LABELS)}

# Training splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model / Training hyperparams
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DROPOUT = 0.4
HIDDEN_SIZE = 256
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4

# Backbone (must be present)
BACKBONE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "base_model.tflite")
IMAGE_SIZE = 224
FEATURE_DIM = 1024

# Output paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
RETRAIN_DIR = os.path.join(OUTPUT_DIR, "retrain")

HEAD_WEIGHTS_PATH = os.path.join(MODELS_DIR, "head_weights.npz")
BEST_CHECKPOINT = os.path.join(RETRAIN_DIR, "best_checkpoint.npz")
METRICS_PATH = os.path.join(RETRAIN_DIR, "retrain_metrics.json")
THRESHOLDS_PATH = os.path.join(RETRAIN_DIR, "thresholds.json")
MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "model_version.txt")

@dataclass
class RetrainConfig:
    dataset_dir: str = DATASET_DIR
    labels: List[str] = None
    num_classes: int = NUM_CLASSES
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    hidden_size: int = HIDDEN_SIZE
    dropout: float = DROPOUT
    label_smoothing: float = LABEL_SMOOTHING

    def __post_init__(self):
        if self.labels is None:
            self.labels = LABELS[:]
        os.makedirs(RETRAIN_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

def get_config(overrides=None) -> RetrainConfig:
    cfg = RetrainConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg

if __name__ == "__main__":
    print("7-Parent Labels:", LABELS)
    print("DATASET_DIR:", DATASET_DIR)
    print("Expected folders:", LABELS)