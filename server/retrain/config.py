"""
config.py - Clean configuration for 7-parent GalleryFL retraining.

Single source of truth for the 7-parent single-label classifier.
"""

import os
from dataclasses import dataclass
from typing import List

# === 7 PARENT LABELS (single-label multiclass) ===
LABELS: List[str] = [
    "people", "places", "activities", "objects",
    "documents", "nature", "events"
]
NUM_CLASSES: int = len(LABELS)
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_LABEL = {i: name for i, name in enumerate(LABELS)}

# === DATASET ===
DATASET_DIR = os.environ.get(
    "DATASET_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "gallery_7tag")
)

# Manifests (preferred) + folder fallback in prepare_dataset
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# === TRAINING HYPERPARAMS (tuned for 7-class COCO proxy) ===
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 8e-4
DROPOUT = 0.45
HIDDEN_SIZE = 256
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 18          # on val macro F1
EARLY_STOP_MIN_DELTA = 0.002

# === BACKBONE (frozen) ===
BACKBONE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "base_model.tflite")
IMAGE_SIZE = 224
FEATURE_DIM = 1024

# === OUTPUT PATHS ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
RETRAIN_DIR = os.path.join(OUTPUT_DIR, "retrain")

HEAD_WEIGHTS_PATH = os.path.join(MODELS_DIR, "head_weights.npz")
BEST_CHECKPOINT = os.path.join(RETRAIN_DIR, "best_checkpoint.npz")
METRICS_PATH = os.path.join(RETRAIN_DIR, "retrain_metrics.json")
THRESHOLDS_PATH = os.path.join(RETRAIN_DIR, "thresholds.json")
MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "model_version.txt")
FEATURE_NORM_PATH = os.path.join(RETRAIN_DIR, "feature_norm.json")

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
    weight_decay: float = WEIGHT_DECAY

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
    print("NUM_CLASSES:", NUM_CLASSES)
    print("DATASET_DIR:", DATASET_DIR)