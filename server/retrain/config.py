"""Configuration for the GalleryFL seven-parent training pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SERVER_DIR.parent

LABELS: Tuple[str, ...] = (
    "people",
    "places",
    "activities",
    "objects",
    "documents",
    "nature",
    "events",
)
LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}
IDX_TO_LABEL = dict(enumerate(LABELS))
NUM_CLASSES = len(LABELS)

FEATURE_DIM = 1024
HIDDEN_SIZE = 256
IMAGE_SIZE = 224

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MIN_SAMPLES_PER_CLASS = 10

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_DROPOUT = 0.35
DEFAULT_WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 1e-3
FEATURE_STD_EPSILON = 1e-6
RANDOM_SEED = 42
MIN_ACCEPTABLE_TEST_MACRO_F1 = 0.20
HISTORICAL_MACRO_F1 = 0.033

DATASET_DIR = Path(
    os.environ.get("DATASET_DIR", PROJECT_DIR / "data" / "gallery_7parent")
).expanduser()
BACKBONE_PATH = SERVER_DIR / "models" / "base_model.tflite"
ANDROID_BACKBONE_PATH = (
    PROJECT_DIR / "android" / "app" / "src" / "main" / "assets" / "models" / "base_model.tflite"
)
MODELS_DIR = SERVER_DIR / "models"
OUTPUT_DIR = SERVER_DIR / "output" / "retrain"

BEST_CHECKPOINT = OUTPUT_DIR / "best_checkpoint.npz"
TRAIN_METRICS_PATH = OUTPUT_DIR / "train_metrics.json"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"
EVAL_FEATURES_PATH = OUTPUT_DIR / "test_features.npz"
FINAL_MODEL_INFO_PATH = OUTPUT_DIR / "final_model_info.json"

HEAD_WEIGHTS_PATH = MODELS_DIR / "head_weights.npz"
MODEL_VERSION_FILE = MODELS_DIR / "model_version.txt"
MODEL_SCHEMA_PATH = MODELS_DIR / "model_schema.json"
DEPLOY_THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    dropout: float = DEFAULT_DROPOUT
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    hidden_size: int = HIDDEN_SIZE
    seed: int = RANDOM_SEED

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch-size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
