"""Frozen TFLite feature extraction shared by train and evaluate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from config import BACKBONE_PATH, FEATURE_DIM, IMAGE_SIZE
from pipeline import PipelineError


class FrozenBackbone:
    """Run the canonical GalleryFL TFLite backbone one image at a time."""

    def __init__(self, model_path: Path | str = BACKBONE_PATH):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise PipelineError(f"Backbone not found: {self.model_path}")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise PipelineError(
                "TensorFlow is required for TFLite feature extraction. "
                "Install server/requirements.txt."
            ) from exc

        self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path), num_threads=4)
        self._interpreter.allocate_tensors()
        inputs = self._interpreter.get_input_details()
        outputs = self._interpreter.get_output_details()
        if len(inputs) != 1:
            raise PipelineError(f"Expected one backbone input, found {len(inputs)}")
        self._input = inputs[0]
        input_shape = tuple(int(value) for value in self._input["shape"])
        if input_shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3) or self._input["dtype"] != np.float32:
            raise PipelineError(
                f"Backbone input must be float32 [1,{IMAGE_SIZE},{IMAGE_SIZE},3], "
                f"found {self._input['dtype']} {input_shape}"
            )

        candidates = [
            output
            for output in outputs
            if tuple(int(value) for value in output["shape"]) == (1, FEATURE_DIM)
            and output["dtype"] == np.float32
        ]
        if len(candidates) != 1:
            shapes = [tuple(int(value) for value in output["shape"]) for output in outputs]
            raise PipelineError(
                f"Backbone must expose exactly one float32 [1,{FEATURE_DIM}] projection; "
                f"found output shapes {shapes}"
            )
        self._projection = candidates[0]

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def preprocess(path: Path | str) -> np.ndarray:
        path = Path(path)
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                pixels = np.asarray(image, dtype=np.float32)
        except Exception as exc:
            raise PipelineError(f"Could not decode image {path}: {exc}") from exc
        return (pixels / 127.5 - 1.0)[None, ...].astype(np.float32)

    def extract(self, paths: Sequence[Path | str], progress_label: str = "features") -> np.ndarray:
        if not paths:
            raise PipelineError("Cannot extract features from an empty image list")
        features = np.empty((len(paths), FEATURE_DIM), dtype=np.float32)
        print(f"[{progress_label}] extracting {len(paths)} images with frozen TFLite backbone")
        progress_step = max(1, len(paths) // 20)
        for index, path in enumerate(paths):
            image = self.preprocess(path)
            self._interpreter.set_tensor(self._input["index"], image)
            self._interpreter.invoke()
            projection = self._interpreter.get_tensor(self._projection["index"])
            features[index] = projection.reshape(FEATURE_DIM)
            if (index + 1) % progress_step == 0 or index + 1 == len(paths):
                print(f"[{progress_label}] {index + 1}/{len(paths)}")
        if not np.all(np.isfinite(features)):
            raise PipelineError("Backbone produced non-finite features")
        return features
