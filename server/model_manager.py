import base64
import json
import logging
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

from taxonomy_parser import TaxonomyParser, get_model_schema

BASE_DIR = Path(__file__).resolve().parent
try:
    taxonomy = TaxonomyParser(BASE_DIR / "taxonomy.json")
    CLASS_LABELS = taxonomy.model_labels
    NUM_CLASSES = taxonomy.num_classes
except Exception as exc:
    raise RuntimeError(f"Failed to load taxonomy.json: {exc}") from exc

LAYER_SCHEMA: List[Tuple[str, Tuple[int, ...]]] = get_model_schema(NUM_CLASSES)


class ModelManager:
    """Own the strict four-tensor, seven-parent FL head contract."""

    def __init__(self, model_dir: str = "models"):
        model_path = Path(model_dir)
        if not model_path.is_absolute() and model_dir == "models":
            model_path = BASE_DIR / model_path
        self.model_dir = str(model_path.resolve())
        os.makedirs(self.model_dir, exist_ok=True)
        self.global_weights: List[np.ndarray] = []
        self.current_version: int = 0
        self._validate_schema_file()
        self._load_or_initialize()
        self._print_startup_metadata()

    def _print_startup_metadata(self) -> None:
        logging.info("=== ModelManager Startup Metadata ===")
        logging.info("Model Version: %s", self.current_version)
        logging.info("NUM_CLASSES: %s", NUM_CLASSES)
        logging.info("Class labels: %s", CLASS_LABELS)
        logging.info("Layer Order: %s", [name for name, _ in LAYER_SCHEMA])
        logging.info("Layer Shapes: %s", [shape for _, shape in LAYER_SCHEMA])
        logging.info("=====================================")

    def _validate_schema_file(self) -> None:
        schema_path = Path(self.model_dir) / "model_schema.json"
        if not schema_path.is_file():
            raise RuntimeError(f"Schema file not found at {schema_path}")
        with schema_path.open("r", encoding="utf-8") as handle:
            schema_data = json.load(handle)
        if int(schema_data.get("num_classes", -1)) != NUM_CLASSES:
            raise RuntimeError(
                f"Schema num_classes mismatch: expected {NUM_CLASSES}, "
                f"got {schema_data.get('num_classes')}"
            )
        labels = schema_data.get("labels")
        if labels is not None and labels != CLASS_LABELS:
            raise RuntimeError(f"Schema labels mismatch: expected {CLASS_LABELS}, got {labels}")
        json_layers = schema_data.get("layers", [])
        if len(json_layers) != len(LAYER_SCHEMA):
            raise RuntimeError(
                f"Schema layer count mismatch: expected {len(LAYER_SCHEMA)}, got {len(json_layers)}"
            )
        for (name, shape), json_layer in zip(LAYER_SCHEMA, json_layers):
            if json_layer.get("name") != name or tuple(json_layer.get("shape", [])) != shape:
                raise RuntimeError(
                    f"Schema mismatch for {name}: expected {shape}, got {json_layer}"
                )

    def _load_or_initialize(self) -> None:
        active_snapshot = Path(self.model_dir) / "head_weights.npz"
        initial_snapshot = Path(self.model_dir) / "initial_head_weights.npz"
        target_snapshot = active_snapshot if active_snapshot.exists() else initial_snapshot
        if not target_snapshot.exists():
            raise RuntimeError(
                f"Neither {active_snapshot} nor {initial_snapshot} exists"
            )

        try:
            with np.load(target_snapshot, allow_pickle=False) as data:
                weights: List[np.ndarray] = []
                for name, expected_shape in LAYER_SCHEMA:
                    if name not in data:
                        raise ValueError(f"Missing key {name!r}")
                    value = np.asarray(data[name], dtype=np.float32)
                    if value.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: expected {expected_shape}, got {value.shape}. "
                            "Export a seven-parent head with server/retrain/export_model.py."
                        )
                    if not np.all(np.isfinite(value)):
                        raise ValueError(f"Non-finite values in {name}")
                    weights.append(value)
            self.global_weights = weights
            version_path = Path(self.model_dir) / "model_version.txt"
            if target_snapshot == active_snapshot and version_path.exists():
                self.current_version = int(version_path.read_text(encoding="utf-8").strip())
            else:
                self.current_version = 1
            logging.info("Loaded model weights v%s from %s", self.current_version, target_snapshot)
        except Exception as exc:
            raise RuntimeError(f"Failed to load saved weights from {target_snapshot}: {exc}") from exc

    def _save_snapshot(self) -> None:
        snapshot_path = Path(self.model_dir) / "head_weights.npz"
        payload = {name: weight for (name, _), weight in zip(LAYER_SCHEMA, self.global_weights)}
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.model_dir, suffix=".npz", delete=False) as handle:
                temp_name = handle.name
                np.savez_compressed(handle, **payload)
            os.replace(temp_name, snapshot_path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        (Path(self.model_dir) / "model_version.txt").write_text(
            f"{self.current_version}\n", encoding="utf-8"
        )

    def get_serialized_weights(self) -> str:
        byte_chunks = []
        for weight, (_, shape) in zip(self.global_weights, LAYER_SCHEMA):
            layer_bytes = weight.reshape(shape).astype("<f4", copy=False).tobytes()
            byte_chunks.append(struct.pack("<I", len(layer_bytes)) + layer_bytes)
        compressed = zlib.compress(b"".join(byte_chunks), level=6)
        return base64.b64encode(compressed).decode("ascii")

    def deserialize_weights(self, encoded: str) -> List[np.ndarray]:
        try:
            combined = zlib.decompress(base64.b64decode(encoded, validate=True))
        except Exception as exc:
            raise ValueError("Payload is not valid base64/zlib data") from exc
        offset = 0
        weights: List[np.ndarray] = []
        for name, shape in LAYER_SCHEMA:
            if offset + 4 > len(combined):
                raise ValueError(f"Missing length prefix for {name}")
            size = struct.unpack_from("<I", combined, offset)[0]
            offset += 4
            expected_size = int(np.prod(shape)) * np.dtype("<f4").itemsize
            if size != expected_size:
                raise ValueError(f"Invalid byte length for {name}: expected {expected_size}, got {size}")
            if offset + size > len(combined):
                raise ValueError(f"Truncated tensor payload for {name}")
            value = np.frombuffer(combined[offset : offset + size], dtype="<f4").copy().reshape(shape)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite values in {name}")
            weights.append(value)
            offset += size
        if offset != len(combined):
            raise ValueError("Unexpected trailing bytes in weight payload")
        return weights

    def get_schema(self) -> dict:
        return {
            "schema_version": 1,
            "serialization": "zlib(base64(little_endian_uint32_length + little_endian_float32_tensor))*",
            "task": "single_label_multiclass",
            "decision_rule": "argmax",
            "num_classes": NUM_CLASSES,
            "labels": CLASS_LABELS,
            "layers": [
                {"name": name, "shape": list(shape), "dtype": "float32"}
                for name, shape in LAYER_SCHEMA
            ],
        }

    def update_global_weights(self, new_weights: List[np.ndarray]) -> None:
        if len(new_weights) != len(LAYER_SCHEMA):
            raise ValueError(f"Expected {len(LAYER_SCHEMA)} tensors, got {len(new_weights)}")
        validated: List[np.ndarray] = []
        for value, (name, shape) in zip(new_weights, LAYER_SCHEMA):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name}: contains non-finite values")
            validated.append(array)
        if self.global_weights:
            self.last_delta = [new - old for new, old in zip(validated, self.global_weights)]
        self.global_weights = validated
        self.current_version += 1
        self._save_snapshot()
        logging.info("Global model updated to v%s", self.current_version)

    def get_serialized_delta(self) -> str:
        if not getattr(self, "last_delta", None):
            return self.get_serialized_weights()
        byte_chunks = []
        for delta, (_, shape) in zip(self.last_delta, LAYER_SCHEMA):
            layer_bytes = delta.reshape(shape).astype("<f4", copy=False).tobytes()
            byte_chunks.append(struct.pack("<I", len(layer_bytes)) + layer_bytes)
        return base64.b64encode(zlib.compress(b"".join(byte_chunks), level=6)).decode("ascii")

    def get_weight_shapes(self) -> List[Tuple[int, ...]]:
        return [shape for _, shape in LAYER_SCHEMA]

    def get_total_parameters(self) -> int:
        return int(sum(np.prod(shape) for _, shape in LAYER_SCHEMA))
