import numpy as np
from typing import List, Tuple
import os
import zlib
import base64
import struct
import logging

NUM_CLASSES = 20

LAYER_SCHEMA: List[Tuple[str, Tuple[int, ...]]] = [
    ("w1", (1024, 256)),
    ("b1", (256,)),
    ("w2", (256, NUM_CLASSES)),
    ("b2", (NUM_CLASSES,)),
]

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.global_weights: List[np.ndarray] = []
        self.current_version: int = 0

        self._load_or_initialize()

    def _load_or_initialize(self):
        snapshot_path = os.path.join(self.model_dir, "head_weights.npz")
        if os.path.exists(snapshot_path):
            try:
                data = np.load(snapshot_path, allow_pickle=True)
                self.global_weights = [data[name] for name, _ in LAYER_SCHEMA]
                for w, (name, shape) in zip(self.global_weights, LAYER_SCHEMA):
                    assert w.shape == shape, f"{name}: expected {shape}, got {w.shape}"
                version_path = os.path.join(self.model_dir, "model_version.txt")
                if os.path.exists(version_path):
                    with open(version_path, 'r') as f:
                        self.current_version = int(f.read().strip())
                else:
                    self.current_version = 1
                logging.info(f"Loaded model weights v{self.current_version} from disk")
                return
            except Exception as e:
                logging.warning(f"Failed to load saved weights: {e}, initializing fresh")

        self._initialize_weights()

    def _initialize_weights(self):
        self.global_weights = []
        for name, shape in LAYER_SCHEMA:
            if name.startswith("b"):
                self.global_weights.append(np.zeros(shape, dtype=np.float32))
            else:
                fan_in = shape[0]
                fan_out = shape[1] if len(shape) > 1 else shape[0]
                self.global_weights.append(
                    (np.random.randn(*shape) * np.sqrt(2.0 / (fan_in + fan_out))).astype(np.float32)
                )
        self.current_version = 1
        self._save_snapshot()

    def _save_snapshot(self):
        snapshot_path = os.path.join(self.model_dir, "head_weights.npz")
        save_dict = {name: w for (name, _), w in zip(LAYER_SCHEMA, self.global_weights)}
        np.savez_compressed(snapshot_path, **save_dict)

        version_path = os.path.join(self.model_dir, "model_version.txt")
        with open(version_path, 'w') as f:
            f.write(str(self.current_version))

    def get_serialized_weights(self) -> str:
        byte_chunks = []
        for w, (name, shape) in zip(self.global_weights, LAYER_SCHEMA):
            flat = w.flatten()
            layer_bytes = flat.tobytes()
            byte_chunks.append(struct.pack('<I', len(layer_bytes)) + layer_bytes)

        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined, level=6)
        return base64.b64encode(compressed).decode('ascii')

    def deserialize_weights(self, encoded: str) -> List[np.ndarray]:
        compressed = base64.b64decode(encoded)
        combined = zlib.decompress(compressed)

        offset = 0
        weights = []
        for name, shape in LAYER_SCHEMA:
            size = struct.unpack_from('<I', combined, offset)[0]
            offset += 4
            layer = np.frombuffer(combined[offset:offset + size], dtype=np.float32).copy().reshape(shape)
            offset += size
            weights.append(layer)

        return weights

    def update_global_weights(self, new_weights: List[np.ndarray]):
        for w, (name, shape) in zip(new_weights, LAYER_SCHEMA):
            assert w.shape == shape, f"{name}: expected {shape}, got {w.shape}"
        self.global_weights = [w.astype(np.float32) for w in new_weights]
        self.current_version += 1
        self._save_snapshot()
        logging.info(f"Global model updated to v{self.current_version}")

    def get_weight_shapes(self) -> List[Tuple[int, ...]]:
        return [shape for _, shape in LAYER_SCHEMA]

    def get_total_parameters(self) -> int:
        total = 0
        for _, shape in LAYER_SCHEMA:
            p = 1
            for d in shape:
                p *= d
            total += p
        return total
