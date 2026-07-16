import numpy as np
from typing import List
import os
import zlib
import base64
import struct
import logging

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.global_weights: List[np.ndarray] = []
        self.current_version: int = 0
        self.weight_history: List[dict] = []
        self.max_snapshots: int = 3

        self._load_or_initialize()

    def _load_or_initialize(self):
        snapshot_path = os.path.join(self.model_dir, "head_weights.npz")
        if os.path.exists(snapshot_path):
            try:
                data = np.load(snapshot_path, allow_pickle=True)
                self.global_weights = [data[f"layer_{i}"] for i in range(len(data.files))]
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
        num_classes = 20
        self.global_weights = [
            (np.random.randn(960, 256) * np.sqrt(2.0 / (960 + 256))).astype(np.float32),
            np.zeros((256,), dtype=np.float32),
            (np.random.randn(256, num_classes) * np.sqrt(2.0 / (256 + num_classes))).astype(np.float32),
            np.zeros((num_classes,), dtype=np.float32)
        ]
        self.current_version = 1
        self._save_snapshot()

    def _save_snapshot(self):
        snapshot_path = os.path.join(self.model_dir, "head_weights.npz")
        save_dict = {f"layer_{i}": w for i, w in enumerate(self.global_weights)}
        np.savez_compressed(snapshot_path, **save_dict)

        version_path = os.path.join(self.model_dir, "model_version.txt")
        with open(version_path, 'w') as f:
            f.write(str(self.current_version))

    def get_serialized_weights(self) -> str:
        byte_chunks = []
        for layer in self.global_weights:
            flat = layer.flatten()
            layer_bytes = flat.tobytes()
            size_bytes = struct.pack('<I', len(layer_bytes))
            byte_chunks.append(size_bytes + layer_bytes)

        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined, level=6)
        return base64.b64encode(compressed).decode('ascii')

    def deserialize_weights(self, encoded: str, expected_shapes: List[tuple]) -> List[np.ndarray]:
        compressed = base64.b64decode(encoded)
        combined = zlib.decompress(compressed)

        offset = 0
        weights = []
        for shape in expected_shapes:
            size = struct.unpack_from('<I', combined, offset)[0]
            offset += 4

            layer = np.frombuffer(combined[offset:offset + size], dtype=np.float32).copy().reshape(shape)
            offset += size
            weights.append(layer)

        return weights

    def update_global_weights(self, new_weights: List[np.ndarray]):
        self.global_weights = [w.astype(np.float32) for w in new_weights]
        self.current_version += 1
        self._save_snapshot()
        logging.info(f"Global model updated to v{self.current_version}")

    def get_weight_shapes(self) -> List[tuple]:
        return [w.shape for w in self.global_weights]

    def get_total_parameters(self) -> int:
        return sum(w.size for w in self.global_weights)
