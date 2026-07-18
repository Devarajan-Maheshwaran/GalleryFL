import numpy as np
from typing import List, Tuple
import os
import json
import zlib
import base64
import struct
import logging
import sys

# Attempt to load taxonomy parser
try:
    from taxonomy_parser import TaxonomyParser
    taxonomy = TaxonomyParser("taxonomy.json")
    NUM_CLASSES = taxonomy.num_classes
    LEAF_CLASSES = taxonomy.leaf_names
except Exception as e:
    logging.error(f"Failed to load taxonomy.json: {e}")
    sys.exit(1)

# Schema defined strictly for the ClassificationHead
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

        self._validate_schema_file()
        self._load_or_initialize()
        
        self._print_startup_metadata()

    def _print_startup_metadata(self):
        logging.info("=== ModelManager Startup Metadata ===")
        logging.info(f"Model Version: {self.current_version}")
        logging.info(f"NUM_CLASSES: {NUM_CLASSES}")
        logging.info(f"Layer Order: {[n for n, _ in LAYER_SCHEMA]}")
        logging.info(f"Layer Shapes: {[s for _, s in LAYER_SCHEMA]}")
        logging.info(f"Leaf Classes: {LEAF_CLASSES[:5]}... ({len(LEAF_CLASSES)} total)")
        logging.info("=====================================")

    def _validate_schema_file(self):
        schema_path = os.path.join(self.model_dir, "model_schema.json")
        if not os.path.exists(schema_path):
            logging.error(f"Schema file not found at {schema_path}! Fail fast.")
            sys.exit(1)
            
        with open(schema_path, "r") as f:
            schema_data = json.load(f)
            
        json_layers = schema_data.get("layers", [])
        if len(json_layers) != len(LAYER_SCHEMA):
            logging.error(f"Schema layer count mismatch: expected {len(LAYER_SCHEMA)}, got {len(json_layers)}")
            sys.exit(1)
            
        for (name, shape), json_layer in zip(LAYER_SCHEMA, json_layers):
            if json_layer["name"] != name:
                logging.error(f"Schema name mismatch: expected {name}, got {json_layer['name']}")
                sys.exit(1)
            if tuple(json_layer["shape"]) != shape:
                logging.error(f"Schema shape mismatch for {name}: expected {shape}, got {tuple(json_layer['shape'])}")
                sys.exit(1)

    def _load_or_initialize(self):
        active_snapshot = os.path.join(self.model_dir, "head_weights.npz")
        initial_snapshot = os.path.join(self.model_dir, "initial_head_weights.npz")
        
        target_snapshot = active_snapshot if os.path.exists(active_snapshot) else initial_snapshot
        
        if not os.path.exists(target_snapshot):
            logging.error(f"Neither {active_snapshot} nor {initial_snapshot} exists! Fail fast.")
            sys.exit(1)

        try:
            data = np.load(target_snapshot, allow_pickle=True)
            self.global_weights = []
            
            for name, expected_shape in LAYER_SCHEMA:
                if name not in data:
                    logging.error(f"Missing key '{name}' in NPZ file. Fail fast.")
                    sys.exit(1)
                w = data[name]
                if w.shape != expected_shape:
                    logging.error(f"Shape mismatch for '{name}': expected {expected_shape}, got {w.shape}. Fail fast.")
                    sys.exit(1)
                self.global_weights.append(w)
                
            version_path = os.path.join(self.model_dir, "model_version.txt")
            if target_snapshot == active_snapshot and os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    self.current_version = int(f.read().strip())
            else:
                self.current_version = 1
                
            logging.info(f"Loaded model weights v{self.current_version} from {target_snapshot}")
        except Exception as e:
            logging.error(f"Failed to load saved weights: {e}. Fail fast.")
            sys.exit(1)

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
            layer_bytes = flat.astype('<f4').tobytes()
            byte_chunks.append(struct.pack('<I', len(layer_bytes)) + layer_bytes)

        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined, level=6)
        return base64.b64encode(compressed).decode('ascii')

    def deserialize_weights(self, encoded: str) -> List[np.ndarray]:
        """Decode the strict, canonical FGT v1 head-weight wire format."""
        try:
            compressed = base64.b64decode(encoded, validate=True)
            combined = zlib.decompress(compressed)
        except Exception as exc:
            raise ValueError("Payload is not valid base64/zlib data") from exc

        offset = 0
        weights = []
        for name, shape in LAYER_SCHEMA:
            if offset + 4 > len(combined):
                raise ValueError(f"Missing length prefix for {name}")
            size = struct.unpack_from('<I', combined, offset)[0]
            offset += 4
            expected_size = int(np.prod(shape)) * np.dtype('<f4').itemsize
            if size != expected_size:
                raise ValueError(f"Invalid byte length for {name}: expected {expected_size}, got {size}")
            if offset + size > len(combined):
                raise ValueError(f"Truncated tensor payload for {name}")
            layer = np.frombuffer(combined[offset:offset + size], dtype='<f4').copy().reshape(shape)
            offset += size
            weights.append(layer)

        if offset != len(combined):
            raise ValueError("Unexpected trailing bytes in weight payload")
        return weights

    def get_schema(self) -> dict:
        return {
            "schema_version": 1,
            "serialization": "zlib(base64(little_endian_uint32_length + little_endian_float32_tensor))*",
            "num_classes": NUM_CLASSES,
            "layers": [{"name": name, "shape": list(shape), "dtype": "float32"}
                       for name, shape in LAYER_SCHEMA],
        }

    def update_global_weights(self, new_weights: List[np.ndarray]):
        for w, (name, shape) in zip(new_weights, LAYER_SCHEMA):
            assert w.shape == shape, f"{name}: expected {shape}, got {w.shape}"
            
        if self.global_weights:
            self.last_delta = [w - old for w, old in zip(new_weights, self.global_weights)]
            
        self.global_weights = [w.astype(np.float32) for w in new_weights]
        self.current_version += 1
        self._save_snapshot()
        logging.info(f"Global model updated to v{self.current_version}")

    def get_serialized_delta(self) -> str:
        if not hasattr(self, 'last_delta') or not self.last_delta:
            return self.get_serialized_weights()
            
        byte_chunks = []
        for w, (name, shape) in zip(self.last_delta, LAYER_SCHEMA):
            flat = w.flatten()
            layer_bytes = flat.astype('<f4').tobytes()
            byte_chunks.append(struct.pack('<I', len(layer_bytes)) + layer_bytes)

        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined, level=6)
        return base64.b64encode(compressed).decode('ascii')

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
