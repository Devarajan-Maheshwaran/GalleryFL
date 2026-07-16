import numpy as np
from typing import List, Optional
import os
import zlib
import base64
import struct

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.global_weights: List[np.ndarray] = []
        self.current_version: int = 0
        
        # In a real scenario, this would load from a saved state or initialize randomly
        # Since this is a test/weekend project, we'll initialize with zeros if empty
        self.initialize_weights()

    def initialize_weights(self):
        # We need the shapes of the ClassificationHead
        # Layer 1: 960 -> 256
        # Layer 2: 256 -> num_classes (e.g., 20)
        num_classes = 20
        self.global_weights = [
            np.zeros((960, 256), dtype=np.float32),  # W1
            np.zeros((256,), dtype=np.float32),       # b1
            np.zeros((256, num_classes), dtype=np.float32), # W2
            np.zeros((num_classes,), dtype=np.float32)      # b2
        ]
        self.current_version = 1

    def get_serialized_weights(self) -> str:
        """
        Serializes weights into Little-Endian bytes, zlib compressed, base64 encoded.
        Format:
        [Size of layer 1 (4 bytes)][Layer 1 bytes][Size of layer 2 (4 bytes)][Layer 2 bytes]...
        """
        byte_chunks = []
        for layer in self.global_weights:
            # Flatten and pack as little-endian float32
            flat = layer.flatten()
            fmt = f'<{len(flat)}f'
            layer_bytes = struct.pack(fmt, *flat)
            
            # Prepend size of layer bytes as little-endian uint32
            size_bytes = struct.pack('<I', len(layer_bytes))
            byte_chunks.append(size_bytes + layer_bytes)
            
        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined)
        return base64.b64encode(compressed).decode('ascii')

    def deserialize_weights(self, encoded: str, expected_shapes: List[tuple]) -> List[np.ndarray]:
        """
        Decodes base64 -> zlib decompress -> little-endian unpack to float32
        """
        compressed = base64.b64decode(encoded)
        combined = zlib.decompress(compressed)
        
        offset = 0
        weights = []
        for shape in expected_shapes:
            # Read size
            size = struct.unpack_from('<I', combined, offset)[0]
            offset += 4
            
            # Read layer bytes
            num_floats = size // 4
            fmt = f'<{num_floats}f'
            layer_floats = struct.unpack_from(fmt, combined, offset)
            offset += size
            
            layer = np.array(layer_floats, dtype=np.float32).reshape(shape)
            weights.append(layer)
            
        return weights

    def update_global_weights(self, new_weights: List[np.ndarray]):
        self.global_weights = new_weights
        self.current_version += 1
        
        # Save snapshot every 5 versions to prevent memory bloat
        # if self.current_version % 5 == 0:
        #    self.save_snapshot()
