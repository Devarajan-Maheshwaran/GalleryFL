import json
import os
from typing import List, Dict, Any, Tuple

class TaxonomyParser:
    def __init__(self, filepath: str = "taxonomy.json"):
        self.filepath = filepath
        self.leaf_names: List[str] = []
        self.leaf_to_index: Dict[str, int] = {}
        self.index_to_leaf: Dict[int, str] = {}
        self.parent_to_indices: Dict[str, List[int]] = {}
        self._load_and_parse()

    def _load_and_parse(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Taxonomy file not found at {self.filepath}")

        with open(self.filepath, 'r') as f:
            data = json.load(f)

        if "categories" not in data:
            raise ValueError("Malformed taxonomy JSON: missing 'categories'")

        current_index = 0
        # Deterministic extraction of leaf classes
        for category in data["categories"]:
            parent_id = category["id"]
            if "children" not in category:
                continue

            indices = []
            for child in category["children"]:
                self.leaf_names.append(child)
                self.leaf_to_index[child] = current_index
                self.index_to_leaf[current_index] = child
                indices.append(current_index)
                current_index += 1

            self.parent_to_indices[parent_id] = indices

    @property
    def num_classes(self) -> int:
        return len(self.leaf_names)

def get_model_schema(num_classes: int) -> List[Tuple[str, Tuple[int, ...]]]:
    """Returns the single source of truth schema for classification head layers."""
    return [
        ("w1", (1024, 256)),
        ("b1", (256,)),
        ("w2", (256, num_classes)),
        ("b2", (num_classes,))
    ]

def validate_schema(weights: List[Any], num_classes: int):
    """Validates that a list of numpy arrays matches the exact schema."""
    schema = get_model_schema(num_classes)
    if len(weights) != len(schema):
        raise ValueError(f"Expected {len(schema)} layers, got {len(weights)}")

    for i, (name, expected_shape) in enumerate(schema):
        if weights[i].shape != expected_shape:
            raise ValueError(f"Shape mismatch for {name}: expected {expected_shape}, got {weights[i].shape}")
