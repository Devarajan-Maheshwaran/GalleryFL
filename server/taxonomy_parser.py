import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


class TaxonomyParser:
    """Parse both the seven model parents and the detailed gallery leaf tags."""

    def __init__(self, filepath: str | os.PathLike = "taxonomy.json"):
        path = Path(filepath)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent / path
        self.filepath = path
        self.parent_labels: List[str] = []
        self.parent_names: List[str] = []
        self.leaf_names: List[str] = []
        self.leaf_to_index: Dict[str, int] = {}
        self.index_to_leaf: Dict[int, str] = {}
        self.parent_to_indices: Dict[str, List[int]] = {}
        self._load_and_parse()

    def _load_and_parse(self) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Taxonomy file not found at {self.filepath}")
        with self.filepath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        categories = data.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError("Malformed taxonomy JSON: categories must be a non-empty list")

        current_index = 0
        seen_parents: set[str] = set()
        seen_leaves: set[str] = set()
        for category in categories:
            parent_id = category.get("id")
            parent_name = category.get("name")
            children = category.get("children")
            if not isinstance(parent_id, str) or not parent_id or parent_id in seen_parents:
                raise ValueError(f"Malformed or duplicate taxonomy parent id: {parent_id!r}")
            if not isinstance(parent_name, str) or not parent_name:
                raise ValueError(f"Taxonomy parent {parent_id!r} has no display name")
            if not isinstance(children, list):
                raise ValueError(f"Taxonomy parent {parent_id!r} has no children list")

            seen_parents.add(parent_id)
            self.parent_labels.append(parent_id)
            self.parent_names.append(parent_name)
            indices: List[int] = []
            for child in children:
                if not isinstance(child, str) or not child or child in seen_leaves:
                    raise ValueError(f"Malformed or duplicate leaf tag: {child!r}")
                seen_leaves.add(child)
                self.leaf_names.append(child)
                self.leaf_to_index[child] = current_index
                self.index_to_leaf[current_index] = child
                indices.append(current_index)
                current_index += 1
            self.parent_to_indices[parent_id] = indices

    @property
    def model_labels(self) -> List[str]:
        """Ordered output labels of the deployed classifier."""
        return list(self.parent_labels)

    @property
    def num_classes(self) -> int:
        """The runtime classifier has one output per taxonomy parent."""
        return len(self.parent_labels)

    @property
    def num_leaf_tags(self) -> int:
        return len(self.leaf_names)


def get_model_schema(num_classes: int) -> List[Tuple[str, Tuple[int, ...]]]:
    return [
        ("w1", (1024, 256)),
        ("b1", (256,)),
        ("w2", (256, num_classes)),
        ("b2", (num_classes,)),
    ]


def validate_schema(weights: List[Any], num_classes: int) -> None:
    schema = get_model_schema(num_classes)
    if len(weights) != len(schema):
        raise ValueError(f"Expected {len(schema)} layers, got {len(weights)}")
    for index, (name, expected_shape) in enumerate(schema):
        if tuple(weights[index].shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {expected_shape}, got {weights[index].shape}"
            )
