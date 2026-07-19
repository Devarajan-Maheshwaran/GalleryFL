import hashlib
import json
from pathlib import Path

import numpy as np

from taxonomy_parser import TaxonomyParser, get_model_schema

LABELS = ["people", "places", "activities", "objects", "documents", "nature", "events"]


def test_taxonomy_parsing():
    parser = TaxonomyParser("taxonomy.json")
    assert parser.num_classes == 7
    assert parser.num_leaf_tags == 34
    assert parser.model_labels == LABELS


def test_schema_alignment():
    assert get_model_schema(7) == [
        ("w1", (1024, 256)),
        ("b1", (256,)),
        ("w2", (256, 7)),
        ("b2", (7,)),
    ]


def test_initial_weights_format():
    with np.load("models/initial_head_weights.npz", allow_pickle=False) as data:
        assert set(data.files) == {"w1", "b1", "w2", "b2"}
        assert data["w1"].shape == (1024, 256)
        assert data["b1"].shape == (256,)
        assert data["w2"].shape == (256, 7)
        assert data["b2"].shape == (7,)
        assert all(np.all(np.isfinite(data[key])) for key in data.files)


def test_model_schema_matches_weights():
    with open("models/model_schema.json", encoding="utf-8") as handle:
        schema = json.load(handle)
    assert schema["num_classes"] == 7
    assert schema["labels"] == LABELS
    assert [(layer["name"], tuple(layer["shape"])) for layer in schema["layers"]] == get_model_schema(7)


def test_android_and_server_use_identical_backbone():
    server_model = Path("models/base_model.tflite").read_bytes()
    android_model = Path("../android/app/src/main/assets/models/base_model.tflite").read_bytes()
    assert hashlib.sha256(server_model).digest() == hashlib.sha256(android_model).digest()
