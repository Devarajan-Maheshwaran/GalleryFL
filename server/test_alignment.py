import os
import json
import numpy as np
from taxonomy_parser import TaxonomyParser, get_model_schema

def test_taxonomy_parsing():
    parser = TaxonomyParser("taxonomy.json")
    assert parser.num_classes == 34
    print("Taxonomy parsing test passed.")

def test_schema_alignment():
    schema = get_model_schema(34)
    assert schema[0] == ("w1", (1024, 256))
    assert schema[1] == ("b1", (256,))
    assert schema[2] == ("w2", (256, 34))
    assert schema[3] == ("b2", (34,))
    print("Schema alignment test passed.")

def test_initial_weights_format():
    if not os.path.exists("models/initial_head_weights.npz"):
        print("Skipping weights format test (not generated yet)")
        return
        
    data = np.load("models/initial_head_weights.npz")
    assert list(data.keys()) == ["w1", "b1", "w2", "b2"]
    assert data["w1"].shape == (1024, 256)
    assert data["b1"].shape == (256,)
    assert data["w2"].shape == (256, 34)
    assert data["b2"].shape == (34,)
    print("Initial weights format test passed.")

def test_model_schema_matches_weights():
    if not os.path.exists("models/model_schema.json"):
        return
    with open("models/model_schema.json", "r") as f:
        schema = json.load(f)
    layers = schema.get("layers", [])
    
    assert layers[0]["name"] == "w1"
    assert tuple(layers[0]["shape"]) == (1024, 256)
    
    assert layers[1]["name"] == "b1"
    assert tuple(layers[1]["shape"]) == (256,)
    
    assert layers[2]["name"] == "w2"
    assert tuple(layers[2]["shape"]) == (256, 34)
    
    assert layers[3]["name"] == "b2"
    assert tuple(layers[3]["shape"]) == (34,)
    print("Model schema JSON alignment test passed.")

if __name__ == "__main__":
    test_taxonomy_parsing()
    test_schema_alignment()
    test_initial_weights_format()
    test_model_schema_matches_weights()
    print("All alignment checks passed.")
