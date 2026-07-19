#!/usr/bin/env python3
"""
export_model.py
Finalize and export the 7-tag head for deployment.

- Takes best checkpoint
- Optionally applies tuned thresholds
- Writes production artifacts:
    models/head_weights.npz          (the one Android/server use)
    models/model_version.txt
    output/retrain/retrain_metrics.json
    output/retrain/final_model_info.json

This script is the last step. After this the head is ready for the FL pipeline.
"""

import argparse
import json
import os
import sys
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HEAD_WEIGHTS_PATH, MODEL_VERSION_FILE,
    BEST_CHECKPOINT, METRICS_REPORT, RETRAIN_OUTPUT_DIR,
    NUM_CLASSES, LABELS
)

def load_checkpoint(path):
    d = np.load(path)
    return [d["w1"], d["b1"], d["w2"], d["b2"]]

def save_head(weights, path):
    w1, b1, w2, b2 = weights
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"[export] Saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    parser.add_argument("--metrics", default=METRICS_REPORT)
    parser.add_argument("--out-head", default=HEAD_WEIGHTS_PATH)
    parser.add_argument("--bump-version", action="store_true", default=True)
    args = parser.parse_args()

    print("=== Exporting 7-Tag Head for GalleryFL ===")

    if not os.path.exists(args.checkpoint):
        print("ERROR: No checkpoint. Run train.py first.")
        sys.exit(1)

    weights = load_checkpoint(args.checkpoint)

    # Save as production head
    save_head(weights, args.out_head)

    # Bump version
    if args.bump_version:
        version = 1
        if os.path.exists(MODEL_VERSION_FILE):
            try:
                version = int(open(MODEL_VERSION_FILE).read().strip()) + 1
            except:
                pass
        with open(MODEL_VERSION_FILE, "w") as f:
            f.write(str(version))
        print(f"[export] model_version bumped to {version}")

    # Copy / enrich metrics
    final_info = {
        "model_type": "7_parent_gallery_classifier",
        "num_classes": NUM_CLASSES,
        "labels": LABELS,
        "head_shape": {
            "w1": list(weights[0].shape),
            "b1": list(weights[1].shape),
            "w2": list(weights[2].shape),
            "b2": list(weights[3].shape),
        },
        "source_checkpoint": os.path.abspath(args.checkpoint),
    }

    if os.path.exists(args.metrics):
        with open(args.metrics) as f:
            metrics = json.load(f)
        final_info["metrics"] = metrics

    os.makedirs(RETRAIN_OUTPUT_DIR, exist_ok=True)
    info_path = os.path.join(RETRAIN_OUTPUT_DIR, "final_model_info.json")
    with open(info_path, "w") as f:
        json.dump(final_info, f, indent=2)

    print(f"[export] Final info: {info_path}")
    print("[export] SUCCESS. The 7-tag head is ready for:")
    print("  - Android app (via existing WeightSerializer)")
    print("  - Server FL coordinator (models/head_weights.npz)")
    print("  - Copy or symlink into your production deployment")

if __name__ == "__main__":
    main()