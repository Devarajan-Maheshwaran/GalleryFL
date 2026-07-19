#!/usr/bin/env python3
"""
export_model.py - Export the trained 7-parent head for GalleryFL.

Produces:
- models/head_weights.npz   (exact 4-layer format: w1, b1, w2, b2)
- models/model_version.txt  (bumped)
- output/retrain/thresholds.json (defaults for single-label)
"""

import argparse
import json
import os
import numpy as np

from config import (
    HEAD_WEIGHTS_PATH, MODEL_VERSION_FILE,
    BEST_CHECKPOINT, THRESHOLDS_PATH, RETRAIN_DIR, LABELS
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    args = parser.parse_args()

    print("=== Exporting 7-Parent GalleryFL Head (single-label) ===")

    if not os.path.exists(args.checkpoint):
        print("ERROR: No checkpoint found. Train first.")
        return

    d = np.load(args.checkpoint)
    w1, b1, w2, b2 = d["w1"], d["b1"], d["w2"], d["b2"]

    os.makedirs(os.path.dirname(HEAD_WEIGHTS_PATH), exist_ok=True)
    np.savez(HEAD_WEIGHTS_PATH, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"Saved production head → {HEAD_WEIGHTS_PATH}")

    # Bump version
    version = 1
    if os.path.exists(MODEL_VERSION_FILE):
        try:
            version = int(open(MODEL_VERSION_FILE).read().strip()) + 1
        except Exception:
            pass
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write(str(version))
    print(f"Bumped model version to {version}")

    # Ensure thresholds.json exists (defaults are fine for argmax primary use)
    os.makedirs(RETRAIN_DIR, exist_ok=True)
    if not os.path.exists(THRESHOLDS_PATH):
        thresh = {lbl: 0.5 for lbl in LABELS}
        with open(THRESHOLDS_PATH, "w") as f:
            json.dump(thresh, f, indent=2)
        print(f"Created default {THRESHOLDS_PATH}")

    print("\nExport complete.")
    print(f"Head: {HEAD_WEIGHTS_PATH}")
    print(f"Version: {MODEL_VERSION_FILE}")

if __name__ == "__main__":
    main()