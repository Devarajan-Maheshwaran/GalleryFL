#!/usr/bin/env python3
"""
export_model.py

Finalize and export the trained 7-parent head for GalleryFL.

Saves:
- models/head_weights.npz (4-layer format: w1, b1, w2, b2)
- models/model_version.txt
- output/retrain/thresholds.json (if available)
"""

import argparse
import json
import os
import numpy as np

from config import (
    HEAD_WEIGHTS_PATH, MODEL_VERSION_FILE,
    BEST_CHECKPOINT, THRESHOLDS_PATH, RETRAIN_DIR
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    args = parser.parse_args()

    print("=== Exporting 7-Parent GalleryFL Head ===")

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
        except:
            pass
    with open(MODEL_VERSION_FILE, "w") as f:
        f.write(str(version))
    print(f"Bumped model version to {version}")

    # Copy thresholds if they exist
    if os.path.exists(os.path.join(RETRAIN_DIR, "thresholds.json")):
        import shutil
        shutil.copy(os.path.join(RETRAIN_DIR, "thresholds.json"), THRESHOLDS_PATH)

    print("\nExport complete. Ready for Android + Server FL.")
    print(f"Head: {HEAD_WEIGHTS_PATH}")

if __name__ == "__main__":
    main()