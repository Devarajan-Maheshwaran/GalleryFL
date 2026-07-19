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

    # === BULLETPROOF THRESHOLDS EXPORT (never SameFileError again) ===
    import shutil
    import json as _json

    os.makedirs(RETRAIN_DIR, exist_ok=True)
    src_thresh = os.path.join(RETRAIN_DIR, "thresholds.json")

    # Always ensure a thresholds.json exists for 7 parents
    if not os.path.exists(src_thresh):
        default_thresh = {
            "people": 0.5, "places": 0.5, "activities": 0.5,
            "objects": 0.5, "documents": 0.5, "nature": 0.5, "events": 0.5
        }
        with open(src_thresh, "w") as f:
            _json.dump(default_thresh, f, indent=2)
        print("[export] Created default thresholds.json (0.5)")

    # Safe copy: only if paths are different
    try:
        src_abs = os.path.abspath(src_thresh)
        dst_abs = os.path.abspath(THRESHOLDS_PATH)
        if src_abs != dst_abs:
            shutil.copy2(src_thresh, THRESHOLDS_PATH)
            print(f"[export] thresholds.json -> {THRESHOLDS_PATH}")
        else:
            print("[export] thresholds.json already at production path")
    except Exception as ex:
        print(f"[export] Warning copying thresholds: {ex} (safe to ignore for 7-class head)")

    print("\nExport complete. Ready for Android + Server FL.")
    print(f"Head: {HEAD_WEIGHTS_PATH}")

if __name__ == "__main__":
    main()