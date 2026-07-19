#!/usr/bin/env python3
"""
convert_coco.py

Convert https://www.kaggle.com/datasets/banuprasadb/coco-minitrain-10k
(YOLO format) into 7-parent GalleryFL structure for single-label training.

Usage:
    python convert_coco.py \
        --coco_root /path/to/coco_minitrain_10k \
        --output /path/to/gallery_7tag \
        --max-per-class 800
"""

import argparse
import os
import shutil
import random
from collections import defaultdict

# COCO class_id (0-79) -> 7 parent
# Focused on visual separability for gallery photos
COCO_TO_7PARENT = {
    # people
    0: "people",
    # places (scenes + indoor)
    9: "places", 10: "places", 11: "places", 12: "places", 13: "places",
    56: "places", 57: "places", 58: "places", 59: "places", 60: "places", 61: "places",
    62: "places", 63: "places", 64: "places", 65: "places", 66: "places", 67: "places",
    68: "places", 69: "places", 70: "places", 71: "places", 72: "places",
    74: "places", 75: "places",
    # activities / sports / recreation
    32: "activities", 33: "activities", 34: "activities", 35: "activities",
    36: "activities", 37: "activities", 38: "activities", 39: "activities",
    # objects (vehicles, food, gadgets, household)
    1: "objects", 2: "objects", 3: "objects", 4: "objects", 5: "objects", 6: "objects", 7: "objects", 8: "objects",
    24: "objects", 25: "objects", 26: "objects", 27: "objects", 28: "objects", 29: "objects", 30: "objects", 31: "objects",
    40: "objects", 41: "objects", 42: "objects", 43: "objects", 44: "objects", 45: "objects",
    46: "objects", 47: "objects", 48: "objects", 49: "objects", 50: "objects", 51: "objects",
    52: "objects", 53: "objects", 54: "objects", 55: "objects",
    73: "objects",
    # nature (animals + outdoor natural)
    14: "nature", 15: "nature", 16: "nature", 17: "nature", 18: "nature", 19: "nature",
    20: "nature", 21: "nature", 22: "nature", 23: "nature",
}

DEFAULT = "objects"

def get_parent(label_path: str) -> str:
    if not os.path.exists(label_path):
        return None
    parents = []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    p = COCO_TO_7PARENT.get(cid, DEFAULT)
                    parents.append(p)
    except:
        return None
    if not parents:
        return None
    from collections import Counter
    return Counter(parents).most_common(1)[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="train2017,val2017")
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    splits = [s.strip() for s in args.splits.split(",")]

    parents = LABELS = ["people", "places", "activities", "objects", "documents", "nature", "events"]
    for p in parents:
        os.makedirs(os.path.join(args.output, p), exist_ok=True)

    per_class = defaultdict(int)
    total = 0

    print(f"Converting from {args.coco_root} → {args.output}")

    for split in splits:
        img_dir = os.path.join(args.coco_root, "images", split)
        lbl_dir = os.path.join(args.coco_root, "labels", split)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            print(f"Skipping {split} (missing dirs)")
            continue

        images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Processing {split}: {len(images)} images")

        for fname in images:
            base = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            img_path = os.path.join(img_dir, fname)

            parent = get_parent(lbl_path)
            if parent is None:
                continue

            if args.max_per_class > 0 and per_class[parent] >= args.max_per_class:
                continue

            dst = os.path.join(args.output, parent, fname)
            try:
                shutil.copy2(img_path, dst)
                per_class[parent] += 1
                total += 1
            except Exception as e:
                print(f"  skip {fname}: {e}")

    print("\n=== Conversion complete ===")
    for p in parents:
        print(f"  {p:12s}: {per_class[p]}")
    print(f"Total images: {total}")
    print(f"\nNow run:")
    print(f"  export DATASET_DIR={args.output}")
    print("  cd server/retrain")
    print("  python prepare_dataset.py --dataset-dir $DATASET_DIR")

if __name__ == "__main__":
    main()