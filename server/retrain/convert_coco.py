#!/usr/bin/env python3
"""
convert_coco.py

Convert the exact Kaggle "coco-minitrain-10k" dataset 
(https://www.kaggle.com/datasets/banuprasadb/coco-minitrain-10k)
into the clean 7-parent GalleryFL folder structure required by the new retrainer.

Dataset layout (from Kaggle):
    coco_minitrain_10k/
        images/
            train2017/   (jpg files)
            val2017/
        labels/
            train2017/   (*.txt in YOLO format)
            val2017/
        train2017.txt
        val2017.txt
        ...

This script reads the YOLO .txt labels, maps COCO class IDs → one of 7 parents,
and copies (or symlinks) images into:

    output_dir/
        people/
        places/
        activities/
        objects/
        documents/
        nature/
        events/

Usage (recommended first run):
    python convert_coco.py \
        --coco_root /path/to/coco_minitrain_10k \
        --output ~/data/gallery_7tag \
        --max-per-class 300

Then:
    export DATASET_DIR=~/data/gallery_7tag
    cd server/retrain
    python prepare_dataset.py --dataset-dir $DATASET_DIR
    python train.py --dataset-dir $DATASET_DIR
"""

import argparse
import os
import shutil
import random
from collections import defaultdict, Counter
from pathlib import Path

# =============================================================================
# COCO CLASS ID (0-79) → 7 PARENT CATEGORY
# Based on standard COCO 2017 class names.
# =============================================================================
COCO_ID_TO_7PARENT = {
    # === PEOPLE (class 0) ===
    0: "people",                    # person

    # === PLACES (indoor + outdoor scenes) ===
    9: "places", 10: "places", 11: "places", 12: "places",   # traffic light, fire hydrant, stop sign, parking meter
    13: "places",                                           # bench
    56: "places", 57: "places", 58: "places",               # chair, couch, potted plant
    59: "places", 60: "places", 61: "places",               # bed, dining table, toilet
    62: "places", 63: "places", 64: "places", 65: "places", # tv, laptop, mouse, remote
    66: "places", 67: "places",                             # keyboard, cell phone
    68: "places", 69: "places", 70: "places", 71: "places", # microwave, oven, toaster, sink
    72: "places",                                           # refrigerator
    74: "places", 75: "places",                             # clock, vase

    # === ACTIVITIES (sports, recreation) ===
    32: "activities", 33: "activities", 34: "activities", 35: "activities",  # sports ball, kite, baseball bat, baseball glove
    36: "activities", 37: "activities", 38: "activities",                   # skateboard, surfboard, tennis racket
    39: "activities",                                                   # frisbee (sometimes listed as 39)

    # === OBJECTS (vehicles, food, gadgets, household) ===
    1: "objects", 2: "objects", 3: "objects", 4: "objects", 5: "objects", 6: "objects", 7: "objects", 8: "objects",  # vehicles
    24: "objects", 25: "objects", 26: "objects", 27: "objects", 28: "objects", 29: "objects", 30: "objects", 31: "objects",  # food
    40: "objects", 41: "objects", 42: "objects", 43: "objects", 44: "objects", 45: "objects", 46: "objects", 47: "objects",
    48: "objects", 49: "objects", 50: "objects", 51: "objects", 52: "objects", 53: "objects", 54: "objects", 55: "objects",
    73: "objects",  # book → objects (or documents)

    # === NATURE (animals + natural elements) ===
    14: "nature", 15: "nature", 16: "nature", 17: "nature", 18: "nature", 19: "nature", 20: "nature", 21: "nature", 22: "nature", 23: "nature",  # animals
    # potted plant (58) already mapped to places, but we can leave it

    # === DOCUMENTS (very few in COCO) ===
    # We map almost nothing here because COCO has almost no document-like classes.
    # "book" (73) is mapped to objects above. You can move some manually later if needed.

    # === EVENTS ===
    # Extremely rare in COCO. We leave this category mostly empty from COCO.
    # Real events will come from user photos during FL.
}

DEFAULT_PARENT = "objects"

def get_parent_for_image(label_path: str) -> str:
    """Read YOLO label file and return the dominant 7-parent category."""
    if not os.path.exists(label_path):
        return None

    parents = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    parent = COCO_ID_TO_7PARENT.get(class_id, DEFAULT_PARENT)
                    parents.append(parent)
                except ValueError:
                    continue
    except Exception:
        return None

    if not parents:
        return None

    # Return the most frequent parent for this image
    return Counter(parents).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle coco-minitrain-10k (YOLO format) to 7-parent GalleryFL structure"
    )
    parser.add_argument("--coco_root", required=True,
                        help="Path to the extracted coco_minitrain_10k folder from Kaggle")
    parser.add_argument("--output", required=True,
                        help="Output directory that will contain the 7 parent folders")
    parser.add_argument("--splits", default="train2017,val2017",
                        help="Comma-separated splits to process (default: train2017,val2017)")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Maximum images per parent category (0 = unlimited)")
    parser.add_argument("--use-symlinks", action="store_true",
                        help="Use symlinks instead of copying (much faster, needs same filesystem)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    splits = [s.strip() for s in args.splits.split(",")]

    # Create the 7 target folders
    parents = ["people", "places", "activities", "objects", "documents", "nature", "events"]
    for parent in parents:
        os.makedirs(os.path.join(args.output, parent), exist_ok=True)

    total_copied = 0
    per_class_count = defaultdict(int)

    print(f"Converting COCO minitrain from: {args.coco_root}")
    print(f"Output 7-tag dataset:          {args.output}")
    print(f"Splits: {splits}")
    print(f"Max per class: {args.max_per_class or 'unlimited'}")
    print("-" * 60)

    for split in splits:
        img_dir = os.path.join(args.coco_root, "images", split)
        lbl_dir = os.path.join(args.coco_root, "labels", split)

        if not os.path.isdir(img_dir):
            print(f"[WARN] Images dir not found: {img_dir} — skipping {split}")
            continue
        if not os.path.isdir(lbl_dir):
            print(f"[WARN] Labels dir not found: {lbl_dir} — skipping {split}")
            continue

        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\nProcessing {split}: {len(images)} images")

        for fname in images:
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, base + ".txt")

            parent = get_parent_for_image(lbl_path)
            if parent is None:
                continue

            # Respect per-class limit
            if args.max_per_class > 0 and per_class_count[parent] >= args.max_per_class:
                continue

            dst_dir = os.path.join(args.output, parent)
            dst_path = os.path.join(dst_dir, fname)

            try:
                if args.use_symlinks:
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.symlink(os.path.abspath(img_path), dst_path)
                else:
                    shutil.copy2(img_path, dst_path)

                per_class_count[parent] += 1
                total_copied += 1

                if total_copied % 1000 == 0:
                    print(f"  ... {total_copied} images processed")

            except Exception as e:
                print(f"  [ERROR] failed to process {fname}: {e}")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total images copied/linked: {total_copied}")
    print("\nPer-category counts:")
    for p in parents:
        print(f"  {p:12s} : {per_class_count[p]:5d}")

    print(f"\nYour 7-tag dataset is ready at:\n  {args.output}")
    print("\nNext steps:")
    print(f"  export DATASET_DIR={args.output}")
    print("  cd server/retrain")
    print("  python prepare_dataset.py --dataset-dir $DATASET_DIR")
    print("  python train.py --dataset-dir $DATASET_DIR --epochs 20")
    print("  python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds")
    print("  python export_model.py")


if __name__ == "__main__":
    main()
