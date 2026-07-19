#!/usr/bin/env python3
"""
prepare_dataset.py
Prepare train / val / test splits for the 7-parent GalleryFL classifier.

Usage (after setting DATASET_DIR):
    python prepare_dataset.py --dataset-dir $DATASET_DIR
    python prepare_dataset.py --dataset-dir $DATASET_DIR --output manifests/

Expected input layout:
    DATASET_DIR/
        people/      *.jpg *.jpeg *.png
        places/
        activities/
        objects/
        documents/
        nature/
        events/

Outputs (in --output dir or retrain/):
    train.csv, val.csv, test.csv
    (columns: image_path,label)

The paths are absolute or relative to DATASET_DIR (configurable).
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from typing import List, Tuple

# Import local config
sys.path.insert(0, os.path.dirname(__file__))
from config import LABELS, LABEL_TO_IDX, DATASET_DIR as DEFAULT_DATASET_DIR

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def find_images_in_class(class_dir: str) -> List[str]:
    images = []
    if not os.path.isdir(class_dir):
        return images
    for fname in sorted(os.listdir(class_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTS:
            images.append(os.path.join(class_dir, fname))
    return images

def collect_dataset(dataset_dir: str) -> dict:
    """Return {label: [full_image_paths]}"""
    data = {}
    for label in LABELS:
        class_dir = os.path.join(dataset_dir, label)
        imgs = find_images_in_class(class_dir)
        data[label] = imgs
        print(f"  {label:12s}: {len(imgs)} images")
    return data

def stratified_split(data: dict, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Return three dicts: train, val, test {label: [paths]}"""
    random.seed(seed)
    train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)

    for label, paths in data.items():
        if not paths:
            continue
        paths = paths.copy()
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # ensure at least one in each if possible
        if n_train == 0 and n > 0: n_train = 1
        if n_val == 0 and n - n_train > 1: n_val = 1

        train[label] = paths[:n_train]
        val[label] = paths[n_train:n_train + n_val]
        test[label] = paths[n_train + n_val:]
    return train, val, test

def write_manifest(path: str, data_dict: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for label, paths in data_dict.items():
            for p in paths:
                writer.writerow([p, label])
    print(f"  Wrote {path} ({sum(len(v) for v in data_dict.values())} rows)")

def main():
    parser = argparse.ArgumentParser(description="Prepare 7-tag dataset splits")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR,
                        help="Root directory containing 7 class folders")
    parser.add_argument("--output", default=None,
                        help="Output directory for manifests (default: <dataset>/manifests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if args.output:
        out_dir = os.path.abspath(args.output)
    else:
        out_dir = os.path.join(dataset_dir, "manifests")

    print(f"=== Preparing 7-tag dataset splits ===")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Output manifests: {out_dir}")

    data = collect_dataset(dataset_dir)
    total = sum(len(v) for v in data.values())
    if total == 0:
        print("ERROR: No images found. Make sure DATASET_DIR has the 7 folders with images.")
        print("Expected folders:", LABELS)
        sys.exit(1)

    train, val, test = stratified_split(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    write_manifest(os.path.join(out_dir, "train.csv"), train)
    write_manifest(os.path.join(out_dir, "val.csv"), val)
    write_manifest(os.path.join(out_dir, "test.csv"), test)

    print("\n=== Split summary ===")
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        counts = {lbl: len(split_data[lbl]) for lbl in LABELS}
        print(f"{split_name:5s}: {sum(counts.values()):4d} total  |  " +
              "  ".join(f"{lbl}:{c}" for lbl, c in counts.items() if c > 0))

    print(f"\nManifests ready. Next: python train.py --dataset-dir {dataset_dir}")

if __name__ == "__main__":
    main()