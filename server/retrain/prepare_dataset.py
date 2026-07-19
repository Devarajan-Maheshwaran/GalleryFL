#!/usr/bin/env python3
"""
prepare_dataset.py

Create stratified train/val/test splits for the 7-parent GalleryFL dataset.

Usage:
    python prepare_dataset.py --dataset-dir $DATASET_DIR
"""

import argparse
import csv
import os
import random
from collections import defaultdict

from config import LABELS, LABEL_TO_IDX, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def collect_images(dataset_dir):
    data = {}
    for label in LABELS:
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            print(f"WARNING: Missing folder {label}")
            data[label] = []
            continue
        imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
        data[label] = imgs
        print(f"  {label:12s}: {len(imgs)} images")
    return data

def stratified_split(data, train_r=TRAIN_SPLIT, val_r=VAL_SPLIT, test_r=TEST_SPLIT, seed=42):
    random.seed(seed)
    splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    for label, paths in data.items():
        if not paths:
            continue
        paths = paths.copy()
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        splits["train"][label] = paths[:n_train]
        splits["val"][label] = paths[n_train:n_train + n_val]
        splits["test"][label] = paths[n_train + n_val:]
    return splits

def write_manifest(path, split_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for label, paths in split_dict.items():
            for p in paths:
                w.writerow([p, label])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Preparing 7-tag dataset splits ===")
    data = collect_images(args.dataset_dir)

    splits = stratified_split(data, seed=args.seed)

    out_dir = os.path.join(args.dataset_dir, "manifests")
    write_manifest(os.path.join(out_dir, "train.csv"), splits["train"])
    write_manifest(os.path.join(out_dir, "val.csv"), splits["val"])
    write_manifest(os.path.join(out_dir, "test.csv"), splits["test"])

    print("\n=== Split summary ===")
    for name in ["train", "val", "test"]:
        total = sum(len(v) for v in splits[name].values())
        print(f"{name:5s}: {total:4d} total")
        for lbl in LABELS:
            print(f"  {lbl:12s}: {len(splits[name][lbl])}")

    print(f"\nManifests written to {out_dir}")
    print("Next: python train.py --dataset-dir", args.dataset_dir)

if __name__ == "__main__":
    main()