#!/usr/bin/env python3
"""Create deterministic, stratified manifests for seven parent folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from config import (
    LABELS,
    LABEL_TO_IDX,
    MIN_SAMPLES_PER_CLASS,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from pipeline import PipelineError, atomic_write_json

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# COCO is an object-detection dataset, not a seven-parent gallery dataset. This
# exhaustive, non-overlapping proxy map is therefore explicit rather than
# pretending the source labels are native GalleryFL labels. The weights stop
# common person/object boxes from swallowing rare document/event proxies.
COCO_PARENT_CLASSES = {
    "people": {0, 24, 25, 26, 28},
    "places": {9, 10, 11, 12, 13, 56, 57, 59, 60, 61, 68, 69, 70, 71, 72},
    "activities": set(range(29, 39)),
    "objects": set(range(1, 9)) | {39, 42, 43, 44, 45} | set(range(46, 55)) | {62, 64, 65, 74, 75, 76, 77, 78, 79},
    "documents": {63, 66, 67, 73},
    "nature": set(range(14, 24)) | {58},
    "events": {27, 40, 41, 55},
}
COCO_PARENT_WEIGHTS = {
    "people": 0.5,
    "places": 1.2,
    "activities": 2.0,
    "objects": 0.5,
    "documents": 3.0,
    "nature": 1.5,
    "events": 3.0,
}
_COCO_CLASS_TO_PARENT = {
    class_id: parent
    for parent, class_ids in COCO_PARENT_CLASSES.items()
    for class_id in class_ids
}
if set(_COCO_CLASS_TO_PARENT) != set(range(80)) or sum(map(len, COCO_PARENT_CLASSES.values())) != 80:
    raise RuntimeError("COCO proxy map must cover each of the 80 classes exactly once")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(dataset_dir: Path) -> dict[str, list[Path]]:
    by_label: dict[str, list[Path]] = {}
    digest_owner: dict[str, tuple[Path, str]] = {}
    for label in LABELS:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            raise PipelineError(f"Required class directory is missing: {class_dir}")

        images: list[Path] = []
        duplicate_count = 0
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            resolved = path.resolve()
            digest = file_sha256(resolved)
            previous = digest_owner.get(digest)
            if previous is not None:
                previous_path, previous_label = previous
                if previous_label != label:
                    raise PipelineError(
                        "The same image content has conflicting labels: "
                        f"{previous_path} ({previous_label}) and {resolved} ({label})"
                    )
                duplicate_count += 1
                continue
            digest_owner[digest] = (resolved, label)
            images.append(resolved)

        if len(images) < MIN_SAMPLES_PER_CLASS:
            raise PipelineError(
                f"{label} has {len(images)} unique images; at least "
                f"{MIN_SAMPLES_PER_CLASS} are required"
            )
        by_label[label] = images
        duplicate_note = f" ({duplicate_count} exact duplicates skipped)" if duplicate_count else ""
        print(f"{label:12s}: {len(images):6d}{duplicate_note}")
    return by_label


def allocation(size: int) -> tuple[int, int, int]:
    ratios = np.asarray([TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT], dtype=np.float64)
    raw = ratios * size
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: size - int(counts.sum())]:
        counts[index] += 1

    # Every class must be represented in every split.
    for index in (1, 2, 0):
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    if np.any(counts <= 0) or counts.sum() != size:
        raise PipelineError(f"Could not split class with {size} images into non-empty splits")
    return int(counts[0]), int(counts[1]), int(counts[2])


def stratified_split(
    by_label: dict[str, list[Path]], seed: int
) -> dict[str, list[tuple[Path, str]]]:
    rng = np.random.default_rng(seed)
    result: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    for label in LABELS:
        paths = np.asarray(by_label[label], dtype=object)
        paths = paths[rng.permutation(len(paths))]
        train_count, val_count, _ = allocation(len(paths))
        boundaries = (train_count, train_count + val_count)
        partitions = np.split(paths, boundaries)
        for split_name, partition in zip(("train", "val", "test"), partitions):
            result[split_name].extend((Path(path), label) for path in partition.tolist())

    # Shuffle each manifest so batches are not arranged by class.
    for split_name, rows in result.items():
        order = rng.permutation(len(rows))
        result[split_name] = [rows[index] for index in order]
    return result


def coco_parent_label(label_path: Path) -> str:
    scores = {label: 0.0 for label in LABELS}
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PipelineError(f"Could not read YOLO label {label_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise PipelineError(f"{label_path}:{line_number}: expected five YOLO fields")
        try:
            class_id = int(fields[0])
            width = float(fields[3])
            height = float(fields[4])
        except ValueError as exc:
            raise PipelineError(f"{label_path}:{line_number}: invalid YOLO row") from exc
        if class_id not in _COCO_CLASS_TO_PARENT:
            raise PipelineError(f"{label_path}:{line_number}: COCO class {class_id} is outside 0..79")
        parent = _COCO_CLASS_TO_PARENT[class_id]
        # sqrt(area) rewards salient boxes without allowing one huge box to
        # completely suppress smaller but semantically specific objects.
        scores[parent] += max(width * height, 1e-8) ** 0.5
    if not any(scores.values()):
        raise PipelineError(f"YOLO label contains no objects: {label_path}")
    return max(LABELS, key=lambda label: scores[label] * COCO_PARENT_WEIGHTS[label])


def collect_coco_split(dataset_dir: Path, split_name: str) -> list[tuple[Path, str]]:
    image_dir = dataset_dir / "images" / split_name
    label_dir = dataset_dir / "labels" / split_name
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise PipelineError(f"Missing COCO YOLO split directories for {split_name}")
    rows: list[tuple[Path, str]] = []
    missing_labels = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            continue
        rows.append((image_path.resolve(), coco_parent_label(label_path)))
    if missing_labels:
        print(f"{split_name}: skipped {missing_labels} images without YOLO labels")
    if not rows:
        raise PipelineError(f"No labelled images found in COCO split {split_name}")
    return rows


def split_coco_yolo(dataset_dir: Path, seed: int) -> dict[str, list[tuple[Path, str]]]:
    """Use COCO train2017 for train/val and untouched val2017 for test."""
    source_train = collect_coco_split(dataset_dir, "train2017")
    source_test = collect_coco_split(dataset_dir, "val2017")
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[tuple[Path, str]]] = {label: [] for label in LABELS}
    for row in source_train:
        grouped[row[1]].append(row)

    train_rows: list[tuple[Path, str]] = []
    val_rows: list[tuple[Path, str]] = []
    for label in LABELS:
        rows = grouped[label]
        if len(rows) < MIN_SAMPLES_PER_CLASS:
            raise PipelineError(f"COCO train proxy has only {len(rows)} samples for {label}")
        order = rng.permutation(len(rows))
        val_count = max(1, int(round(len(rows) * 0.15)))
        val_rows.extend(rows[index] for index in order[:val_count])
        train_rows.extend(rows[index] for index in order[val_count:])

    for rows in (train_rows, val_rows):
        rng.shuffle(rows)
    test_counts = Counter(label for _, label in source_test)
    missing_test = [label for label in LABELS if test_counts[label] == 0]
    if missing_test:
        raise PipelineError(f"COCO test proxy has no samples for: {', '.join(missing_test)}")
    return {"train": train_rows, "val": val_rows, "test": source_test}


def portable_path(path: Path, dataset_dir: Path) -> str:
    try:
        return path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return str(path)


def write_manifest(path: Path, rows: list[tuple[Path, str]], dataset_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "class_index"))
        writer.writeheader()
        for image_path, label in rows:
            writer.writerow(
                {
                    "path": portable_path(image_path, dataset_dir),
                    "label": label,
                    "class_index": LABEL_TO_IDX[label],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--source-format",
        choices=("auto", "parent-folders", "coco-yolo"),
        default="auto",
        help="auto detects seven parent folders or COCO images/labels/{train2017,val2017}",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise PipelineError(f"Dataset directory not found: {dataset_dir}")

    has_parent_folders = all((dataset_dir / label).is_dir() for label in LABELS)
    has_coco_yolo = all(
        (dataset_dir / kind / split).is_dir()
        for kind in ("images", "labels")
        for split in ("train2017", "val2017")
    )
    source_format = args.source_format
    if source_format == "auto":
        if has_parent_folders:
            source_format = "parent-folders"
        elif has_coco_yolo:
            source_format = "coco-yolo"
        else:
            raise PipelineError(
                "Could not detect dataset layout. Expected seven parent folders or "
                "COCO YOLO images/labels/{train2017,val2017}."
            )

    if source_format == "parent-folders":
        print("Collecting native seven-parent single-label folders...")
        splits = stratified_split(collect_images(dataset_dir), args.seed)
    else:
        print("Converting COCO YOLO object labels to explicit seven-parent proxy labels...")
        splits = split_coco_yolo(dataset_dir, args.seed)
    manifest_dir = dataset_dir / "manifests"

    summary: dict[str, dict] = {}
    for split_name in ("train", "val", "test"):
        manifest_path = manifest_dir / f"{split_name}.csv"
        write_manifest(manifest_path, splits[split_name], dataset_dir)
        counts = Counter(label for _, label in splits[split_name])
        summary[split_name] = {
            "total": len(splits[split_name]),
            "per_class": {label: counts[label] for label in LABELS},
        }
        print(f"{split_name:5s}: {len(splits[split_name]):6d} -> {manifest_path}")

    metadata = {
        "schema_version": 1,
        "task": "single_label_multiclass",
        "labels": list(LABELS),
        "source_format": source_format,
        "seed": args.seed,
        "ratios": (
            {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT}
            if source_format == "parent-folders"
            else {"train_source": "85% of train2017", "val_source": "15% of train2017", "test_source": "val2017"}
        ),
        "coco_proxy_map": (
            {
                label: {
                    "class_ids": sorted(COCO_PARENT_CLASSES[label]),
                    "salience_weight": COCO_PARENT_WEIGHTS[label],
                }
                for label in LABELS
            }
            if source_format == "coco-yolo"
            else None
        ),
        "paths_are_relative_to": str(dataset_dir),
        "splits": summary,
    }
    atomic_write_json(manifest_dir / "split_summary.json", metadata)
    print("Dataset manifests are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
