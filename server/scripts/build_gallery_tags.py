"""
Build real training features for the GalleryFL head DIRECTLY from a YOLO-Darknet
COCO-minitrain layout (images/<split> + labels/<split>/*.txt). No COCO JSON needed.

This is the YOLO-native twin of build_coco_features.py: same frozen backbone,
same 224x224 RGB -> [-1,1] normalization (x/127.5 - 1), so the head stays
bit-compatible with on-device inference. It maps each image's YOLO class ids to
the 34-leaf GalleryFL taxonomy (multi-label) and writes:

    data/bootstrap_seed/features.npz   (keys X: Nx1024, Y: Nx34)

which retrain_head.py auto-detects and trains on.

Layout expected:
    <coco_root>/images/<split>/*.jpg
    <coco_root>/labels/<split>/*.txt      # <class_id> <x_c> <y_c> <w> <h>  (normalized)

Usage:
    python scripts/build_gallery_tags.py --coco_root /path/to/coco_minitrain_10k --split train2017
    python scripts/build_gallery_tags.py --coco_root /path/to/coco_minitrain_10k --split train2017 --limit 2000

Requires: tensorflow (or tflite-runtime) and numpy. base_model.tflite must exist.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)  # so `import taxonomy_parser` resolves

MODELS = os.path.normpath(os.path.join(SERVER, "models"))
# Must match retrain_head.py's load_real() path (server/../data/bootstrap_seed).
SEED_DIR = os.path.normpath(os.path.join(SERVER, "..", "data", "bootstrap_seed"))

IMAGE_SIZE = 224
FEATURE_SIZE = 1024
NUM_CLASSES = 34

# YOLO/COCO class index (0-79) -> list of GalleryFL leaf indices.
# Mirrors server/scripts/build_coco_features.py COCO_LEAF_MAP and must match
# taxonomy_parser.py's 34-leaf ordering:
#   people     0:selfie 1:group_photo 2:portrait 3:crowd
#   places     4:beach 5:mountain 6:city 7:indoor 8:rural
#   activities 9:sports 10:cooking 11:celebration 12:work 13:travel
#   objects    14:food 15:vehicle 16:gadget 17:clothing 18:art
#   documents  19:screenshot 20:receipt 21:id_card 22:handwritten 23:printed
#   nature     24:landscape 25:animal 26:plant 27:sky 28:water
#   events     29:wedding 30:birthday 31:concert 32:graduation 33:holiday
# 'person' (0) is resolved by instance count in the loop, so it is intentionally
# absent here. Classes with no gallery equivalent (bench, scissors, teddy bear,
# toothbrush) are also omitted.
YOLO_TO_SMART_TAGS = {
    1: [15], 2: [15], 3: [15], 4: [15], 5: [15], 6: [15], 7: [15], 8: [15, 28],
    9: [6], 10: [6], 11: [6], 12: [6],
    14: [25], 15: [25], 16: [25], 17: [25], 18: [25], 19: [25], 20: [25],
    21: [25], 22: [25], 23: [25],
    24: [17], 25: [4], 26: [17], 27: [17], 28: [13],
    29: [9], 30: [9], 31: [9], 32: [9], 33: [9], 34: [9], 35: [9], 36: [9],
    37: [9], 38: [9],
    39: [14], 40: [14], 41: [14], 42: [14], 43: [14], 44: [14], 45: [14],
    46: [14], 47: [14], 48: [14], 49: [14], 50: [14], 51: [14], 52: [14],
    53: [14], 54: [14], 55: [14],
    56: [7], 57: [7], 58: [26], 59: [7], 60: [7], 61: [7],
    62: [16], 63: [16], 64: [16], 65: [16], 66: [16], 67: [16],
    68: [7], 69: [7], 70: [7], 71: [7], 72: [7],
    73: [23], 74: [16], 75: [18],
    78: [16],
}


def load_tflite_backbone():
    import tensorflow as tf
    model_path = os.path.normpath(os.path.join(MODELS, "base_model.tflite"))
    if not os.path.exists(model_path):
        raise SystemExit(
            f"[error] {model_path} not found. The frozen backbone is required to "
            "extract real features; refusing to emit random/synthetic features."
        )
    interp = tf.lite.Interpreter(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()
    proj = next(
        (o for o in out if o["shape"][-1] == FEATURE_SIZE and len(o["shape"]) == 2),
        out[0],
    )
    return interp, inp, proj


def process_image_pixels(interp, inp, proj, path):
    import tensorflow as tf
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    img = tf.expand_dims(img, 0)
    interp.set_tensor(inp["index"], img.numpy())
    interp.invoke()
    return interp.get_tensor(proj["index"])[0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", default="../data/coco_minitrain_10k")
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--out", default=os.path.join(SEED_DIR, "features.npz"))
    ap.add_argument("--limit", type=int, default=0,
                    help="Max images to process (0 = all)")
    args = ap.parse_args()

    img_dir = os.path.normpath(os.path.join(args.coco_root, "images", args.split))
    label_dir = os.path.normpath(os.path.join(args.coco_root, "labels", args.split))
    out_path = os.path.normpath(args.out)

    if not os.path.isdir(img_dir):
        raise SystemExit(f"[error] image folder missing: {img_dir}")
    if not os.path.isdir(label_dir):
        raise SystemExit(f"[error] label folder missing: {label_dir}")

    image_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not image_files:
        raise SystemExit(f"[error] no images found in {img_dir}")

    interp, inp, proj = load_tflite_backbone()
    print(f"[model] backbone loaded; projection dim={FEATURE_SIZE}")

    X, Y = [], []
    per_leaf = np.zeros(NUM_CLASSES, dtype=np.int64)
    used = skipped = 0
    total = min(args.limit, len(image_files)) if args.limit else len(image_files)
    print(f"[info] encoding up to {total} images from {img_dir}")

    for fname in image_files:
        if args.limit and used >= args.limit:
            break
        img_path = os.path.join(img_dir, fname)
        base_name = os.path.splitext(fname)[0]
        lbl_path = os.path.join(label_dir, f"{base_name}.txt")

        smart_tags = set()
        person_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if class_id == 0:
                        person_count += 1
                    elif class_id in YOLO_TO_SMART_TAGS:
                        smart_tags.update(YOLO_TO_SMART_TAGS[class_id])

        # Resolve 'person' by instance count (smart-album logic).
        if person_count == 1:
            smart_tags.add(2)      # portrait
        elif person_count == 2:
            smart_tags.add(1)      # group_photo
        elif person_count >= 3:
            smart_tags.add(3)      # crowd

        if not smart_tags:
            skipped += 1
            continue

        try:
            feat = process_image_pixels(interp, inp, proj, img_path)
        except Exception as e:
            print(f"[warn] failed {fname}: {e}")
            skipped += 1
            continue

        X.append(feat)
        y = np.zeros(NUM_CLASSES, dtype=np.float32)
        for tag in smart_tags:
            y[tag] = 1.0
            per_leaf[tag] += 1
        Y.append(y)
        used += 1
        if used % 250 == 0:
            print(f"    -> encoded {used} images")

    if not X:
        raise SystemExit("[error] no images mapped to any gallery tag. "
                         "Check --coco_root/--split and that labels use COCO class ids.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        X=np.array(X, dtype=np.float32),
                        Y=np.array(Y, dtype=np.float32))
    print(f"[done] processed {used} images, skipped {skipped}; saved -> {out_path}")

    # Coverage report (uses taxonomy names when available).
    try:
        from taxonomy_parser import TaxonomyParser
        leaf_names = TaxonomyParser(os.path.join(SERVER, "taxonomy.json")).leaf_names
    except Exception:
        leaf_names = [str(i) for i in range(NUM_CLASSES)]
    print("[coverage] positive count per leaf (index:name:count):")
    for i in range(NUM_CLASSES):
        if per_leaf[i]:
            print(f"   {i:2d}: {leaf_names[i]:14s} {int(per_leaf[i])}")
    unmapped = [leaf_names[i] for i in range(NUM_CLASSES) if not per_leaf[i]]
    if unmapped:
        print(f"[note] leaves with no COCO coverage ({len(unmapped)}): {', '.join(unmapped)}")
        print("       (expected for documents/events/selfie; on-device FL will fill these in.)")


if __name__ == "__main__":
    main()
