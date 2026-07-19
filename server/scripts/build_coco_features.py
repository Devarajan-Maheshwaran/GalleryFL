"""
Build real training features for the GalleryFL head from a COCO-format dataset
(e.g. COCO minitrain) so the head can be retrained on actual photos instead of
synthetic data.

What it does
------------
1. Loads the frozen on-device backbone (models/base_model.tflite).
2. For every COCO image, runs it through the backbone and extracts the SAME
   1024-d projection that the Android FeatureExtractor produces
   (224x224, RGB, normalized to [-1,1] via x/127.5 - 1). This keeps the
   server-trained head bit-compatible with on-device inference.
3. Reads COCO instance annotations and maps each category to the GalleryFL
   34-leaf taxonomy (see COCO_LEAF_MAP). Multi-label: an image activates
   every mapped leaf.
4. Saves data/bootstrap_seed/features.npz with keys X (N,1024) and Y (N,34).

After this, run `python scripts/retrain_head.py` -- it auto-detects
features.npz and trains the head with focal loss on the real data.

Usage
-----
    python scripts/build_coco_features.py \
        --coco_root /path/to/coco \
        --split train2017 \
        --out ../data/bootstrap_seed/features.npz

    # quick smoke test on a subset
    python scripts/build_coco_features.py --coco_root ./coco --split train2017 --limit 2000

COCO directory layout expected
------------------------------
    <coco_root>/images/<split>/<file_name>          (images, nested layout)
    <coco_root>/annotations/instances_<split>.json   (COCO instance annotations)

If your layout is flat (<coco_root>/<split>/<file_name>), the script falls back
to that automatically, or you can pass --images explicitly.

Requires: tensorflow (or tflite-runtime) and numpy. No Pillow needed; images
are decoded with tf.image.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)  # so `import taxonomy_parser` resolves
MODELS = os.path.normpath(os.path.join(SERVER, "models"))
SEED_DIR = os.path.normpath(os.path.join(SERVER, "..", "data", "bootstrap_seed"))

IMAGE_SIZE = 224
FEATURE_SIZE = 1024

# GalleryFL leaf indices (must match server/taxonomy_parser.py ordering):
#   people     0:selfie 1:group_photo 2:portrait 3:crowd
#   places     4:beach 5:mountain 6:city 7:indoor 8:rural
#   activities 9:sports 10:cooking 11:celebration 12:work 13:travel
#   objects    14:food 15:vehicle 16:gadget 17:clothing 18:art
#   documents  19:screenshot 20:receipt 21:id_card 22:handwritten 23:printed
#   nature     24:landscape 25:animal 26:plant 27:sky 28:water
#   events     29:wedding 30:birthday 31:concert 32:graduation 33:holiday
#
# "person" is handled specially by instance count (see to_leaf_indices).
COCO_LEAF_MAP = {
    "bicycle": [15], "car": [15], "motorcycle": [15], "airplane": [15],
    "bus": [15], "train": [15], "truck": [15], "boat": [15, 28],
    "traffic light": [6], "fire hydrant": [6], "stop sign": [6], "parking meter": [6],
    "bird": [25], "cat": [25], "dog": [25], "horse": [25], "sheep": [25],
    "cow": [25], "elephant": [25], "bear": [25], "zebra": [25], "giraffe": [25],
    "backpack": [17], "umbrella": [4], "handbag": [17], "tie": [17], "suitcase": [13],
    "frisbee": [9], "skis": [9], "snowboard": [9], "sports ball": [9], "kite": [9],
    "baseball bat": [9], "baseball glove": [9], "skateboard": [9], "surfboard": [9],
    "tennis racket": [9], "bottle": [14], "wine glass": [14], "cup": [14],
    "fork": [14], "knife": [14], "spoon": [14], "bowl": [14], "banana": [14],
    "apple": [14], "sandwich": [14], "orange": [14], "broccoli": [14], "carrot": [14],
    "hot dog": [14], "pizza": [14], "donut": [14], "cake": [14], "chair": [7],
    "couch": [7], "potted plant": [26], "bed": [7], "dining table": [7], "toilet": [7],
    "tv": [16], "laptop": [16], "mouse": [16], "remote": [16], "keyboard": [16],
    "cell phone": [16], "microwave": [7], "oven": [7], "toaster": [7], "sink": [7],
    "refrigerator": [7], "book": [23], "clock": [16], "vase": [18], "hair drier": [16],
}


def to_leaf_indices(cat_names):
    """Map a list of COCO category names for one image to taxonomy leaf indices.

    'person' is resolved by instance count so a single person -> portrait,
    a pair -> group_photo, a crowd (>=3) -> crowd. Other categories are mapped
    directly. Returns a set of leaf indices (may be empty).
    """
    idx = set()
    person_count = sum(1 for c in cat_names if c == "person")
    if person_count == 1:
        idx.add(2)        # portrait
    elif person_count == 2:
        idx.add(1)        # group_photo
    elif person_count >= 3:
        idx.add(3)        # crowd
    for c in cat_names:
        if c in COCO_LEAF_MAP:
            idx.update(COCO_LEAF_MAP[c])
    return idx


def load_interpreter():
    import tensorflow as tf
    tflite_model_path = os.path.normpath(os.path.join(MODELS, "base_model.tflite"))
    if not os.path.exists(tflite_model_path):
        raise SystemExit(
            f"Model file not found at: {tflite_model_path}. "
            "Make sure models/base_model.tflite is present."
        )
    interp = tf.lite.Interpreter(tflite_model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()
    proj = next(
        (o for o in out if o["shape"][-1] == FEATURE_SIZE and len(o["shape"]) == 2),
        out[0],
    )
    return interp, inp, proj


def extract_projection(interp, inp, proj, image_path):
    import tensorflow as tf
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    img = tf.expand_dims(img, 0)
    interp.set_tensor(inp["index"], img.numpy())
    interp.invoke()
    return interp.get_tensor(proj["index"])[0].astype(np.float32)


def resolve_image_path(img_dir, fname):
    """Join img_dir with fname, tolerating COCO file_names that already include
    a split subfolder (e.g. 'train2017/0001.jpg')."""
    p = os.path.normpath(os.path.join(img_dir, fname))
    if os.path.exists(p):
        return p
    base = os.path.basename(fname)
    if base != fname:
        p2 = os.path.normpath(os.path.join(img_dir, base))
        if os.path.exists(p2):
            return p2
    return p  # best guess so the skip logic reports the missing file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root",
                    default=os.path.normpath(os.path.join(SERVER, "..", "data", "coco")))
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--images", default=None, help="Override image dir")
    ap.add_argument("--annotations", default=None, help="Override annotations json path")
    ap.add_argument("--out",
                    default=os.path.normpath(os.path.join(SEED_DIR, "features.npz")))
    ap.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    args = ap.parse_args()

    from taxonomy_parser import TaxonomyParser
    parser = TaxonomyParser(os.path.normpath(os.path.join(SERVER, "taxonomy.json")))
    num_classes = parser.num_classes
    assert num_classes == 34, f"expected 34 classes, got {num_classes}"

    # Image dir resolution: prefer nested <coco_root>/images/<split>, fall back to flat.
    if args.images:
        img_dir = os.path.normpath(args.images)
    else:
        nested_img_path = os.path.normpath(os.path.join(args.coco_root, "images", args.split))
        img_dir = nested_img_path if os.path.isdir(nested_img_path) \
            else os.path.normpath(os.path.join(args.coco_root, args.split))

    ann_path = args.annotations or os.path.normpath(os.path.join(
        args.coco_root, "annotations", f"instances_{args.split}.json"))
    out_path = os.path.normpath(args.out)

    if not os.path.isdir(img_dir):
        raise SystemExit(f"Image dir not found: {img_dir}")
    if not os.path.exists(ann_path):
        raise SystemExit(f"Annotations not found: {ann_path}")

    with open(ann_path) as f:
        data = json.load(f)
    file_by_id = {im["id"]: im["file_name"] for im in data["images"]}
    name_by_cat = {c["id"]: c["name"] for c in data["categories"]}
    img_cats = defaultdict(list)
    for a in data["annotations"]:
        img_cats[a["image_id"]].append(name_by_cat[a["category_id"]])
    print(f"[coco] {len(file_by_id)} images, {len(data['annotations'])} annotations, "
          f"{len(name_by_cat)} categories")

    interp, inp, proj = load_interpreter()
    print(f"[model] backbone loaded; projection dim={FEATURE_SIZE}")

    X, Y = [], []
    per_leaf = np.zeros(num_classes, dtype=np.int64)
    used = skipped = 0
    for img_id, fname in file_by_id.items():
        if args.limit and used >= args.limit:
            break
        path = resolve_image_path(img_dir, fname)
        if not os.path.exists(path):
            skipped += 1
            continue
        cats = img_cats.get(img_id, [])
        leaf = to_leaf_indices(cats)
        if not leaf:
            skipped += 1
            continue
        try:
            feat = extract_projection(interp, inp, proj, path)
        except Exception as e:
            print(f"[warn] failed {fname}: {e}")
            skipped += 1
            continue
        X.append(feat)
        y = np.zeros(num_classes, dtype=np.float32)
        for li in leaf:
            y[li] = 1.0
            per_leaf[li] += 1
        Y.append(y)
        used += 1

    if not X:
        raise SystemExit("No usable images found (no mapped categories). "
                         "Check --coco_root/--split and that annotations map to COCO classes.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        X=np.array(X, dtype=np.float32),
                        Y=np.array(Y, dtype=np.float32))
    print(f"[done] processed {used} images, skipped {skipped}; saved -> {out_path}")
    print("[coverage] positive count per leaf (index:name:count):")
    for i in range(num_classes):
        if per_leaf[i]:
            print(f"   {i:2d}: {parser.leaf_names[i]:14s} {int(per_leaf[i])}")
    unmapped = [parser.leaf_names[i] for i in range(num_classes) if not per_leaf[i]]
    if unmapped:
        print(f"[note] leaves with no COCO coverage ({len(unmapped)}): {', '.join(unmapped)}")
        print("       (expected for documents/events/selfie; on-device FL will fill these in.)")


if __name__ == "__main__":
    main()
