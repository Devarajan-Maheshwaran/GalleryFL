"""Reset the GalleryFL classification head and (optionally) retrain it properly.

Why this exists
---------------
The coordinator's "model" is a small classification *head* on top of a frozen
ImageNet MobileNetV3 backbone (`models/base_model.tflite`). The head is what
federated learning updates. If the head ever collapses to a single dominant
class (e.g. the reported "only people" behaviour) or simply never learned
because the DP/gradient path destroyed the signal, the correct recovery is to
**delete the learned head and start from a balanced, class-neutral init**, then
retrain.

What this script does (no TensorFlow required for the reset path)
-----------------------------------------------------------------
1. Wipes the learned head state:
     models/head_weights.npz, models/model_version.txt
   and all evaluation / demand / metrics artefacts in output/:
     metrics_history.json, latest_eval.json, bootstrap_eval_report.json,
     fl_probe.npz, tag_demand.json
2. Regenerates `models/initial_head_weights.npz` with a *balanced* init:
     - Glorot-uniform for w1 (1024x256) and w2 (256xNUM_CLASSES)
     - ZERO biases (b1, b2) so the untrained head scores every class ~0.5
       uniformly instead of favouring any group.
   The frozen backbone (`base_model.tflite`) is intentionally NOT touched.
3. If `data/bootstrap_seed/` (labels_train.csv + labels_val.csv + images)
   exists AND TensorFlow is installed, runs a *class-balanced* bootstrap train
   to produce a genuinely trained head. Otherwise it leaves the balanced init
   in place and prints guidance — real photo quality comes from on-device FL
   with real user data, which this sandbox cannot synthesise.

Usage
-----
    python scripts/reset_and_retrain.py            # reset + balanced init
    python scripts/reset_and_retrain.py --train    # also bootstrap-train if data present
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)

from taxonomy_parser import TaxonomyParser
import numpy as np

MODELS = os.path.join(SERVER, "models")
OUTPUT = os.path.join(SERVER, "output")


def _glorot(shape):
    """Glorot/Xavier uniform init — keeps the head class-neutral at start."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)


def balanced_init(num_classes):
    """w1 (1024x256) Glorot, b1 zeros, w2 (256xN) Glorot, b2 zeros."""
    w1 = _glorot((1024, 256))
    b1 = np.zeros((256,), dtype=np.float32)
    w2 = _glorot((256, num_classes))
    b2 = np.zeros((num_classes,), dtype=np.float32)
    return w1, b1, w2, b2


def wipe_state():
    targets = [
        os.path.join(MODELS, "head_weights.npz"),
        os.path.join(MODELS, "model_version.txt"),
        os.path.join(OUTPUT, "metrics_history.json"),
        os.path.join(OUTPUT, "latest_eval.json"),
        os.path.join(OUTPUT, "bootstrap_eval_report.json"),
        os.path.join(OUTPUT, "fl_probe.npz"),
        os.path.join(OUTPUT, "tag_demand.json"),
    ]
    for t in targets:
        if os.path.exists(t):
            os.remove(t)
            print(f"  removed {os.path.relpath(t, SERVER)}")


def regenerate_initial(num_classes):
    w1, b1, w2, b2 = balanced_init(num_classes)
    path = os.path.join(MODELS, "initial_head_weights.npz")
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"  wrote balanced init -> {os.path.relpath(path, SERVER)} "
          f"(w1 {w1.shape}, w2 {w2.shape}, b2 all-zero={bool((b2 == 0).all())})")


def bootstrap_train():
    seed = os.path.join(SERVER, "..", "data", "bootstrap_seed")
    if not os.path.isdir(seed):
        print("  no data/bootstrap_seed found — skipping supervised bootstrap.")
        print("  Real training happens via on-device FL with user photos.")
        return
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        print(f"  TensorFlow not available ({e}) — skipping supervised bootstrap.")
        return
    print("  data + TensorFlow present — running class-balanced bootstrap train...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("prep_model", os.path.join(SERVER, "prep_model.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    sys.argv = ["prep_model", "bootstrap_train", "--epochs", "12"]
    prep.cmd_bootstrap_train(prep.argparse.Namespace(
        epochs=12, batch_size=16, lr=1e-3))
    # Promote the freshly trained head to the active snapshot.
    trained = os.path.join(MODELS, "initial_head_weights.npz")
    active = os.path.join(MODELS, "head_weights.npz")
    if os.path.exists(trained):
        shutil.copyfile(trained, active)
        with open(os.path.join(MODELS, "model_version.txt"), "w") as f:
            f.write("1")
        print(f"  promoted trained head -> {os.path.relpath(active, SERVER)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="also bootstrap-train if data present")
    args = ap.parse_args()

    parser = TaxonomyParser(os.path.join(SERVER, "taxonomy.json"))
    num_classes = parser.num_classes
    print(f"=== GalleryFL model reset (NUM_CLASSES={num_classes}) ===")
    print("[1] wiping learned head + eval/demand/metrics artefacts")
    wipe_state()
    print("[2] regenerating balanced initial head")
    regenerate_initial(num_classes)
    if args.train:
        print("[3] bootstrap train (if data + TF available)")
        bootstrap_train()
    print("=== done. Start the server; FL rounds will now train from a clean, balanced head. ===")


if __name__ == "__main__":
    main()
