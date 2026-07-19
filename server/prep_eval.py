"""TF-free server-side evaluation for the FGT classification head.

Mirrors the Android ClassificationHead forward pass (ReLU + sigmoid BCE) in
pure numpy, scores a deterministic held-out feature probe, and writes
per-class precision/recall/F1/support to the JSON files consumed by the
comparison endpoint:

    output/bootstrap_eval_report.json  -- the INITIAL (untrained) head
    output/latest_eval.json            -- the CURRENT trained head

No TensorFlow is required, so this runs in the same venv as the server and can
be invoked after every aggregation round (see fl_coordinator._run_evaluation).

The probe is deterministic. If verify_fl_loop.py exported its own held-out
probe (output/fl_probe.npz, labelled by the weights it trained on) we prefer
that so the baseline->federated F1 delta is meaningful; otherwise we fall back
to a canonical probe generated from a fixed seed. In both cases the SAME probe
is used for the initial and the current head, so the comparison is fair.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from taxonomy_parser import TaxonomyParser

_TAX = TaxonomyParser()
NUM_CLASSES = _TAX.num_classes
LEAF_NAMES = _TAX.leaf_names

PROBE_PATH = os.path.join(HERE, "output", "fl_probe.npz")
OUT_DIR = os.path.join(HERE, "output")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ w2 + b2
    return _sigmoid(z2)


def load_probe():
    """Return (X, Y) for the held-out probe, generating + caching if needed."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(PROBE_PATH):
        d = np.load(PROBE_PATH, allow_pickle=True)
        return d["X"], d["Y"]
    # Canonical deterministic probe (fixed seed) used only when no run-specific
    # probe was exported by the verification loop.
    rng = np.random.default_rng(20240719)
    W0 = rng.standard_normal((1024, NUM_CLASSES)).astype(np.float32) * 0.3
    b0 = rng.standard_normal(NUM_CLASSES).astype(np.float32) * 0.1
    X = rng.standard_normal((2000, 1024)).astype(np.float32)
    logits = X @ W0 + b0
    Y = (np.float32(_sigmoid(logits)) > 0.5).astype(np.float32)
    np.savez(PROBE_PATH, X=X, Y=Y)
    return X, Y


def load_head(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return [d["w1"], d["b1"], d["w2"], d["b2"]]


def evaluate(weights):
    X, Y = load_probe()
    w1, b1, w2, b2 = weights
    P = forward(X, w1, b1, w2, b2)
    yhat = (P > 0.5).astype(np.float32)
    per_class = {}
    f1s = []
    for c in range(NUM_CLASSES):
        yt = Y[:, c]
        yp = yhat[:, c]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        support = int(np.sum(yt == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = float(np.mean(yt == yp))
        per_class[LEAF_NAMES[c]] = {
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
            "support": support,
        }
        f1s.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": round(float(np.mean(f1s)), 4),
        "num_samples": int(X.shape[0]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    baseline = load_head(os.path.join(HERE, "models", "initial_head_weights.npz"))
    current = load_head(os.path.join(HERE, "models", "head_weights.npz")) or baseline

    base_report = evaluate(baseline)
    fed_report = evaluate(current)

    with open(os.path.join(OUT_DIR, "bootstrap_eval_report.json"), "w") as f:
        json.dump(base_report, f, indent=2)
    with open(os.path.join(OUT_DIR, "latest_eval.json"), "w") as f:
        json.dump(fed_report, f, indent=2)

    print(f"[prep_eval] baseline  macro-F1 = {base_report['macro_f1']:.4f}")
    print(f"[prep_eval] federated macro-F1 = {fed_report['macro_f1']:.4f}")
    print(f"[prep_eval] wrote {OUT_DIR}/bootstrap_eval_report.json and latest_eval.json")


if __name__ == "__main__":
    main()
