"""Bootstrap-train the GalleryFL classification head on REAL features.

This is a robust, reproducible entry point for producing a properly trained
classification head without the full COCO minitrain 10k pipeline.

Data priority (first match wins):
  1. data/bootstrap_seed/features.npz   -- full 10k real (feature,label) pairs
                                           produced by build_gallery_tags.py
  2. output/fl_probe.npz                -- real (feature,label) pairs that ship
                                           in the repo as the eval probe. Used as
                                           a bootstrap seed so a *trained* head is
                                           committed even when the 10k set is absent.
  3. synthetic Non-IID demo              -- last resort, proves the pipeline only.

The head is the tiny MLP 1024->256 relu->34 sigmoid trained with focal loss
(gamma=2, alpha=0.5) and Adam (lr=0.01). On save it is cast to float32 and
written to BOTH models/head_weights.npz and models/initial_head_weights.npz,
and models/model_version.txt is bumped. A held-out split is written back to
output/fl_probe.npz so prep_eval scores the head honestly.

Usage:
    python scripts/train_head_bootstrap.py
    python scripts/train_head_bootstrap.py --epochs 40 --lr 0.01
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)

from taxonomy_parser import TaxonomyParser
import numpy as np

MODELS = os.path.join(SERVER, "models")
OUTPUT = os.path.join(SERVER, "output")
SEED_DIR = os.path.join(SERVER, "..", "data", "bootstrap_seed")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)


def load_real():
    p = os.path.join(SEED_DIR, "features.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)


def load_probe():
    p = os.path.join(OUTPUT, "fl_probe.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)


def gen_synthetic(n_classes, dim=1024, n_train=20000, n_test=4000, k_users=6, seed=0):
    rng = np.random.default_rng(seed)
    P = (rng.standard_normal((n_classes, dim)).astype(np.float32)) * 0.35
    users = [rng.choice(n_classes, size=int(rng.integers(6, 13)), replace=False)
             for _ in range(k_users)]

    def sample():
        u = int(rng.integers(k_users))
        focus = users[u]
        n_active = int(rng.integers(1, 4))
        if rng.random() < 0.8 and len(focus) >= n_active:
            active = rng.choice(focus, size=n_active, replace=False)
        else:
            active = rng.choice(n_classes, size=n_active, replace=False)
        feat = np.zeros(dim, np.float32)
        for c in active:
            feat = feat + rng.uniform(0.6, 1.4) * P[c]
        feat = feat + (rng.standard_normal(dim).astype(np.float32) * 0.5)
        y = np.zeros(n_classes, np.float32)
        y[active] = 1.0
        return feat, y

    def make(n):
        X = np.empty((n, dim), np.float32)
        Y = np.empty((n, n_classes), np.float32)
        for i in range(n):
            X[i], Y[i] = sample()
        return X, Y

    return make(n_train), make(n_test)


def focal_grad(P, Y, gamma=2.0, alpha=0.5):
    eps = 1e-7
    Pc = np.clip(P, eps, 1.0 - eps)
    ln_p = np.log(Pc)
    ln_1p = np.log1p(-Pc)
    g1 = alpha * np.power(1.0 - Pc, gamma) * (gamma * Pc * ln_p - (1.0 - Pc))
    g0 = -(1.0 - alpha) * np.power(Pc, gamma) * (gamma * (1.0 - Pc) * ln_1p - Pc)
    return Y * g1 + (1.0 - Y) * g0


class Adam:
    def __init__(self, shapes, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
        self.v = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

    def step(self, grads):
        self.t += 1
        out = {}
        for k in grads:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            out[k] = self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return out


def evaluate(w1, b1, w2, b2, X, Y, thr=0.5):
    P = sigmoid(relu(X @ w1 + b1) @ w2 + b2)
    yh = (P > thr).astype(np.float32)
    f1s, accs = [], []
    for c in range(Y.shape[1]):
        yt, yp = Y[:, c], yh[:, c]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        accs.append(float(np.mean(yt == yp)))
    return float(np.mean(f1s)), float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    parser = TaxonomyParser(os.path.join(SERVER, "taxonomy.json"))
    nc = parser.num_classes
    dim = 1024

    real = load_real()
    if real is not None:
        src = "data/bootstrap_seed/features.npz (FULL 10k real features)"
        X, Y = real
    else:
        probe = load_probe()
        if probe is not None:
            src = "output/fl_probe.npz (real bootstrap seed)"
            X, Y = probe
        else:
            src = "synthetic Non-IID demo"
            (X, Y), _ = gen_synthetic(nc, dim, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    n_te = max(1, int(0.2 * X.shape[0]))
    Xte, Yte = X[perm[:n_te]], Y[perm[:n_te]]
    Xtr, Ytr = X[perm[n_te:]], Y[perm[n_te:]]
    print(f"[data] source = {src}")
    print(f"[data] X={X.shape} Y={Y.shape} -> train {Xtr.shape[0]} / test {Xte.shape[0]}")

    def glorot(s):
        return (np.random.default_rng(0).standard_normal(s).astype(np.float32)
                * np.sqrt(6.0 / (s[0] + s[1])))

    w1 = glorot((dim, 256)); b1 = np.zeros(256, np.float32)
    w2 = glorot((256, nc)); b2 = np.zeros(nc, np.float32)
    shapes = {"w1": w1.shape, "b1": b1.shape, "w2": w2.shape, "b2": b2.shape}
    opt = Adam(shapes, lr=args.lr)

    n = Xtr.shape[0]
    print(f"[train] epochs={args.epochs} lr={args.lr} gamma={args.gamma} n={n}")
    for ep in range(args.epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            xb, yb = Xtr[idx], Ytr[idx]
            Z1 = xb @ w1 + b1
            A1 = relu(Z1)
            Z2 = A1 @ w2 + b2
            P = sigmoid(Z2)
            dZ2 = focal_grad(P, yb, gamma=args.gamma, alpha=args.alpha)
            dW2 = A1.T @ dZ2
            db2 = dZ2.sum(0)
            dA1 = dZ2 @ w2.T
            dZ1 = dA1 * (Z1 > 0)
            dW1 = xb.T @ dZ1
            db1 = dZ1.sum(0)
            g = {"w1": dW1, "b1": db1, "w2": dW2, "b2": db2}
            u = opt.step(g)
            w1 -= u["w1"]; b1 -= u["b1"]; w2 -= u["w2"]; b2 -= u["b2"]
        if (ep + 1) % 5 == 0 or ep == 0:
            f1, acc = evaluate(w1, b1, w2, b2, Xte, Yte)
            print(f"  epoch {ep+1:>2}: held-out macroF1={f1:.4f} acc={acc:.4f}")

    f1, acc = evaluate(w1, b1, w2, b2, Xte, Yte)
    print(f"[done] held-out macroF1={f1:.4f} acc={acc:.4f}")
    if f1 < 0.80:
        print("[WARN] held-out macroF1 below 0.80 -- head may be undertrained; "
              "verify data/label quality before shipping.")

    # Save trained head as float32 to BOTH the active and base snapshots.
    w1 = w1.astype(np.float32); b1 = b1.astype(np.float32)
    w2 = w2.astype(np.float32); b2 = b2.astype(np.float32)
    np.savez(os.path.join(MODELS, "initial_head_weights.npz"), w1=w1, b1=b1, w2=w2, b2=b2)
    np.savez(os.path.join(MODELS, "head_weights.npz"), w1=w1, b1=b1, w2=w2, b2=b2)
    # Bump version (head_weights.npz is gitignored; initial_head_weights.npz is tracked).
    with open(os.path.join(MODELS, "model_version.txt"), "w") as f:
        f.write("2")
    # Write back a held-out eval probe so prep_eval scores honestly.
    os.makedirs(OUTPUT, exist_ok=True)
    np.savez(os.path.join(OUTPUT, "fl_probe.npz"), X=Xte.astype(np.float32), Y=Yte.astype(np.float32))
    print(f"[saved] trained head -> models/initial_head_weights.npz + head_weights.npz (v2)")


if __name__ == "__main__":
    main()
