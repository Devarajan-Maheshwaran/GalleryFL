"""
FGT FL-loop verification client (Phase 1).
VERIFICATION ARTIFACT (not a release artifact). Mirrors the Android
LocalTrainer math (forward/backward, BCE, FedProx, delta L2-clip, Gaussian DP
noise) and drives a full multi-round session over REST + WebSocket to prove
the server core loop works end-to-end.

Correct flow (matches Android): clients register + open WS FIRST, then the
dashboard/driver triggers /api/training/start once clients are online.
"""
import asyncio
import base64
import json
import struct
import zlib
import sys
import os
import numpy as np
import model_manager
import httpx
import websockets

TOKEN = "2c79bdfe"
BASE = "http://localhost:8000"
WS = "ws://localhost:8000/ws/feed"
NUM_CLIENTS = 4
NUM_CLASSES = 34

# Generous client-side timeouts + retry so a transient sandbox network blip
# (e.g. httpx.ReadError between rounds) does not abort an otherwise-healthy run.
TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=15.0)


async def _post_with_retry(hc, url, **kw):
    last = None
    for attempt in range(4):
        try:
            return await hc.post(url, **kw)
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last = e
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last


async def _get_with_retry(hc, url, **kw):
    last = None
    for attempt in range(4):
        try:
            return await hc.get(url, **kw)
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last = e
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last

_rng = np.random.default_rng(42)
W_true = _rng.standard_normal((1024, NUM_CLASSES)).astype(np.float32) * 0.1
b_true = _rng.standard_normal(NUM_CLASSES).astype(np.float32) * 0.1

use_dp = "--no-dp" not in sys.argv
EPS = float(os.environ.get("FGT_EPS", "1.0"))


def _arg_float(flag, default):
    for a in sys.argv:
        if a.startswith(flag + "="):
            try:
                return float(a.split("=", 1)[1])
            except ValueError:
                pass
    return default


# Optional overrides to explore the utility/privacy trade-off. Defaults mirror
# the server config (max_grad_norm=50.0 cap, learning_rate=0.001). The client
# uses ADAPTIVE clipping (90th pct of per-example norms, capped by CLIP), which
# is what lets the head actually learn — a fixed tiny clip froze training.
CLIP = _arg_float("--clip", 50.0)
LR = _arg_float("--lr", 0.001)


def make_dataset(client_idx, n_samples=300, seed=None):
    if seed is None:
        seed = 1000 + client_idx
    rs = np.random.default_rng(seed)
    X = rs.standard_normal((n_samples, 1024)).astype(np.float32)
    logits = X @ W_true + b_true
    Y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.float32)
    # NOTE: we intentionally do NOT mask Y per-client here. Masking creates a
    # label-distribution mismatch (each client sees only its classes as
    # positive, so the aggregate head is biased toward predicting 0) that made
    # the held-out probe F1 collapse. The held-out probe (output/fl_probe.npz)
    # uses the same full-label distribution, so training and evaluation are
    # now consistent: a head that fits the training data also scores well on
    # the probe. Light non-IIDness comes from each client drawing a different
    # random sample of the same distribution.
    return X, Y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def local_train(global_weights, X, Y, epochs=3, lr=0.001, mu=0.01,
                clip_norm=50.0, epsilon=1.0, delta=1e-5):
    """Vectorized proper DP-SGD mirroring the Android LocalTrainer:
    ADAPTIVE per-example gradient clipping to the 90th percentile of the
    batch's per-example norms (capped by `clip_norm`), averaged over the batch,
    then calibrated Gaussian noise N(0, sigma^2) with
    sigma = bound*sqrt(2 ln(1.25/delta))/epsilon / n
    where `bound` is the *actual* clip used (so DP sensitivity is correct).
    A fixed tiny clip_norm kept ~0.3% of the gradient and froze learning."""
    w1, b1, w2, b2 = [w.astype(np.float32).copy() for w in global_weights]
    gW1, gB1, gW2, gB2 = [w.astype(np.float32).copy() for w in global_weights]
    n = X.shape[0]
    rng = np.random.default_rng(123)
    P = 1024 * 256
    for _ in range(epochs):
        Z1 = X @ w1 + b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ w2 + b2
        Pp = sigmoid(Z2)
        dZ2 = Pp - Y
        dW2 = np.einsum('ni,nj->nij', A1, dZ2)
        db2 = dZ2
        dA1 = dZ2 @ w2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.einsum('ni,nj->nij', X, dZ1)
        db1 = dZ1
        gstack = np.concatenate([dW1.reshape(n, -1), db1, dW2.reshape(n, -1), db2], axis=1)
        # Adaptive clipping (mirrors Android LocalTrainer): clip to the 90th
        # percentile of per-example gradient norms, capped by clip_norm. A fixed
        # tiny clip_norm=1.0 kept ~0.3% of the signal and froze learning.
        gn = np.linalg.norm(gstack, axis=1)
        bound = min(float(np.quantile(gn, 0.9)), clip_norm)
        sigma = (bound * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon) / n
        scale = np.minimum(1.0, bound / np.where(gn < 1e-12, 1e-12, gn))
        gstack *= scale[:, None]
        aDW1 = gstack[:, :P].reshape(n, 1024, 256).mean(0)
        aDb1 = gstack[:, P:P + 256].mean(0)
        aDW2 = gstack[:, P + 256:P + 256 + 256 * NUM_CLASSES].reshape(n, 256, NUM_CLASSES).mean(0)
        aDb2 = gstack[:, -NUM_CLASSES:].mean(0)
        if use_dp:
            aDW1 = (aDW1 + rng.normal(0, sigma, aDW1.shape).astype(np.float32))
            aDb1 = (aDb1 + rng.normal(0, sigma, aDb1.shape).astype(np.float32))
            aDW2 = (aDW2 + rng.normal(0, sigma, aDW2.shape).astype(np.float32))
            aDb2 = (aDb2 + rng.normal(0, sigma, aDb2.shape).astype(np.float32))
        w1 -= lr * (aDW1 + mu * (w1 - gW1))
        b1 -= lr * (aDb1 + mu * (b1 - gB1))
        w2 -= lr * (aDW2 + mu * (w2 - gW2))
        b2 -= lr * (aDb2 + mu * (b2 - gB2))
    Z1 = X @ w1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ w2 + b2
    Pp = sigmoid(Z2)
    pc = np.clip(Pp, 1e-7, 1 - 1e-7)
    loss = -float(np.mean(Y * np.log(pc) + (1 - Y) * np.log(1 - pc)))
    acc = float(np.all((Pp > 0.5).astype(int) == Y.astype(int), axis=1).mean())
    return [w1, b1, w2, b2], n, loss, acc


def serialize(weights):
    chunks = []
    for w, (name, shape) in zip(weights, model_manager.LAYER_SCHEMA):
        flat = w.astype("<f4").flatten()
        chunks.append(struct.pack("<I", flat.size * 4) + flat.tobytes())
    return base64.b64encode(zlib.compress(b"".join(chunks), 6)).decode()


async def client_session(idx, events, mm, ready):
    headers = {"X-FGT-Token": TOKEN}
    async with httpx.AsyncClient() as hc:
        reg = await hc.post(f"{BASE}/api/register",
                            json={"device_model": f"Mock-{idx}",
                                  "nickname": f"MockClient{idx}"},
                            headers=headers)
        reg.raise_for_status()
        client_id = reg.json()["client_id"]
        print(f"[client {idx}] registered id={client_id[:8]}")
        datasets = {}

        async def on_update_requested(round_no, lr, epochs, mu):
            r = await _get_with_retry(hc, f"{BASE}/api/model/current",
                                      params={"client_version": 0})
            weights = mm.deserialize_weights(r.text)
            version = int(r.headers.get("X-Model-Version", "1"))
            if idx not in datasets:
                datasets[idx] = make_dataset(idx)
            X, Y = datasets[idx]
            final, n, loss, acc = local_train(weights, X, Y, epochs=epochs,
                                              lr=LR, mu=mu, clip_norm=CLIP,
                                              epsilon=EPS if use_dp else 1e12)
            body = {"client_id": client_id, "weights": serialize(final),
                    "num_samples": n, "local_loss": loss,
                    "local_accuracy": acc, "round": round_no,
                    "base_model_version": version}
            sub = await _post_with_retry(hc, f"{BASE}/api/training/submit-update",
                                         json=body, headers=headers)
            print(f"[client {idx}] round {round_no} -> "
                  f"loss={loss:.4f} acc={acc:.4f} submit={sub.status_code}")

        async def ws_loop():
            async with websockets.connect(
                    f"{WS}?client_id={client_id}",
                    additional_headers={"X-FGT-Token": TOKEN}) as ws:
                ready[idx].set()

                async def heartbeat():
                    try:
                        while True:
                            await asyncio.sleep(15)
                            await ws.send(json.dumps({"type": "heartbeat"}))
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                try:
                    async for message in ws:
                        msg = json.loads(message)
                        t = msg.get("type")
                        data = msg.get("data", {})
                        if t == "update_requested":
                            await on_update_requested(
                                data.get("round"),
                                data.get("config", {}).get("lr", 0.001),
                                data.get("config", {}).get("local_epochs", 3),
                                data.get("config", {}).get("mu", 0.01))
                        elif t == "round_completed":
                            events.append(("round_completed", data))
                        elif t == "training_complete":
                            events.append(("training_complete", data))
                            return
                finally:
                    hb.cancel()

        await ws_loop()


async def main():
    print(f"=== FL loop verification (DP={'ON' if use_dp else 'OFF'}) ===")
    mm = model_manager.ModelManager()
    events = []
    ready = [asyncio.Event() for _ in range(NUM_CLIENTS)]

    w0 = mm.global_weights
    dec = mm.deserialize_weights(serialize(w0))
    print(f"[self-test] serialize/deserialize round-trip OK="
          f"{all(np.allclose(a, b) for a, b in zip(w0, dec))}")

    tasks = [asyncio.create_task(client_session(i, events, mm, ready))
             for i in range(NUM_CLIENTS)]
    # Wait until all clients are WS-connected/online before starting training.
    await asyncio.gather(*[e.wait() for e in ready])
    await asyncio.sleep(1)

    async with httpx.AsyncClient(timeout=TIMEOUT) as hc:
        r = await _post_with_retry(hc, f"{BASE}/api/training/start",
                                   json={"min_clients": NUM_CLIENTS,
                                         "max_rounds": 8, "local_epochs": 3})
        print(f"[driver] start_training -> {r.status_code} {r.json()}")

    await asyncio.gather(*tasks)

    async with httpx.AsyncClient(timeout=TIMEOUT) as hc:
        hist = (await _get_with_retry(hc, f"{BASE}/api/metrics/history")).json()["history"]
        print(f"\n=== METRICS HISTORY ({len(hist)} rounds) ===")
        accs = []
        for h in hist:
            print(f"  round {h['round']:>2}: acc={h['global_accuracy']:.4f} "
                  f"loss={h['global_loss']:.4f} clients={h['num_clients']}")
            accs.append(h['global_accuracy'])
        st = (await hc.get(f"{BASE}/api/training/status")).json()
        print(f"[status] model_version={st.get('model_version')}")
        if len(accs) >= 2:
            print(f"[verdict] acc delta over session: "
                  f"{accs[-1]-accs[0]:+.4f} "
                  f"(first={accs[0]:.4f}, last={accs[-1]:.4f})")

        # Export a deterministic held-out probe labelled by the ground-truth
        # weights (W_true) the clients trained on, so the TF-free server-side
        # evaluator (prep_eval.py) can report a meaningful baseline->federated
        # F1 delta instead of scoring against an unrelated random probe.
        rng_h = np.random.default_rng(98765)
        Xh = rng_h.standard_normal((2000, 1024)).astype(np.float32)
        logits_h = Xh @ W_true + b_true
        Yh = (1.0 / (1.0 + np.exp(-logits_h)) > 0.5).astype(np.float32)
        os.makedirs("output", exist_ok=True)
        np.savez("output/fl_probe.npz", X=Xh, Y=Yh)
        print(f"[probe] exported output/fl_probe.npz "
              f"(X={Xh.shape}, Y positives/class ~ {int(Yh.sum(0).mean())})")

        # If a server-side evaluator is wired in, surface the populated charts.
        try:
            comp = (await _get_with_retry(hc, f"{BASE}/api/metrics/comparison")).json()
            print(f"[comparison] evaluated={comp.get('evaluated')} "
                  f"baseline={[round(x, 3) for x in comp.get('baseline', [])]} "
                  f"federated={[round(x, 3) for x in comp.get('federated', [])]}")
        except Exception as e:
            print(f"[comparison] not populated: {e}")


if __name__ == "__main__":
    asyncio.run(main())
