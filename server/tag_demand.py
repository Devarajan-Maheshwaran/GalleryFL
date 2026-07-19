"""
tag_demand.py — Demand-weighted tag signal store.

GalleryFL's taxonomy is static (categories + leaf tags in taxonomy.json). To make
the product feel *alive* and to let the system learn which tags matter to real
users, clients report the tags they actually USE: tags a user applies while
organizing photos, or tags the on-device model predicts with high confidence.
The server accumulates these into a recency-weighted "demand" signal that powers
the dashboard's Trending Tags view and can later bias taxonomy ordering.

Design constraints honoured:
  * Additive only. This module never touches model weights, aggregation, the
    coordinator, or the WebSocket stack. It is a standalone side-store.
  * Thread-safe (async endpoints call it concurrently) via a single lock.
  * Durable: counts are persisted to output/tag_demand.json so demand survives
    server restarts, and a corrupt/missing file degrades gracefully to empty.
  * Genuinely *dynamic*: a recency decay is applied on every record() call, so
    tags that stop being used fade out instead of dominating a lifetime tally.
"""
import json
import os
import threading
import time
from typing import Dict, List, Any, Optional


class TagDemandStore:
    def __init__(
        self,
        persist_path: str = "output/tag_demand.json",
        decay: float = 0.95,
        persist_every: int = 5,
    ):
        # decay in (0, 1): each record() call multiplies existing counts by this
        # factor before adding the new signal, giving recent usage more weight.
        self.decay = max(0.0, min(0.999, float(decay)))
        self.persist_path = persist_path
        self.persist_every = max(1, int(persist_every))
        self._lock = threading.Lock()
        self._counts: Dict[str, float] = {}
        # Per-client demand: each client's own tag-usage histogram. This is the
        # server half of demand-driven Non-IID — it lets the coordinator see how
        # each user's photo distribution differs (Non-IID) and weight/serve
        # personalisation accordingly.
        self._per_client: Dict[str, Dict[str, float]] = {}
        self._total_signals = 0
        self._writes = 0
        self._updated_at = 0.0
        self._load()

    # --------------------------------------------------------------- load/persist
    def _load(self) -> None:
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                self._counts = {str(k): float(v) for k, v in (data.get("counts") or {}).items()}
                self._total_signals = int(data.get("total_signals", 0) or 0)
                self._updated_at = float(data.get("updated_at", 0.0) or 0.0)
                self._per_client = {
                    str(cid): {str(k): float(v) for k, v in (c or {}).items()}
                    for cid, c in (data.get("per_client") or {}).items()
                }
        except Exception:
            # Corrupt or missing file: start clean rather than crash the server.
            self._counts = {}
            self._per_client = {}
            self._total_signals = 0
            self._updated_at = 0.0

    def _persist(self) -> None:
        try:
            parent = os.path.dirname(self.persist_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(
                    {
                        "counts": self._counts,
                        "per_client": self._per_client,
                        "total_signals": self._total_signals,
                        "updated_at": self._updated_at,
                    },
                    f,
                )
        except Exception:
            # Persistence is best-effort; never block the request path on it.
            pass

    # --------------------------------------------------------------------- ingest
    def record(self, tags: List[str], client_id: Optional[str] = None, weight: float = 1.0) -> int:
        """Record one demand signal containing `tags`.

        Applies recency decay to existing counts first, then adds the new signal.
        When `client_id` is supplied the signal is also folded into that
        client's personal histogram (used for Non-IID personalisation).
        Returns the number of valid (non-empty string) tags recorded.
        """
        valid = [str(t) for t in (tags or []) if isinstance(t, str) and t.strip()]
        if not valid:
            return 0
        w = float(weight)
        with self._lock:
            if self.decay < 1.0:
                for k in list(self._counts.keys()):
                    self._counts[k] *= self.decay
                    if self._counts[k] < 1e-6:
                        del self._counts[k]
            for t in valid:
                self._counts[t] = self._counts.get(t, 0.0) + w
            if client_id:
                cid = str(client_id)
                self._per_client.setdefault(cid, {})
                if self.decay < 1.0:
                    for k in list(self._per_client[cid].keys()):
                        self._per_client[cid][k] *= self.decay
                        if self._per_client[cid][k] < 1e-6:
                            del self._per_client[cid][k]
                for t in valid:
                    self._per_client[cid][t] = self._per_client[cid].get(t, 0.0) + w
            self._total_signals += 1
            self._updated_at = time.time()
            self._writes += 1
            if self._writes % self.persist_every == 0:
                self._persist()
        return len(valid)

    def demand_weight_for_client(self, client_id: str, default: float = 1.0) -> float:
        """Demand-aware aggregation weight for one client.

        Returns a factor in ~[0.75, 1.25]: clients whose local tag usage aligns
        with the global (community) demand get slightly more say in the global
        head, so the aggregated model is "smarter" for what users actually want,
        while per-user personalisation (on-device bias offsets) still handles each
        user's own Non-IID skew. Defaults to 1.0 when a client has no demand
        history yet (e.g. the verification harness), so it never breaks training.
        """
        with self._lock:
            pc = self._per_client.get(client_id)
            if not pc:
                return default
            global_c = dict(self._counts)
        if not global_c:
            return default
        tags = list(set(pc.keys()) | set(global_c.keys()))
        a = np.array([pc.get(t, 0.0) for t in tags], dtype=float)
        b = np.array([global_c.get(t, 0.0) for t in tags], dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return default
        sim = float(np.dot(a, b) / (na * nb))
        return float(0.75 + 0.5 * max(0.0, min(1.0, sim)))

    def flush(self) -> None:
        with self._lock:
            self._persist()

    # ---------------------------------------------------------------------- read
    def snapshot(self, top_n: int = 12, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Return demand. If `client_id` is given and that client has a personal
        histogram, return THAT client's demand (Non-IID personalisation view);
        otherwise return the global recency-weighted demand."""
        with self._lock:
            if client_id and client_id in self._per_client and self._per_client[client_id]:
                counts = dict(self._per_client[client_id])
            else:
                counts = dict(self._counts)
        max_c = max(counts.values()) if counts else 0.0
        weights = {
            t: {
                "count": round(c, 4),
                "weight": round(c / max_c, 4) if max_c > 0 else 0.0,
            }
            for t, c in counts.items()
        }
        trending = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_n))]
        return {
            "scope": "client" if (client_id and client_id in self._per_client) else "global",
            "client_id": client_id,
            "updated_at": self._updated_at,
            "total_signals": self._total_signals,
            "num_active_tags": len(counts),
            "weights": weights,
            "trending": [
                {
                    "tag": t,
                    "count": round(c, 4),
                    "weight": round(c / max_c, 4) if max_c > 0 else 0.0,
                }
                for t, c in trending
            ],
        }
