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
        except Exception:
            # Corrupt or missing file: start clean rather than crash the server.
            self._counts = {}
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
                        "total_signals": self._total_signals,
                        "updated_at": self._updated_at,
                    },
                    f,
                )
        except Exception:
            # Persistence is best-effort; never block the request path on it.
            pass

    # --------------------------------------------------------------------- ingest
    def record(self, tags: List[str], weight: float = 1.0) -> int:
        """Record one demand signal containing `tags`.

        Applies recency decay to existing counts first, then adds the new signal.
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
            self._total_signals += 1
            self._updated_at = time.time()
            self._writes += 1
            if self._writes % self.persist_every == 0:
                self._persist()
        return len(valid)

    def flush(self) -> None:
        with self._lock:
            self._persist()

    # ---------------------------------------------------------------------- read
    def snapshot(self, top_n: int = 12) -> Dict[str, Any]:
        with self._lock:
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
