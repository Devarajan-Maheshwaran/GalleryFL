import numpy as np
from typing import List, Tuple, Dict
import json
import os
import time
import secrets
import string


def generate_access_code(length: int = 8) -> str:
    """Simple, human-friendly access code: uppercase letters + digits,
    excluding visually ambiguous characters (0/O, 1/I/L)."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))

class AuditLog:
    def __init__(self, log_file: str = "output/audit.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, event_type: str, details: dict):
        entry = {
            "timestamp": time.time(),
            "event": event_type,
            "details": details
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

class TrustScorer:
    def __init__(self):
        self.scores: Dict[str, float] = {}

    def get_score(self, client_id: str) -> float:
        return self.scores.get(client_id, 1.0)

    def update_score(self, client_id: str, is_valid: bool, reason: str = ""):
        current = self.get_score(client_id)
        if is_valid:
            self.scores[client_id] = min(1.0, current + 0.05)
        else:
            self.scores[client_id] = max(0.0, current - 0.2)

def validate_update(
    client_id: str,
    update_weights: List[np.ndarray],
    global_weights: List[np.ndarray],
    update_norm: float,
    historical_norms: List[float]
) -> Tuple[bool, str]:
    if len(update_weights) != len(global_weights):
        return False, "Layer count mismatch"

    for u_w, g_w in zip(update_weights, global_weights):
        if u_w.shape != g_w.shape:
            return False, f"Shape mismatch: {u_w.shape} vs {g_w.shape}"

    for i, w in enumerate(update_weights):
        if not np.isfinite(w).all():
            return False, f"Layer {i} contains NaN or Inf"

    # Anomaly gate on the DELTA norm (submitted - current global). Legitimate
    # FL deltas can be tiny (~1e-3) or clipped to ~1.0 depending on client
    # strategy, so a fixed absolute floor is fragile. We combine:
    #   - reject an essentially-zero delta (client sent unchanged weights),
    #   - reject egregious deltas with an absolute upper bound (poisoning),
    #   - apply the 3-sigma test ONLY when historical variance is meaningful.
    # The previous 3-sigma-only check collapsed to "reject everything" whenever
    # historical norms were near-identical (std ~ 0), which wrongly dropped
    # every legitimate small update after the first rounds.
    if update_norm < 1e-9:
        return False, "Delta norm ~0 (client submitted unchanged weights)"

    if len(historical_norms) > 5:
        mean_norm = float(np.mean(historical_norms[-50:]))
        std_norm = float(np.std(historical_norms[-50:]))
        if std_norm > 1e-6 and update_norm > mean_norm + 3 * std_norm:
            return False, f"Norm {update_norm:.4f} exceeds 3-sigma ({mean_norm:.4f} + 3*{std_norm:.4f})"

    if update_norm > 100.0:
        return False, f"Delta norm {update_norm:.4f} exceeds absolute bound 100.0"

    return True, ""

class RateLimiter:
    """Anti-flood guard keyed by (client_id, round).

    A legitimate FL client submits exactly one update per round. The
    coordinator already rejects duplicate/stale submissions for the current
    round, so the only thing this limiter needs to prevent is a client
    replaying the *same* round many times in quick succession. It must NOT
    block a client from submitting across successive rounds, even if rounds
    complete faster than a fixed wall-clock window (e.g. fast automated or
    on-device clients). Keying on (client, round) achieves that.
    """

    def __init__(self, window_seconds: float = 2.0):
        self.window = window_seconds
        self.last_update: Dict[tuple, float] = {}

    def check(self, client_id: str, round_no: int) -> bool:
        key = (client_id, round_no)
        now = time.time()
        last = self.last_update.get(key, 0.0)
        if now - last < self.window:
            return False
        self.last_update[key] = now
        return True
