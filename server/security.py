import numpy as np
from typing import List, Tuple, Dict
import json
import os
import time

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

    if len(historical_norms) > 5:
        mean_norm = np.mean(historical_norms[-50:])
        std_norm = np.std(historical_norms[-50:])
        if std_norm > 0 and update_norm > mean_norm + (3 * std_norm):
            return False, f"Norm {update_norm:.2f} exceeds 3-sigma ({mean_norm:.2f} + 3*{std_norm:.2f})"

    return True, ""
