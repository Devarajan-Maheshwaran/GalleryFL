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
    """
    Validates a client update for anomaly detection/poisoning.
    """
    # 1. Check weight shapes
    if len(update_weights) != len(global_weights):
        return False, "Weight layer count mismatch"
        
    for u_w, g_w in zip(update_weights, global_weights):
        if u_w.shape != g_w.shape:
            return False, "Weight shape mismatch"
            
    # 2. Check for NaN/Inf
    for w in update_weights:
        if not np.isfinite(w).all():
            return False, "Contains NaN or Inf values"
            
    # 3. Check update norm against historical distribution
    if len(historical_norms) > 5:
        mean_norm = np.mean(historical_norms)
        std_norm = np.std(historical_norms)
        
        # Reject if > 3 standard deviations from mean
        if update_norm > mean_norm + (3 * std_norm):
            return False, f"Norm {update_norm:.4f} exceeds 3-sigma bound"
            
    return True, ""
