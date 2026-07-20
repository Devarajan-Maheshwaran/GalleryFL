from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field

from security import generate_access_code


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    min_clients: int = Field(default=2, ge=2)
    max_rounds: int = Field(default=10, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=0.05, gt=0)
    mu: float = Field(default=0.01, ge=0)

    # Local example-level DP. Epsilon/delta describe one FL round. Android
    # divides this budget across local epochs using basic composition.
    dp_epsilon: float = Field(default=3.0, ge=0)
    dp_delta: float = Field(default=1e-5, gt=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)

    # Weak pseudo-labels are allowed only at high confidence and count less than
    # explicit user corrections during local optimization/aggregation.
    pseudo_label_threshold: float = Field(default=0.80, ge=0, le=1)
    pseudo_label_weight: float = Field(default=0.25, ge=0, le=1)
    min_local_samples: int = Field(default=8, ge=1)

    # Robust FedAvg and held-out regression guard.
    server_delta_clip_norm: float = Field(default=1.0, gt=0)
    min_unlabeled_f1_improvement: float = Field(default=1e-6, ge=0, le=1)
    max_global_f1_drop: float = Field(default=0.01, ge=0, le=1)
    convergence_threshold: float = Field(default=0.001, ge=0)

    server_token: str = os.environ.get("FGT_SERVER_TOKEN", generate_access_code())

    @classmethod
    def load_or_default(cls) -> "ServerConfig":
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r", encoding="utf-8") as handle:
                    return cls(**json.load(handle))
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        with open("config.json", "w", encoding="utf-8") as handle:
            json.dump(self.model_dump(), handle, indent=2)
            handle.write("\n")
