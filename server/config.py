from pydantic import BaseModel
import os

import json

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    min_clients: int = 2
    max_rounds: int = 10
    local_epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 16
    mu: float = 0.01
    trim_pct: float = 0.1
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    max_grad_norm: float = 1.0
    convergence_threshold: float = 0.001
    server_token: str = os.environ.get("FGT_SERVER_TOKEN", "fgt-pass")


    model_path: str = "models/base_model.tflite"
    validation_dir: str = "validation_data/"
    output_dir: str = "output/"

    @classmethod
    def load_or_default(cls) -> "ServerConfig":
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    return cls(**json.load(f))
            except Exception:
                pass
        return cls()

    def save(self):
        try:
            with open("config.json", "w") as f:
                json.dump(self.model_dump(), f, indent=2)
        except Exception:
            pass
