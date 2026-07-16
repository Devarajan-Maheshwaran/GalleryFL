import time
import json
import os
from typing import Dict, List, Optional
from pydantic import BaseModel

class ClientContribution(BaseModel):
    client_id: str
    nickname: str
    rounds_participated: int = 0
    images_contributed: int = 0
    avg_local_accuracy: float = 0.0
    trust_score: float = 1.0

class RoundMetrics(BaseModel):
    round: int
    timestamp: float
    num_clients: int
    global_loss: float
    global_accuracy: float
    per_class_f1: Dict[str, float]
    client_contributions: Dict[str, int]

class MetricsStore:
    def __init__(self, history_file: str = "output/metrics_history.json"):
        self.history_file = history_file
        self.history: List[RoundMetrics] = []
        self.client_cumulative: Dict[str, ClientContribution] = {}
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        self.load()

    def load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history = [RoundMetrics(**r) for r in data.get('history', [])]
                    self.client_cumulative = {
                        cid: ClientContribution(**c) for cid, c in data.get('client_cumulative', {}).items()
                    }
            except Exception as e:
                print(f"Failed to load metrics history: {e}")

    def save(self):
        data = {
            'history': [r.model_dump() for r in self.history],
            'client_cumulative': {cid: c.model_dump() for cid, c in self.client_cumulative.items()}
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def log_round(self, metrics: RoundMetrics):
        self.history.append(metrics)
        self.save()

    def update_client_contribution(self, client_id: str, nickname: str, images: int, local_acc: float):
        if client_id not in self.client_cumulative:
            self.client_cumulative[client_id] = ClientContribution(
                client_id=client_id,
                nickname=nickname
            )
        c = self.client_cumulative[client_id]
        c.rounds_participated += 1
        c.images_contributed += images
        # Simple moving average for local accuracy
        c.avg_local_accuracy = ((c.avg_local_accuracy * (c.rounds_participated - 1)) + local_acc) / c.rounds_participated
        self.save()

    def get_leaderboard(self) -> List[dict]:
        # Score calculation: images 40%, rounds 30%, accuracy 30%
        # We normalize dynamically for the leaderboard view
        max_images = max([c.images_contributed for c in self.client_cumulative.values()] + [1])
        max_rounds = max([c.rounds_participated for c in self.client_cumulative.values()] + [1])
        
        ranked = []
        for c in self.client_cumulative.values():
            img_score = c.images_contributed / max_images
            rnd_score = c.rounds_participated / max_rounds
            acc_score = c.avg_local_accuracy
            
            score = (img_score * 0.4) + (rnd_score * 0.3) + (acc_score * 0.3)
            
            ranked.append({
                "client_id": c.client_id,
                "nickname": c.nickname,
                "images": c.images_contributed,
                "rounds": c.rounds_participated,
                "accuracy": round(c.avg_local_accuracy, 4),
                "score": round(score, 4)
            })
            
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked
