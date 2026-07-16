import time
import json
import os
from typing import Dict, List
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
    total_samples: int = 0
    client_contributions: Dict[str, int] = {}

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
            except Exception:
                pass

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
            self.client_cumulative[client_id] = ClientContribution(client_id=client_id, nickname=nickname)
        c = self.client_cumulative[client_id]
        c.rounds_participated += 1
        c.images_contributed += images
        c.avg_local_accuracy = ((c.avg_local_accuracy * (c.rounds_participated - 1)) + local_acc) / c.rounds_participated
        self.save()

    def get_leaderboard(self) -> List[dict]:
        if not self.client_cumulative:
            return []
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

    def get_summary(self) -> dict:
        if not self.history:
            return {"total_rounds": 0, "total_participants": 0, "total_images": 0, "latest_accuracy": 0, "latest_loss": 0}
        latest = self.history[-1]
        total_images = sum(c.images_contributed for c in self.client_cumulative.values())
        return {
            "total_rounds": len(self.history),
            "total_participants": len(self.client_cumulative),
            "total_images": total_images,
            "latest_accuracy": round(latest.global_accuracy, 4),
            "latest_loss": round(latest.global_loss, 4),
        }

    def generate_report(self) -> dict:
        summary = self.get_summary()
        duration = 0
        if len(self.history) >= 2:
            duration = self.history[-1].timestamp - self.history[0].timestamp

        accuracy_trend = [{"round": h.round, "accuracy": round(h.global_accuracy, 4), "loss": round(h.global_loss, 4)} for h in self.history]

        return {
            "session": {
                "total_rounds": summary["total_rounds"],
                "total_participants": summary["total_participants"],
                "total_images_trained": summary["total_images"],
                "duration_seconds": round(duration, 1),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "results": {
                "final_accuracy": summary["latest_accuracy"],
                "final_loss": summary["latest_loss"],
                "accuracy_trend": accuracy_trend,
            },
            "privacy": {
                "aggregation_method": "Trimmed Mean",
                "differential_privacy": True,
                "gradient_clipping": True,
                "data_transmitted": "model_weights_only",
            },
            "leaderboard": self.get_leaderboard(),
            "per_round_history": [r.model_dump() for r in self.history],
        }
