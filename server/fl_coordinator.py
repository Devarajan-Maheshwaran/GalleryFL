import asyncio
import time
import numpy as np
from typing import Dict, List, Set
from config import ServerConfig
from metrics import MetricsStore, RoundMetrics
from model_manager import ModelManager
from ws_manager import ws_manager
from security import validate_update
import logging

class FLCoordinator:
    def __init__(self, config: ServerConfig, metrics_store: MetricsStore, model_manager: ModelManager, trust_scorer=None):
        self.config = config
        self.metrics_store = metrics_store
        self.model_manager = model_manager
        self.trust_scorer = trust_scorer

        self.current_round: int = 0
        self.is_training: bool = False
        self.session_start_time: float = 0

        self.registered_clients: Dict[str, dict] = {}
        self.client_updates: Dict[str, List[np.ndarray]] = {}
        self.client_metadata: Dict[str, dict] = {}
        self.round_participants: Set[str] = set()
        self.round_timeout_seconds: int = 120
        self._aggregation_lock = asyncio.Lock()

    def register_client(self, client_id: str, metadata: dict) -> bool:
        metadata["last_heartbeat"] = time.time()
        self.registered_clients[client_id] = metadata
        logging.info(f"Client registered: {client_id} ({metadata.get('nickname', 'unknown')})")
        return True

    def get_online_registered(self) -> List[str]:
        return [cid for cid in self.registered_clients if cid in ws_manager.online_clients]

    def record_heartbeat(self, client_id: str) -> None:
        if client_id in self.registered_clients:
            self.registered_clients[client_id]["last_heartbeat"] = time.time()

    async def on_client_reconnected(self, client_id: str) -> None:
        """Resume an interrupted participant without admitting new mid-round clients."""
        self.record_heartbeat(client_id)
        if self.is_training and client_id in self.round_participants and client_id not in self.client_updates:
            logging.info("Reissuing round %s request to reconnected client %s", self.current_round, client_id[:8])
            await self._request_update(client_id)

    async def start_training(self):
        if self.is_training:
            return

        online = self.get_online_registered()
        if len(online) < self.config.min_clients:
            logging.warning(f"Need {self.config.min_clients} clients, only {len(online)} online")
            await ws_manager.broadcast({"type": "error", "data": {"message": f"Need at least {self.config.min_clients} online clients"}})
            return

        self.is_training = True
        self.current_round = 1
        self.session_start_time = time.time()
        logging.info(f"Training session started: {self.config.max_rounds} rounds, {len(online)} clients")

        await ws_manager.broadcast({
            "type": "round_started",
            "data": {"round": self.current_round, "total_rounds": self.config.max_rounds}
        })

        await self._run_round()

    async def _run_round(self):
        if not self.is_training:
            return

        logging.info(f"Round {self.current_round}/{self.config.max_rounds} starting")
        self.client_updates.clear()
        self.client_metadata.clear()

        selected = self.get_online_registered()
        if len(selected) < self.config.min_clients:
            logging.error(f"Round aborted: {len(selected)} online, need {self.config.min_clients}")
            self.is_training = False
            await ws_manager.broadcast({"type": "training_complete", "data": {"reason": "insufficient_clients", "final_round": self.current_round - 1}})
            return

        self.round_participants = set(selected)

        for cid in selected:
            await self._request_update(cid)

        asyncio.create_task(self._round_timeout())

    async def _request_update(self, client_id: str) -> None:
        await ws_manager.send_personal_message({
            "type": "update_requested",
            "data": {"round": self.current_round, "config": {
                "local_epochs": self.config.local_epochs,
                "lr": self.config.learning_rate,
                "mu": self.config.mu,
            }},
        }, client_id)

    async def _round_timeout(self):
        round_at_start = self.current_round
        await asyncio.sleep(self.round_timeout_seconds)
        if self.is_training and self.current_round == round_at_start and len(self.client_updates) > 0:
            logging.warning(f"Round {round_at_start} timed out with {len(self.client_updates)} updates, proceeding to aggregate")
            await self._aggregate_and_advance()

    def submit_client_update(self, client_id: str, weights: List[np.ndarray], metadata: dict) -> bool:
        if not self.is_training:
            return False
        if metadata.get("round") != self.current_round:
            logging.warning("Rejected stale update from %s for round %s (current %s)", client_id[:8], metadata.get("round"), self.current_round)
            return False
        if metadata.get("base_model_version") != self.model_manager.current_version:
            logging.warning("Rejected update from %s based on model v%s (current v%s)", client_id[:8], metadata.get("base_model_version"), self.model_manager.current_version)
            return False
        if client_id in self.client_updates:
            logging.warning("Rejected duplicate update from %s for round %s", client_id[:8], self.current_round)
            return False

        self.client_updates[client_id] = weights
        self.client_metadata[client_id] = metadata

        self.metrics_store.update_client_contribution(
            client_id=client_id,
            nickname=self.registered_clients.get(client_id, {}).get("nickname", client_id[:8]),
            images=metadata.get("num_samples", 0),
            local_acc=metadata.get("local_accuracy", 0.0),
            epsilon_used=self.config.dp_epsilon
        )

        expected = len(self.round_participants)
        logging.info(f"Registered client update from {client_id[:8]} (total round updates: {len(self.client_updates)}/{expected})")
        
        if len(self.client_updates) >= max(self.config.min_clients, expected):
            asyncio.create_task(self._aggregate_and_advance())
        return True

    async def _aggregate_and_advance(self):
        async with self._aggregation_lock:
            if not self.is_training or len(self.client_updates) == 0:
                return

            num_updates = len(self.client_updates)
            logging.info(f"=== [AGGREGATING] Aggregating {num_updates} client updates for round {self.current_round} ===")
            aggregated = self._trimmed_mean_aggregate()

            # FedProx is enforced by the client local objective. Applying it a
            # second time here is not FedProx and double-regularises updates.
            self.model_manager.update_global_weights(aggregated)
            asyncio.create_task(self._run_evaluation())

            avg_loss = float(np.mean([m.get("local_loss", 0.0) for m in self.client_metadata.values()]))
            avg_acc = float(np.mean([m.get("local_accuracy", 0.0) for m in self.client_metadata.values()]))
            total_samples = sum(m.get("num_samples", 0) for m in self.client_metadata.values())
            metrics = RoundMetrics(
                round=self.current_round, timestamp=time.time(), num_clients=num_updates,
                global_loss=avg_loss, global_accuracy=avg_acc, total_samples=total_samples,
                client_contributions={cid: m.get("num_samples", 0) for cid, m in self.client_metadata.items()},
            )
            converged = bool(self.metrics_store.history and abs(avg_loss - self.metrics_store.history[-1].global_loss) < self.config.convergence_threshold)
            self.metrics_store.log_round(metrics)
            await ws_manager.broadcast({"type": "round_completed", "data": metrics.model_dump()})

            if self.current_round >= self.config.max_rounds or converged:
                elapsed = time.time() - self.session_start_time
                reason = "converged" if converged else "max_rounds_reached"
                logging.info(f"Training complete after {self.current_round} rounds ({elapsed:.0f}s) - {reason}")
                self.is_training = False
                await ws_manager.broadcast({"type": "training_complete", "data": {
                    "final_round": self.current_round, "final_accuracy": avg_acc, "final_loss": avg_loss,
                    "duration_seconds": round(elapsed, 1), "reason": reason,
                }})
            else:
                self.current_round += 1
                await ws_manager.broadcast({"type": "round_started", "data": {
                    "round": self.current_round, "total_rounds": self.config.max_rounds,
                }})
                await self._run_round()

    def _trimmed_mean_aggregate(self) -> List[np.ndarray]:
        aggregated = []
        num_layers = len(self.model_manager.global_weights)
        sample_counts = np.array([self.client_metadata[cid].get("num_samples", 1) for cid in self.client_updates])
        
        # Apply trust scores
        if self.trust_scorer:
            trust_scores = np.array([self.trust_scorer.get_score(cid) for cid in self.client_updates])
        else:
            trust_scores = np.ones_like(sample_counts, dtype=float)
            
        effective_weights = sample_counts * trust_scores
        sample_weights = effective_weights / (effective_weights.sum() + 1e-9)

        for i in range(num_layers):
            layer_updates = [self.client_updates[cid][i] for cid in self.client_updates]
            stacked = np.stack(layer_updates, axis=0)

            if len(layer_updates) >= 4:
                trim_count = max(1, int(len(layer_updates) * self.config.trim_pct))
                sorted_stack = np.sort(stacked, axis=0)
                trimmed = sorted_stack[trim_count:-trim_count]
                mean_layer = np.mean(trimmed, axis=0)
            elif len(layer_updates) >= 2:
                weighted = np.zeros_like(layer_updates[0])
                for j, cid in enumerate(self.client_updates):
                    weighted += sample_weights[j] * layer_updates[j]
                mean_layer = weighted
            else:
                mean_layer = layer_updates[0]

            aggregated.append(mean_layer.astype(np.float32))

        return aggregated

    async def _run_evaluation(self):
        try:
            import subprocess
            logging.info("Starting background model evaluation on server...")
            subprocess.Popen(["python", "prep_model.py", "evaluate"])
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")

    async def on_client_disconnected(self, client_id: str) -> None:
        if client_id in self.round_participants:
            self.round_participants.discard(client_id)
            logging.info(f"Client {client_id[:8]} disconnected. Removed from round participants. Remaining: {len(self.round_participants)}")
            # Clean up their update if they disconnected before aggregation
            self.client_updates.pop(client_id, None)
            self.client_metadata.pop(client_id, None)
            
            # If we now have enough updates from the remaining participants, aggregate!
            expected = len(self.round_participants)
            if self.is_training and len(self.client_updates) >= max(self.config.min_clients, expected) and expected > 0:
                asyncio.create_task(self._aggregate_and_advance())

