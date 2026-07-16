import asyncio
import numpy as np
from typing import Dict, List, Optional
from config import ServerConfig
from metrics import MetricsStore, RoundMetrics
from model_manager import ModelManager
from ws_manager import ws_manager
import logging

class FLCoordinator:
    def __init__(self, config: ServerConfig, metrics_store: MetricsStore, model_manager: ModelManager):
        self.config = config
        self.metrics_store = metrics_store
        self.model_manager = model_manager
        
        self.current_round: int = 0
        self.is_training: bool = False
        
        self.registered_clients: Dict[str, dict] = {}
        self.client_updates: Dict[str, List[np.ndarray]] = {}
        self.client_metadata: Dict[str, dict] = {}

    def register_client(self, client_id: str, metadata: dict) -> bool:
        self.registered_clients[client_id] = metadata
        return True

    def get_registered_clients(self) -> List[str]:
        return list(self.registered_clients.keys())

    async def start_training(self):
        if self.is_training:
            return
            
        if len(self.registered_clients) < self.config.min_clients:
            logging.warning("Not enough clients to start training")
            return
            
        self.is_training = True
        self.current_round = 1
        logging.info("Training started")
        
        # Notify dashboard
        await ws_manager.broadcast({
            "type": "round_started",
            "data": {"round": self.current_round, "total_rounds": self.config.max_rounds}
        })
        
        # Actually start the round logic asynchronously
        asyncio.create_task(self._run_round())

    async def _run_round(self):
        logging.info(f"--- Round {self.current_round} starting ---")
        self.client_updates.clear()
        self.client_metadata.clear()
        
        # 1. Select clients
        # For this prototype, we'll select all connected & registered clients
        selected_clients = [cid for cid in self.registered_clients.keys() if cid in ws_manager.online_clients]
        if len(selected_clients) < self.config.min_clients:
            logging.error("Not enough online clients to run round")
            self.is_training = False
            return
            
        logging.info(f"Selected clients: {selected_clients}")
        
        # 2. Notify clients to start training
        for cid in selected_clients:
            await ws_manager.send_personal_message({
                "type": "update_requested",
                "data": {"round": self.current_round}
            }, cid)
            
        # 3. Wait for updates (handled in submit_update endpoint)
        # In a robust system, we would have a timeout here.
        
    def submit_client_update(self, client_id: str, weights: List[np.ndarray], metadata: dict):
        if not self.is_training:
            logging.warning("Received update but not training")
            return
            
        self.client_updates[client_id] = weights
        self.client_metadata[client_id] = metadata
        
        # Update contribution
        self.metrics_store.update_client_contribution(
            client_id=client_id,
            nickname=self.registered_clients.get(client_id, {}).get("nickname", client_id),
            images=metadata.get("num_samples", 0),
            local_acc=metadata.get("local_accuracy", 0.0)
        )
        
        # Check if we have enough updates
        if len(self.client_updates) >= len(self.registered_clients):
            asyncio.create_task(self._aggregate_and_finish_round())

    async def _aggregate_and_finish_round(self):
        logging.info(f"Aggregating {len(self.client_updates)} updates")
        
        # 1. Aggregation (Trimmed Mean for robust aggregation as per plan)
        aggregated_weights = []
        num_layers = len(self.model_manager.global_weights)
        
        for i in range(num_layers):
            layer_updates = [weights[i] for weights in self.client_updates.values()]
            stacked = np.stack(layer_updates, axis=0)
            
            # Trimmed mean: remove top and bottom trim_pct before averaging
            if len(layer_updates) >= 3:
                # E.g. with 10 clients, trim_pct 0.1 -> trim 1 from each end
                trim_count = max(1, int(len(layer_updates) * self.config.trim_pct))
                sorted_stacked = np.sort(stacked, axis=0)
                trimmed_stacked = sorted_stacked[trim_count:-trim_count]
                mean_layer = np.mean(trimmed_stacked, axis=0)
            else:
                # Not enough clients to trim, simple FedAvg
                mean_layer = np.mean(stacked, axis=0)
                
            aggregated_weights.append(mean_layer)
            
        # 2. Update global model
        self.model_manager.update_global_weights(aggregated_weights)
        
        # 3. Calculate metrics
        avg_loss = np.mean([m.get("local_loss", 0.0) for m in self.client_metadata.values()])
        avg_acc = np.mean([m.get("local_accuracy", 0.0) for m in self.client_metadata.values()])
        
        import time
        metrics = RoundMetrics(
            round=self.current_round,
            timestamp=time.time(),
            num_clients=len(self.client_updates),
            global_loss=float(avg_loss),
            global_accuracy=float(avg_acc),
            per_class_f1={},
            client_contributions={cid: m.get("num_samples", 0) for cid, m in self.client_metadata.items()}
        )
        self.metrics_store.log_round(metrics)
        
        # 4. Notify dashboard
        await ws_manager.broadcast({
            "type": "round_completed",
            "data": metrics.model_dump()
        })
        
        # 5. Check if finished
        if self.current_round >= self.config.max_rounds:
            logging.info("Training complete")
            self.is_training = False
            await ws_manager.broadcast({
                "type": "training_complete",
                "data": {"final_accuracy": avg_acc}
            })
        else:
            self.current_round += 1
            await self._run_round()
