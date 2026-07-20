import asyncio
import json
import os
import time
import numpy as np
from typing import Dict, List, Set
from config import ServerConfig
from metrics import MetricsStore, RoundMetrics
from model_manager import ModelManager
from ws_manager import ws_manager
from security import validate_update
from fl_math import clipped_fedavg
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
        self.skipped_clients: Set[str] = set()
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
        if (
            self.is_training
            and client_id in self.round_participants
            and client_id not in self.client_updates
            and client_id not in self.skipped_clients
        ):
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
        self.skipped_clients.clear()

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
                "dp_epsilon": self.config.dp_epsilon,
                "dp_delta": self.config.dp_delta,
                "max_grad_norm": self.config.max_grad_norm,
                "pseudo_label_threshold": self.config.pseudo_label_threshold,
                "pseudo_label_weight": self.config.pseudo_label_weight,
                "min_local_samples": self.config.min_local_samples,
            }},
        }, client_id)

    async def _round_timeout(self):
        round_at_start = self.current_round
        await asyncio.sleep(self.round_timeout_seconds)
        if not self.is_training or self.current_round != round_at_start:
            return
        if len(self.client_updates) >= self.config.min_clients:
            logging.warning(
                "Round %s timed out with %s usable updates; aggregating",
                round_at_start,
                len(self.client_updates),
            )
            await self._aggregate_and_advance()
        else:
            logging.warning(
                "Round %s stopped: only %s usable updates (need %s)",
                round_at_start,
                len(self.client_updates),
                self.config.min_clients,
            )
            self.is_training = False
            await ws_manager.broadcast({
                "type": "training_complete",
                "data": {
                    "reason": "insufficient_usable_updates",
                    "final_round": max(0, self.current_round - 1),
                },
            })

    def submit_client_update(self, client_id: str, weights: List[np.ndarray], metadata: dict) -> bool:
        if not self.is_training or client_id not in self.round_participants:
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
            epsilon_used=self.config.dp_epsilon,
            delta_used=self.config.dp_delta if self.config.dp_epsilon > 0 else 0.0,
        )

        expected = len(self.round_participants)
        logging.info(
            "Registered client update from %s (usable=%s, skipped=%s, expected=%s)",
            client_id[:8],
            len(self.client_updates),
            len(self.skipped_clients),
            expected,
        )
        if len(self.client_updates) + len(self.skipped_clients) >= expected:
            if len(self.client_updates) >= self.config.min_clients:
                asyncio.create_task(self._aggregate_and_advance())
            else:
                asyncio.create_task(self._stop_for_insufficient_updates())
        return True

    def skip_client_update(self, client_id: str, metadata: dict) -> bool:
        """Record a valid response from a device with too little usable data."""
        if not self.is_training or client_id not in self.round_participants:
            return False
        if metadata.get("round") != self.current_round:
            return False
        if metadata.get("base_model_version") != self.model_manager.current_version:
            return False
        if client_id in self.client_updates or client_id in self.skipped_clients:
            return False
        self.skipped_clients.add(client_id)
        logging.info("Client %s skipped round %s: %s", client_id[:8], self.current_round, metadata.get("reason"))
        expected = len(self.round_participants)
        if len(self.client_updates) + len(self.skipped_clients) >= expected:
            if len(self.client_updates) >= self.config.min_clients:
                asyncio.create_task(self._aggregate_and_advance())
            else:
                asyncio.create_task(self._stop_for_insufficient_updates())
        return True

    async def _stop_for_insufficient_updates(self) -> None:
        async with self._aggregation_lock:
            if not self.is_training:
                return
            self.is_training = False
            await ws_manager.broadcast({
                "type": "training_complete",
                "data": {
                    "reason": "insufficient_usable_updates",
                    "final_round": max(0, self.current_round - 1),
                },
            })

    async def _aggregate_and_advance(self):
        async with self._aggregation_lock:
            if not self.is_training or len(self.client_updates) == 0:
                return

            num_updates = len(self.client_updates)
            logging.info("=== [AGGREGATING] %s usable client updates for round %s ===", num_updates, self.current_round)

            # FedProx is the local objective; the server performs robust,
            # effective-sample-weighted FedAvg over bounded client deltas.
            candidate_weights = self._robust_fedavg_aggregate()
            current_eval = await asyncio.to_thread(
                self._run_evaluation, self.model_manager.global_weights, False, self.model_manager.current_version
            )
            candidate_eval = await asyncio.to_thread(
                self._run_evaluation, candidate_weights, False, self.model_manager.current_version + 1
            )

            accepted_candidate = True
            if current_eval.get("available") and candidate_eval.get("available"):
                human_labels = sum(
                    int(metadata.get("human_labeled_samples", 0))
                    for metadata in self.client_metadata.values()
                )
                # Unlabelled self-training is accepted only when it does not
                # regress the held-out proxy. Trusted corrections may trade a
                # small proxy drop for real gallery-domain adaptation.
                if human_labels > 0:
                    baseline_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "models", "baseline_metrics.json"
                    )
                    try:
                        with open(baseline_path, "r", encoding="utf-8") as handle:
                            deployment_baseline = float(json.load(handle)["macro_f1"])
                    except Exception:
                        deployment_baseline = float(current_eval["macro_f1"])
                    allowed_floor = deployment_baseline - self.config.max_global_f1_drop
                else:
                    # Unlabelled self-training must demonstrate a strict held-out
                    # improvement; equality is not enough to create a new model.
                    allowed_floor = (
                        float(current_eval["macro_f1"])
                        + self.config.min_unlabeled_f1_improvement
                    )
                if float(candidate_eval["macro_f1"]) < allowed_floor:
                    accepted_candidate = False
                    logging.warning(
                        "Rejected round %s candidate: macro F1 %.4f < guarded floor %.4f",
                        self.current_round,
                        candidate_eval["macro_f1"],
                        allowed_floor,
                    )

            if accepted_candidate:
                self.model_manager.update_global_weights(candidate_weights)
                eval_result = await asyncio.to_thread(
                    self._run_evaluation, self.model_manager.global_weights, True, self.model_manager.current_version
                )
            else:
                eval_result = await asyncio.to_thread(
                    self._run_evaluation, self.model_manager.global_weights, True, self.model_manager.current_version
                )

            total_samples = sum(m.get("num_samples", 0) for m in self.client_metadata.values())
            if eval_result.get("available"):
                avg_loss = float(eval_result["loss"])
                avg_acc = float(eval_result["macro_f1"])
            else:
                denominator = max(total_samples, 1)
                avg_loss = sum(
                    float(m.get("local_loss", 0.0)) * int(m.get("num_samples", 0))
                    for m in self.client_metadata.values()
                ) / denominator
                avg_acc = sum(
                    float(m.get("local_accuracy", 0.0)) * int(m.get("num_samples", 0))
                    for m in self.client_metadata.values()
                ) / denominator
            metrics = RoundMetrics(
                round=self.current_round, timestamp=time.time(), num_clients=num_updates,
                global_loss=avg_loss, global_accuracy=avg_acc, total_samples=total_samples,
                client_contributions={cid: m.get("num_samples", 0) for cid, m in self.client_metadata.items()},
            )
            converged = bool(self.metrics_store.history and abs(avg_loss - self.metrics_store.history[-1].global_loss) < self.config.convergence_threshold)
            self.metrics_store.log_round(metrics)
            await ws_manager.broadcast({"type": "round_completed", "data": metrics.model_dump()})

            # NOTE: we deliberately do NOT early-stop on `converged` here. The
            # absolute convergence_threshold (0.001) is far too tight for FL
            # with DP/local reporting noise — client-reported mean loss is often
            # stable to <0.001 round-to-round, which falsely triggered a stop
            # after 2 rounds. Let the session run to max_rounds; the dashboard
            # shows the full curve and the user can stop manually.
            if self.current_round >= self.config.max_rounds:
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

    def _robust_fedavg_aggregate(self) -> List[np.ndarray]:
        client_ids = list(self.client_updates)
        updates = [self.client_updates[client_id] for client_id in client_ids]
        effective_counts = []
        for client_id in client_ids:
            metadata = self.client_metadata[client_id]
            # Explicit corrections count fully; pseudo-labels have the public
            # reduced weight configured by the server. Trust can only reduce a
            # device's influence. Tag-demand popularity does not alter training.
            human = max(0, int(metadata.get("human_labeled_samples", 0)))
            pseudo = max(0, int(metadata.get("pseudo_labeled_samples", 0)))
            effective = human + self.config.pseudo_label_weight * pseudo
            if effective <= 0:
                effective = float(metadata.get("num_samples", 1))
            trust = self.trust_scorer.get_score(client_id) if self.trust_scorer else 1.0
            effective_counts.append(max(1e-6, effective * trust))
        return clipped_fedavg(
            self.model_manager.global_weights,
            updates,
            effective_counts,
            self.config.server_delta_clip_norm,
        )

    def _run_evaluation(
        self,
        weights: List[np.ndarray] | None = None,
        persist: bool = True,
        model_version: int | None = None,
    ) -> dict:
        """Evaluate a four-tensor head on cached held-out raw features.

        ``models/validation_features.npz`` contains the held-out raw backbone
        features used for the deployment baseline. The deployed head has feature
        normalization folded into w1/b1, so this pure NumPy path is also the
        Android inference graph and requires no training framework.
        """
        from model_eval import LABELS, atomic_write_json, evaluate

        server_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(server_dir, "models", "validation_features.npz")
        if not os.path.exists(cache_path):
            logging.error("Required validation artifact is missing: %s", cache_path)
            return {"available": False}

        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                features = np.asarray(cache["features"], dtype=np.float32)
                labels = np.asarray(cache["labels"], dtype=np.int64)
                cached_names = tuple(str(value) for value in cache["label_names"].tolist())
            if cached_names != LABELS:
                raise ValueError(f"cache labels {cached_names} do not match runtime labels {LABELS}")

            evaluated_weights = weights if weights is not None else self.model_manager.global_weights
            report = evaluate(features, labels, evaluated_weights)
            report.update({
                "schema_version": 1,
                "task": "single_label_multiclass",
                "decision_rule": "argmax",
                "labels": list(LABELS),
                "available": True,
                "model_version": model_version if model_version is not None else self.model_manager.current_version,
            })
            if persist:
                eval_path = os.path.join(server_dir, "output", "latest_eval.json")
                atomic_write_json(eval_path, report)
            logging.info(
                "Held-out eval -> loss=%.4f macro_f1=%.4f accuracy=%.4f",
                report["loss"],
                report["macro_f1"],
                report["accuracy"],
            )
            return report
        except Exception as exc:
            logging.error("Held-out evaluation failed: %s", exc)
            return {"available": False}

    async def on_client_disconnected(self, client_id: str) -> None:
        if client_id in self.round_participants:
            self.round_participants.discard(client_id)
            logging.info(f"Client {client_id[:8]} disconnected. Removed from round participants. Remaining: {len(self.round_participants)}")
            # Clean up their update if they disconnected before aggregation
            self.client_updates.pop(client_id, None)
            self.client_metadata.pop(client_id, None)
            
            # If we now have enough updates from the remaining participants, aggregate!
            expected = len(self.round_participants)
            responses = len(self.client_updates) + len(self.skipped_clients)
            if self.is_training and expected > 0 and responses >= expected:
                if len(self.client_updates) >= self.config.min_clients:
                    asyncio.create_task(self._aggregate_and_advance())
                else:
                    asyncio.create_task(self._stop_for_insufficient_updates())

