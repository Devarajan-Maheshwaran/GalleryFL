import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import uuid
import secrets
import numpy as np
import socket
import base64
import json
from typing import Optional, List

from config import ServerConfig
from metrics import MetricsStore
from model_manager import ModelManager
from fl_coordinator import FLCoordinator
from ws_manager import ws_manager
from security import validate_update, AuditLog, TrustScorer, RateLimiter, generate_access_code
from taxonomy_parser import TaxonomyParser
from tag_demand import TagDemandStore

# Additive product feature: recency-weighted demand over taxonomy tags.
# Clients POST the tags they actually use (applied while organizing, or predicted
# with high confidence). This store powers the dashboard's Trending Tags view and
# is fully decoupled from the FL training loop, aggregation, and WebSocket stack.
tag_demand = TagDemandStore()
try:
    _TAX = TaxonomyParser()
    # Detailed leaf tags remain valid demand signals; parent labels are the
    # seven outputs of the deployed classifier.
    _VALID_TAGS = set(_TAX.leaf_names) | set(_TAX.model_labels)
except Exception:
    _TAX = None
    _VALID_TAGS = set()

class TrainingStartRequest(BaseModel):
    min_clients: Optional[int] = None
    max_rounds: Optional[int] = None
    local_epochs: Optional[int] = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = FastAPI(title="FGT Server API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

config = ServerConfig.load_or_default()
metrics_store = MetricsStore()
model_manager = ModelManager()
audit_log = AuditLog()
trust_scorer = TrustScorer()
rate_limiter = RateLimiter()
coordinator = FLCoordinator(config, metrics_store, model_manager, trust_scorer)
historical_norms: list = []

def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@app.on_event("startup")
async def startup_event():
    lan_ip = get_lan_ip()
    port = config.port

    # Start UDP Broadcast Discovery Listener in a daemon thread
    import socket
    import threading
    def listen_udp():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('', 8002))
            while True:
                data, addr = sock.recvfrom(1024)
                if data == b"FGT_DISCOVER":
                    response = f"FGT_OFFER|http://{lan_ip}:{port}".encode('utf-8')
                    sock.sendto(response, addr)
        except Exception as e:
            logging.error(f"UDP listener error: {e}")
        finally:
            sock.close()
            
    threading.Thread(target=listen_udp, daemon=True).start()

    # Start background cleanup task for inactive clients (heartbeat timeout)
    import time
    async def cleanup_offline_clients():
        while True:
            await asyncio.sleep(10)
            now = time.time()
            to_disconnect = []
            for cid, meta in list(coordinator.registered_clients.items()):
                if cid == "dashboard":
                    continue
                # If they haven't sent a heartbeat in 30 seconds, clean them up
                if now - meta.get("last_heartbeat", 0) > 30:
                    to_disconnect.append(cid)
            
            for cid in to_disconnect:
                logging.info(f"Cleaning up inactive client {cid[:8]} due to heartbeat timeout")
                ws = ws_manager.active_connections.get(cid)
                if ws:
                    try:
                        await ws.close(code=4000, reason="Heartbeat timeout")
                    except Exception:
                        pass
                ws_manager.disconnect(cid)
                await coordinator.on_client_disconnected(cid)
                coordinator.registered_clients.pop(cid, None)
                await ws_manager.broadcast({"type": "client_disconnected", "data": {"client_id": cid}})
                
    asyncio.create_task(cleanup_offline_clients())

    logging.info("=========================================")
    logging.info("          FGT SERVER STARTUP             ")
    logging.info("=========================================")
    logging.info(f"LAN IP: {lan_ip}")
    logging.info(f"Port: {port}")
    logging.info(f"FGT Access Code: {config.server_token}")
    logging.info("=========================================")
    # Populate a real current-model baseline for the dashboard at startup.
    await asyncio.to_thread(coordinator._run_evaluation)

class RegisterRequest(BaseModel):
    device_model: str
    nickname: str
    client_id: Optional[str] = None

class ClientUpdateRequest(BaseModel):
    client_id: str
    weights: str
    num_samples: int
    human_labeled_samples: int = 0
    pseudo_labeled_samples: int = 0
    local_loss: float
    local_accuracy: float
    round: int
    base_model_version: int


class ClientSkipRequest(BaseModel):
    client_id: str
    round: int
    base_model_version: int
    reason: str = "insufficient_usable_examples"

async def verify_token(x_fgt_token: Optional[str] = Header(None)):
    if x_fgt_token != config.server_token:
        raise HTTPException(status_code=401, detail="Invalid or missing access token")

@app.post("/api/register", dependencies=[Depends(verify_token)])
async def register_client(req: RegisterRequest):
    client_id = req.client_id if req.client_id else str(uuid.uuid4())
    coordinator.register_client(client_id, req.model_dump())
    audit_log.log("client_registered", {"client_id": client_id, "nickname": req.nickname})
    await ws_manager.broadcast({"type": "client_connected", "data": {"client_id": client_id, "nickname": req.nickname, "device_model": req.device_model}})
    return {"client_id": client_id, "status": "registered", "model_version": model_manager.current_version}

@app.get("/api/model/current")
async def get_current_model(client_version: int = 0):
    if client_version == model_manager.current_version - 1 and hasattr(model_manager, 'last_delta'):
        return Response(
            content=model_manager.get_serialized_delta(), 
            media_type="text/plain", 
            headers={"X-Model-Format": "delta", "X-Model-Version": str(model_manager.current_version)}
        )
    return Response(
        content=model_manager.get_serialized_weights(), 
        media_type="text/plain",
        headers={"X-Model-Format": "full", "X-Model-Version": str(model_manager.current_version)}
    )

@app.get("/api/model/status")
async def model_status():
    """Lightweight liveness/availability check the app can poll on first launch
    (or before Scan & Group) to detect an unsynced / unloaded model and show a
    'Model is not synced' prompt instead of failing silently."""
    head_present = (
        os.path.exists(os.path.join(model_manager.model_dir, "head_weights.npz"))
        or os.path.exists(os.path.join(model_manager.model_dir, "initial_head_weights.npz"))
    )
    return {
        "loaded": len(model_manager.global_weights) > 0,
        "model_version": model_manager.current_version,
        "head_present": head_present,
        "min_clients": config.min_clients,
    }


@app.get("/api/model/schema")
async def get_model_schema():
    """Canonical, versioned tensor and serialization contract for FL clients."""
    return model_manager.get_schema() | {"model_version": model_manager.current_version}

@app.post("/api/training/submit-update", dependencies=[Depends(verify_token)])
async def submit_update(req: ClientUpdateRequest):
    if req.client_id not in coordinator.registered_clients:
        raise HTTPException(status_code=403, detail="Client must register before submitting updates")
    if req.num_samples <= 0:
        raise HTTPException(status_code=400, detail="num_samples must be positive")
    if req.human_labeled_samples < 0 or req.pseudo_labeled_samples < 0:
        raise HTTPException(status_code=400, detail="label-source counts must be non-negative")
    if req.human_labeled_samples + req.pseudo_labeled_samples != req.num_samples:
        raise HTTPException(status_code=400, detail="label-source counts must sum to num_samples")
    if not rate_limiter.check(req.client_id, req.round):
        raise HTTPException(status_code=429, detail="Rate limit exceeded: too many updates for the same round")

    try:
        weights = model_manager.deserialize_weights(req.weights)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Weight decode failed: {str(e)}")

    # Anomaly detection must run on the DELTA (submitted - current global), not
    # on the full weight vector. The full-vector norm is dominated by the
    # near-invariant global head and has ~zero variance across rounds, which made
    # the 3-sigma gate reject every legitimate update after the first two rounds.
    delta = [w - g for w, g in zip(weights, model_manager.global_weights)]
    update_norm = float(sum(np.linalg.norm(d) for d in delta))
    is_valid, reason = validate_update(req.client_id, weights, model_manager.global_weights, update_norm, historical_norms)
    if not is_valid:
        trust_scorer.update_score(req.client_id, False, reason)
        audit_log.log("update_rejected", {"client_id": req.client_id, "reason": reason})
        logging.warning(f"Update REJECTED from client {req.client_id[:8]}: {reason}")
        raise HTTPException(status_code=400, detail=f"Update rejected: {reason}")

    trust_scorer.update_score(req.client_id, True)
    historical_norms.append(update_norm)
    audit_log.log("update_accepted", {"client_id": req.client_id, "norm": update_norm, "samples": req.num_samples})

    logging.info(f"Update ACCEPTED from client {req.client_id[:8]} (samples={req.num_samples}, loss={req.local_loss:.4f}, accuracy={req.local_accuracy:.4f})")

    accepted = coordinator.submit_client_update(
        req.client_id,
        weights,
        {
            "num_samples": req.num_samples,
            "human_labeled_samples": req.human_labeled_samples,
            "pseudo_labeled_samples": req.pseudo_labeled_samples,
            "local_loss": req.local_loss,
            "local_accuracy": req.local_accuracy,
            "round": req.round,
            "base_model_version": req.base_model_version,
        },
    )
    if not accepted:
        raise HTTPException(status_code=409, detail="Update is stale, duplicate, or no training round is active")
    await ws_manager.broadcast({"type": "update_received", "data": {"client_id": req.client_id, "round": coordinator.current_round}})
    return {"status": "accepted"}


@app.post("/api/training/skip-update", dependencies=[Depends(verify_token)])
async def skip_update(req: ClientSkipRequest):
    if req.client_id not in coordinator.registered_clients:
        raise HTTPException(status_code=403, detail="Client must register before responding")
    accepted = coordinator.skip_client_update(
        req.client_id,
        {
            "round": req.round,
            "base_model_version": req.base_model_version,
            "reason": req.reason[:120],
        },
    )
    if not accepted:
        raise HTTPException(status_code=409, detail="Skip response is stale, duplicate, or no round is active")
    return {"status": "skipped"}


@app.get("/api/training/status")
async def get_training_status():
    online_list = []
    for cid in ws_manager.online_clients:
        if cid == "dashboard":
            continue
        meta = coordinator.registered_clients.get(cid, {})
        online_list.append({
            "client_id": cid,
            "nickname": meta.get("nickname", cid[:8]),
            "device_model": meta.get("device_model", "Unknown Device")
        })

    return {
        "is_training": coordinator.is_training,
        "current_round": coordinator.current_round,
        "max_rounds": config.max_rounds,
        "model_version": model_manager.current_version,
        "total_parameters": model_manager.get_total_parameters(),
        "connected_clients": len(online_list),
        "registered_clients": len(coordinator.registered_clients),
        "access_code": f"{get_lan_ip()}:{config.port}@{config.server_token}",
        "access_token": config.server_token,
        "server_address": f"http://{get_lan_ip()}:{config.port}",
        "online_clients": online_list
    }

@app.post("/api/training/regenerate-token")
async def regenerate_token():
    config.server_token = generate_access_code(8)
    config.save()
    
    # Revoke registrations
    coordinator.registered_clients.clear()
    
    # Kick active training client sessions
    to_kick = [cid for cid in ws_manager.online_clients if cid != "dashboard"]
    for cid in to_kick:
        ws = ws_manager.active_connections.get(cid)
        if ws:
            try:
                await ws.close(code=4001, reason="Token regenerated")
            except Exception:
                pass
            ws_manager.disconnect(cid)
            
    await ws_manager.broadcast({"type": "clients_cleared"})
    logging.info(f"Regenerated access token. All old registrations revoked. New FGT Access Code: {config.server_token}")
    return {"access_code": f"{get_lan_ip()}:{config.port}@{config.server_token}", "access_token": config.server_token, "server_address": f"http://{get_lan_ip()}:{config.port}"}

@app.get("/api/metrics/history")
async def get_metrics_history():
    return {"history": [h.model_dump() for h in metrics_store.history]}

@app.get("/api/metrics/leaderboard")
async def get_leaderboard():
    return {"leaderboard": metrics_store.get_leaderboard()}

@app.get("/api/metrics/summary")
async def get_summary():
    return metrics_store.get_summary()

@app.post("/api/training/start")
async def start_training(req: Optional[TrainingStartRequest] = None):
    if coordinator.is_training:
        return {"status": "already_training"}
    if req:
        if req.min_clients is not None:
            if req.min_clients < 2:
                raise HTTPException(status_code=400, detail="Federated training requires at least two clients")
            config.min_clients = req.min_clients
        if req.max_rounds is not None:
            config.max_rounds = max(1, req.max_rounds)
        if req.local_epochs is not None:
            config.local_epochs = max(1, req.local_epochs)
        logging.info(f"Updated training config from dashboard: min_clients={config.min_clients}, max_rounds={config.max_rounds}, local_epochs={config.local_epochs}")
    # Fresh anomaly-detection baseline for this session. The 3-sigma gate
    # compares each round's delta norm against recent history; carrying history
    # across sessions (or a client-strategy change, e.g. switching from a tiny
    # fixed clip to adaptive clipping) would reject legitimate larger deltas as
    # "anomalies". Clearing here keeps the gate meaningful without blocking
    # valid updates.
    historical_norms.clear()
    asyncio.create_task(coordinator.start_training())
    return {"status": "started"}

@app.get("/api/taxonomy")
async def get_taxonomy():
    """Live taxonomy (categories + leaf tags) so the dashboard and clients can
    render the tag system from the server's source of truth rather than a
    hardcoded UI list."""
    tax_path = "taxonomy.json"
    if not os.path.exists(tax_path):
        return {"categories": []}
    try:
        with open(tax_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"categories": []}


class TagSignalRequest(BaseModel):
    client_id: Optional[str] = None
    tags: List[str] = []


@app.post("/api/taxonomy/signal", dependencies=[Depends(verify_token)])
async def post_tag_signal(req: TagSignalRequest):
    """Clients report the tags they actually use (applied while organizing, or
    predicted with high confidence). The server accumulates a recency-weighted
    demand signal that drives the dashboard's Trending Tags view and can later
    bias taxonomy ordering. Unknown tags are ignored; this is additive and does
    not touch the FL training loop."""
    incoming = req.tags or []
    if _VALID_TAGS:
        accepted = [t for t in incoming if t in _VALID_TAGS]
    else:
        accepted = [t for t in incoming if isinstance(t, str) and t]
    recorded = tag_demand.record(accepted, client_id=req.client_id)
    return {"ok": True, "received": len(incoming), "recorded": recorded, "client_id": req.client_id}


@app.get("/api/taxonomy/demand")
async def get_tag_demand(client_id: Optional[str] = None):
    """Recency-weighted demand over the taxonomy: which tags are hot right now.

    Pass `?client_id=...` to get that client's *personal* demand histogram
    (the Non-IID personalisation view) instead of the global aggregate.
    """
    return tag_demand.snapshot(client_id=client_id)


@app.get("/api/config")
async def get_runtime_config():
    """Read-only view of the coordinator's active training / privacy knobs.
    Surfaced by the dashboard so operators can see the live aggregation and
    differential-privacy configuration."""
    num_classes = None
    try:
        num_classes = TaxonomyParser().num_classes
    except Exception:
        num_classes = None
    return {
        "min_clients": config.min_clients,
        "max_rounds": config.max_rounds,
        "local_epochs": config.local_epochs,
        "learning_rate": config.learning_rate,
        "mu": config.mu,
        "aggregation": "clipped_sample_weighted_fedavg",
        "dp_epsilon_per_round": config.dp_epsilon,
        "dp_delta_per_round": config.dp_delta,
        "max_grad_norm": config.max_grad_norm,
        "pseudo_label_threshold": config.pseudo_label_threshold,
        "pseudo_label_weight": config.pseudo_label_weight,
        "min_local_samples": config.min_local_samples,
        "server_delta_clip_norm": config.server_delta_clip_norm,
        "min_unlabeled_f1_improvement": config.min_unlabeled_f1_improvement,
        "max_global_f1_drop": config.max_global_f1_drop,
        "num_classes": num_classes,
    }


@app.get("/api/metrics/comparison")
async def get_comparison():
    categories = []
    
    import json as json_mod
    tax_path = "taxonomy.json"
    if os.path.exists(tax_path):
        try:
            with open(tax_path, "r") as tf:
                tax = json_mod.load(tf)
                categories = [c["id"] for c in tax.get("categories", [])]
        except Exception:
            pass
            
    if not categories:
        categories = ["people", "places", "activities", "objects", "documents", "nature", "events"]

    def compute_scores(eval_file):
        scores = [0.0] * len(categories)
        if os.path.exists(eval_file) and os.path.exists(tax_path):
            try:
                with open(eval_file, "r") as f:
                    data = json_mod.load(f)
                with open(tax_path, "r") as tf:
                    tax = json_mod.load(tf)
                
                for i, cat in enumerate(categories):
                    # Current reports are seven-parent single-label reports.
                    parent_metrics = data.get("per_class", {}).get(cat)
                    if isinstance(parent_metrics, dict):
                        scores[i] = float(parent_metrics.get("f1", 0.0))
                        continue
                    # Compatibility only for historical 34-leaf reports.
                    cat_node = next((c for c in tax.get("categories", []) if c["id"] == cat), None)
                    if cat_node and "children" in cat_node:
                        leaf_names = cat_node["children"]
                        cat_f1s = [data.get("per_class", {}).get(name, {}).get("f1", 0.0) for name in leaf_names]
                        scores[i] = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0.0
            except Exception as e:
                logging.error(f"Failed to read real evaluation metrics from {eval_file}: {e}")
        return scores

    baseline_path = "models/baseline_metrics.json"
    federated_path = "output/latest_eval.json"
    baseline = compute_scores(baseline_path)
    federated = compute_scores(federated_path)

    evaluated = os.path.exists(baseline_path) and os.path.exists(federated_path)

    return {
        "categories": categories,
        "baseline": baseline,
        "federated": federated,
        # Honest signal for the dashboard: until the runtime evaluator has produced a current held-out report, the UI should show "awaiting evaluation" rather than fake 0s.
        "evaluated": evaluated
    }

@app.post("/api/training/stop")
async def stop_training():
    if not coordinator.is_training:
        return {"status": "not_training"}
    coordinator.is_training = False
    await ws_manager.broadcast({"type": "training_complete", "data": {"reason": "stopped_by_user", "final_round": coordinator.current_round}})
    return {"status": "stopped"}

@app.get("/api/export/model")
async def export_model():
    model_path = os.path.join(model_manager.model_dir, "base_model.tflite")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(model_path, media_type="application/octet-stream", filename="fgt_model.tflite")

@app.get("/api/export/report")
async def export_report():
    return JSONResponse(content=metrics_store.generate_report())

@app.websocket("/ws/feed")
async def websocket_endpoint(websocket: WebSocket):
    client_id = websocket.query_params.get("client_id", f"anon_{id(websocket)}")
    token = websocket.headers.get("x-fgt-token")
    is_dashboard = client_id == "dashboard"
    if not is_dashboard and (token != config.server_token or client_id not in coordinator.registered_clients):
        await websocket.close(code=1008, reason="Register with a valid access code before opening WebSocket")
        return

    await ws_manager.connect(client_id, websocket)
    if not is_dashboard:
        await coordinator.on_client_reconnected(client_id)
    try:
        while True:
            raw_message = await websocket.receive_text()
            if not is_dashboard:
                try:
                    message = json.loads(raw_message)
                    if message.get("type") == "heartbeat":
                        coordinator.record_heartbeat(client_id)
                except (TypeError, ValueError):
                    logging.debug("Ignoring malformed WebSocket message from %s", client_id[:8])
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
        await coordinator.on_client_disconnected(client_id)
        await ws_manager.broadcast({"type": "client_disconnected", "data": {"client_id": client_id}})
