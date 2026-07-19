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
from typing import Optional

from config import ServerConfig
from metrics import MetricsStore
from model_manager import ModelManager
from fl_coordinator import FLCoordinator
from ws_manager import ws_manager
from security import validate_update, AuditLog, TrustScorer, RateLimiter

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

def encrypt_access_code(url: str, token: str) -> str:
    raw_bytes = f"{url}|{token}".encode('utf-8')
    key_bytes = "FGT-SECURE-KEY-2026".encode('utf-8')
    xor_bytes = bytearray(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(raw_bytes))
    return base64.b64encode(xor_bytes).decode('utf-8')

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
    encoded_code = encrypt_access_code(f"http://{lan_ip}:{port}", config.server_token)
    
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
    logging.info(f"Backup Full Code: {encoded_code}")
    logging.info("=========================================")

class RegisterRequest(BaseModel):
    device_model: str
    nickname: str
    client_id: Optional[str] = None

class ClientUpdateRequest(BaseModel):
    client_id: str
    weights: str
    num_samples: int
    local_loss: float
    local_accuracy: float
    round: int
    base_model_version: int

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
    if not rate_limiter.check(req.client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded: max 1 update per 5 seconds")

    try:
        weights = model_manager.deserialize_weights(req.weights)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Weight decode failed: {str(e)}")

    update_norm = float(sum(np.linalg.norm(w) for w in weights))
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
        req.client_id, weights,
        {"num_samples": req.num_samples, "local_loss": req.local_loss, "local_accuracy": req.local_accuracy,
         "round": req.round, "base_model_version": req.base_model_version}
    )
    if not accepted:
        raise HTTPException(status_code=409, detail="Update is stale, duplicate, or no training round is active")
    await ws_manager.broadcast({"type": "update_received", "data": {"client_id": req.client_id, "round": coordinator.current_round}})
    return {"status": "accepted"}

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
        "online_clients": online_list
    }

@app.post("/api/training/regenerate-token")
async def regenerate_token():
    config.server_token = secrets.token_hex(4)
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
    return {"access_code": f"{get_lan_ip()}:{config.port}@{config.server_token}"}

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
            config.min_clients = req.min_clients
        if req.max_rounds is not None:
            config.max_rounds = req.max_rounds
        if req.local_epochs is not None:
            config.local_epochs = req.local_epochs
        logging.info(f"Updated training config from dashboard: min_clients={config.min_clients}, max_rounds={config.max_rounds}, local_epochs={config.local_epochs}")
    asyncio.create_task(coordinator.start_training())
    return {"status": "started"}

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
                    cat_node = next((c for c in tax.get("categories", []) if c["id"] == cat), None)
                    if cat_node and "children" in cat_node:
                        leaf_names = cat_node["children"]
                        cat_f1s = [data.get("per_class", {}).get(name, {}).get("f1", 0.0) for name in leaf_names]
                        scores[i] = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0.0
            except Exception as e:
                logging.error(f"Failed to read real evaluation metrics from {eval_file}: {e}")
        return scores

    baseline = compute_scores("output/bootstrap_eval_report.json")
    federated = compute_scores("output/latest_eval.json")
    
    return {
        "categories": categories,
        "baseline": baseline,
        "federated": federated
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
