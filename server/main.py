from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import uuid
import numpy as np
from typing import Dict

from config import ServerConfig
from metrics import MetricsStore
from model_manager import ModelManager
from fl_coordinator import FLCoordinator
from ws_manager import ws_manager
from security import validate_update, AuditLog, TrustScorer

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
coordinator = FLCoordinator(config, metrics_store, model_manager)
audit_log = AuditLog()
trust_scorer = TrustScorer()
historical_norms: list = []

class RegisterRequest(BaseModel):
    device_model: str
    nickname: str

class ClientUpdateRequest(BaseModel):
    client_id: str
    weights: str
    num_samples: int
    local_loss: float
    local_accuracy: float

@app.post("/api/register")
async def register_client(req: RegisterRequest):
    client_id = str(uuid.uuid4())
    coordinator.register_client(client_id, req.model_dump())
    audit_log.log("client_registered", {"client_id": client_id, "nickname": req.nickname})
    await ws_manager.broadcast({"type": "client_connected", "data": {"client_id": client_id, "nickname": req.nickname, "device_model": req.device_model}})
    return {"client_id": client_id, "status": "registered", "model_version": model_manager.current_version}

@app.get("/api/model/current")
async def get_current_model():
    return Response(content=model_manager.get_serialized_weights(), media_type="text/plain")

@app.post("/api/training/submit-update")
async def submit_update(req: ClientUpdateRequest):
    shapes = model_manager.get_weight_shapes()
    try:
        weights = model_manager.deserialize_weights(req.weights, shapes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Weight decode failed: {str(e)}")

    update_norm = float(sum(np.linalg.norm(w) for w in weights))
    is_valid, reason = validate_update(req.client_id, weights, model_manager.global_weights, update_norm, historical_norms)
    if not is_valid:
        trust_scorer.update_score(req.client_id, False, reason)
        audit_log.log("update_rejected", {"client_id": req.client_id, "reason": reason})
        raise HTTPException(status_code=400, detail=f"Update rejected: {reason}")

    trust_scorer.update_score(req.client_id, True)
    historical_norms.append(update_norm)
    audit_log.log("update_accepted", {"client_id": req.client_id, "norm": update_norm, "samples": req.num_samples})

    coordinator.submit_client_update(
        req.client_id, weights,
        {"num_samples": req.num_samples, "local_loss": req.local_loss, "local_accuracy": req.local_accuracy}
    )
    await ws_manager.broadcast({"type": "update_received", "data": {"client_id": req.client_id, "round": coordinator.current_round}})
    return {"status": "accepted"}

@app.get("/api/training/status")
async def get_training_status():
    return {
        "is_training": coordinator.is_training,
        "current_round": coordinator.current_round,
        "max_rounds": config.max_rounds,
        "model_version": model_manager.current_version,
        "total_parameters": model_manager.get_total_parameters(),
        "connected_clients": len(ws_manager.online_clients),
        "registered_clients": len(coordinator.registered_clients),
    }

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
async def start_training():
    if coordinator.is_training:
        return {"status": "already_training"}
    import asyncio
    asyncio.create_task(coordinator.start_training())
    return {"status": "started"}

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
    await ws_manager.connect(client_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
        await ws_manager.broadcast({"type": "client_disconnected", "data": {"client_id": client_id}})
