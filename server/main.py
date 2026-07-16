from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, Any

from config import ServerConfig
from metrics import MetricsStore
from model_manager import ModelManager
from fl_coordinator import FLCoordinator
from ws_manager import ws_manager

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = FastAPI(title="FGT Server API")
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

config = ServerConfig.load_or_default()
metrics_store = MetricsStore()
model_manager = ModelManager()
coordinator = FLCoordinator(config, metrics_store, model_manager)

class RegisterRequest(BaseModel):
    device_model: str
    nickname: str

class ClientUpdateRequest(BaseModel):
    client_id: str
    weights: str # base64
    num_samples: int
    local_loss: float
    local_accuracy: float

@app.post("/api/register")
async def register_client(req: RegisterRequest):
    import uuid
    client_id = str(uuid.uuid4())
    metadata = req.model_dump()
    coordinator.register_client(client_id, metadata)
    return {"client_id": client_id, "status": "registered", "model_version": model_manager.current_version}

@app.get("/api/model/current")
async def get_current_model():
    weights_str = model_manager.get_serialized_weights()
    return Response(content=weights_str, media_type="text/plain")

@app.post("/api/training/submit-update")
async def submit_update(req: ClientUpdateRequest):
    # Determine shape of weights needed based on current model shapes
    shapes = [(w.shape) for w in model_manager.global_weights]
    try:
        weights = model_manager.deserialize_weights(req.weights, shapes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode weights: {str(e)}")
        
    coordinator.submit_client_update(
        req.client_id, 
        weights, 
        {"num_samples": req.num_samples, "local_loss": req.local_loss, "local_accuracy": req.local_accuracy}
    )
    return {"status": "accepted"}

@app.get("/api/training/status")
async def get_training_status():
    return {
        "is_training": coordinator.is_training,
        "current_round": coordinator.current_round,
        "max_rounds": config.max_rounds,
        "model_version": model_manager.current_version,
        "connected_clients": len(ws_manager.online_clients),
        "registered_clients": len(coordinator.registered_clients)
    }

@app.get("/api/metrics/history")
async def get_metrics_history():
    return {"history": [h.model_dump() for h in metrics_store.history]}

@app.get("/api/metrics/leaderboard")
async def get_leaderboard():
    return {"leaderboard": metrics_store.get_leaderboard()}

@app.post("/api/training/start")
async def start_training():
    if coordinator.is_training:
        return {"status": "already_training"}
    import asyncio
    asyncio.create_task(coordinator.start_training())
    return {"status": "started"}

@app.websocket("/ws/feed")
async def websocket_endpoint(websocket: WebSocket):
    # In a real app, client_id might be passed as a query param or auth token
    client_id = websocket.query_params.get("client_id", f"anon_{id(websocket)}")
    await ws_manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client-sent ws messages if any
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
