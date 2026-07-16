from fastapi import WebSocket
from typing import Dict, Set
import json
import logging

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.online_clients: Set[str] = set()

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.online_clients.add(client_id)
        logging.info(f"WS connected: {client_id}")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self.online_clients.discard(client_id)
        logging.info(f"WS disconnected: {client_id}")

    async def send_personal_message(self, message: dict, client_id: str):
        ws = self.active_connections.get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        dead = []
        for client_id, ws in self.active_connections.items():
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(client_id)
        for cid in dead:
            self.disconnect(cid)

ws_manager = ConnectionManager()
