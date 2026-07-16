from fastapi import WebSocket
from typing import Dict, Set
import json
import logging

class ConnectionManager:
    def __init__(self):
        # Maps client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        # Track clients that are currently online
        self.online_clients: Set[str] = set()

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.online_clients.add(client_id)
        logging.info(f"Client {client_id} connected via WS")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.online_clients:
            self.online_clients.remove(client_id)
        logging.info(f"Client {client_id} disconnected from WS")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Failed to send to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        dead_clients = []
        payload = json.dumps(message)
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(payload)
            except Exception as e:
                logging.error(f"Failed to broadcast to {client_id}: {e}")
                dead_clients.append(client_id)
                
        for cid in dead_clients:
            self.disconnect(cid)

ws_manager = ConnectionManager()
