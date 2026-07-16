import asyncio
import websockets
import httpx
import json
import numpy as np
import base64
import zlib
import struct
import argparse
import time

class MockFLClient:
    def __init__(self, server_url="http://localhost:8080", nickname="MockClient"):
        self.server_url = server_url
        self.ws_url = server_url.replace("http", "ws").replace("https", "wss") + "/ws/feed"
        self.nickname = nickname
        self.client_id = None
        self.model_version = 0
        
        # Mock weights shape based on MobileNetV3Large (960) -> 256 -> 20 classes
        self.shapes = [(960, 256), (256,), (256, 20), (20,)]
        self.local_weights = [np.zeros(shape, dtype=np.float32) for shape in self.shapes]

    async def register(self):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/api/register",
                json={"device_model": "Mock Python Client", "nickname": self.nickname}
            )
            data = resp.json()
            self.client_id = data["client_id"]
            self.model_version = data["model_version"]
            print(f"[{self.nickname}] Registered with ID: {self.client_id}")

    def serialize_weights(self) -> str:
        byte_chunks = []
        for layer in self.local_weights:
            flat = layer.flatten()
            fmt = f'<{len(flat)}f'
            layer_bytes = struct.pack(fmt, *flat)
            size_bytes = struct.pack('<I', len(layer_bytes))
            byte_chunks.append(size_bytes + layer_bytes)
        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined)
        return base64.b64encode(compressed).decode('ascii')
        
    async def get_current_model(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.server_url}/api/model/current")
            # In a real app we'd decode this, but for the mock we just need to know it works
            print(f"[{self.nickname}] Downloaded current model weights ({len(resp.content)} bytes)")
            
    async def submit_update(self, round_num: int):
        # Simulate local training by adding some small noise
        self.local_weights = [w + np.random.normal(0, 0.01, size=w.shape).astype(np.float32) for w in self.local_weights]
        
        encoded = self.serialize_weights()
        
        # Simulated metrics
        loss = float(np.random.uniform(0.1, 0.5))
        acc = float(np.random.uniform(0.7, 0.95))
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/api/training/submit-update",
                json={
                    "client_id": self.client_id,
                    "weights": encoded,
                    "num_samples": 50,
                    "local_loss": loss,
                    "local_accuracy": acc
                }
            )
            print(f"[{self.nickname}] Submitted update for round {round_num}. Status: {resp.status_code}")

    async def connect_ws(self):
        ws_endpoint = f"{self.ws_url}?client_id={self.client_id}"
        async with websockets.connect(ws_endpoint) as ws:
            print(f"[{self.nickname}] Connected to WebSocket")
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    event_type = data.get("type")
                    
                    if event_type == "update_requested":
                        round_num = data["data"]["round"]
                        print(f"[{self.nickname}] Server requested update for round {round_num}")
                        
                        await self.get_current_model()
                        
                        # Simulate training time
                        await asyncio.sleep(2) 
                        
                        await self.submit_update(round_num)
                        
                    elif event_type == "round_completed":
                        print(f"[{self.nickname}] Round {data['data']['round']} completed globally")
                        
                    elif event_type == "training_complete":
                        print(f"[{self.nickname}] Global training session complete!")
                        
            except websockets.exceptions.ConnectionClosed:
                print(f"[{self.nickname}] WebSocket connection closed")

    async def run(self):
        await self.register()
        await self.connect_ws()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2, help="Number of mock clients")
    args = parser.parse_args()
    
    clients = [MockFLClient(nickname=f"MockClient-{i+1}") for i in range(args.count)]
    tasks = [asyncio.create_task(client.run()) for client in clients]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
