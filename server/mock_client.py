import asyncio
import websockets
import httpx
import json
import numpy as np
import base64
import zlib
import struct
import argparse

SHAPES = [(960, 256), (256,), (256, 20), (20,)]

def serialize_weights(weights):
    chunks = []
    for layer in weights:
        layer_bytes = layer.astype(np.float32).flatten().tobytes()
        chunks.append(struct.pack('<I', len(layer_bytes)) + layer_bytes)
    return base64.b64encode(zlib.compress(b''.join(chunks), 6)).decode('ascii')

def deserialize_weights(encoded):
    combined = zlib.decompress(base64.b64decode(encoded))
    offset = 0
    weights = []
    for shape in SHAPES:
        size = struct.unpack_from('<I', combined, offset)[0]
        offset += 4
        layer = np.frombuffer(combined[offset:offset + size], dtype=np.float32).copy().reshape(shape)
        offset += size
        weights.append(layer)
    return weights

def local_train(weights, num_samples, lr, mu, epochs):
    trained = []
    for w in weights:
        grad = np.random.randn(*w.shape).astype(np.float32) * 0.01
        prox_term = mu * (w - w)
        updated = w - lr * (grad + prox_term)
        trained.append(updated)

    loss = float(np.random.uniform(0.2, 2.0))
    acc = float(np.clip(1.0 - loss * 0.3 + np.random.uniform(-0.05, 0.05), 0.0, 1.0))
    return trained, loss, acc

class FLClient:
    def __init__(self, server_url, nickname):
        self.server_url = server_url
        self.ws_url = server_url.replace("http", "ws") + "/ws/feed"
        self.nickname = nickname
        self.client_id = None
        self.weights = [np.zeros(s, dtype=np.float32) for s in SHAPES]

    async def register(self):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/api/register",
                json={"device_model": "Python Simulator", "nickname": self.nickname}
            )
            data = resp.json()
            self.client_id = data["client_id"]
            print(f"[{self.nickname}] Registered: {self.client_id[:8]}")

    async def download_model(self):
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.server_url}/api/model/current")
            self.weights = deserialize_weights(resp.text)

    async def submit(self, round_num, trained_weights, loss, acc, num_samples):
        encoded = serialize_weights(trained_weights)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.server_url}/api/training/submit-update",
                json={
                    "client_id": self.client_id,
                    "weights": encoded,
                    "num_samples": num_samples,
                    "local_loss": loss,
                    "local_accuracy": acc
                }
            )
            print(f"[{self.nickname}] Round {round_num}: loss={loss:.4f} acc={acc:.4f} ({resp.status_code})")

    async def run(self):
        await self.register()
        ws_endpoint = f"{self.ws_url}?client_id={self.client_id}"
        async with websockets.connect(ws_endpoint) as ws:
            print(f"[{self.nickname}] WebSocket connected")
            try:
                async for message in ws:
                    data = json.loads(message)
                    event = data.get("type")

                    if event == "update_requested":
                        round_num = data["data"]["round"]
                        config = data["data"].get("config", {})
                        lr = config.get("lr", 0.001)
                        mu = config.get("mu", 0.01)
                        epochs = config.get("local_epochs", 3)

                        await self.download_model()

                        num_samples = np.random.randint(20, 200)
                        trained, loss, acc = local_train(self.weights, num_samples, lr, mu, epochs)
                        await self.submit(round_num, trained, loss, acc, num_samples)

                    elif event == "training_complete":
                        print(f"[{self.nickname}] Training complete")
                        break

            except websockets.exceptions.ConnectionClosed:
                print(f"[{self.nickname}] Disconnected")

async def main():
    parser = argparse.ArgumentParser(description="FGT FL Client Simulator")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--server", type=str, default="http://localhost:8080")
    args = parser.parse_args()

    clients = [FLClient(args.server, f"Client-{i+1}") for i in range(args.count)]
    await asyncio.gather(*[c.run() for c in clients])

if __name__ == "__main__":
    asyncio.run(main())
