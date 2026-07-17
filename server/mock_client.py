import asyncio
import httpx
import websockets
import json
import base64
import zlib
import struct
import numpy as np
import logging
import argparse
import sys
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s")

try:
    from taxonomy_parser import TaxonomyParser
    taxonomy = TaxonomyParser("taxonomy.json")
    NUM_CLASSES = taxonomy.num_classes
    LEAF_CLASSES = taxonomy.leaf_names
except Exception as e:
    logging.error(f"Failed to load taxonomy.json: {e}")
    sys.exit(1)

LAYER_SCHEMA = [
    ("w1", (1024, 256)),
    ("b1", (256,)),
    ("w2", (256, NUM_CLASSES)),
    ("b2", (NUM_CLASSES,)),
]

class MockClient:
    def __init__(self, name: str, focus_classes: list, server_url: str = "http://localhost:8080", token: str = "dev-token-change-me"):
        self.name = name
        self.focus_classes = focus_classes
        self.server_url = server_url
        self.ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
        self.token = token
        
        self.client_id = None
        self.logger = logging.getLogger(f"Client-{self.name}")
        
        self.model = self._build_model()
        self.x_train, self.y_train = self._generate_synthetic_data()

    def _build_model(self):
        # Must exactly match the ClassificationHead schema
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1024,)),
            tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid', name='dense_2')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['binary_accuracy'])
        return model

    def _generate_synthetic_data(self):
        num_samples = 50
        x = np.random.randn(num_samples, 1024).astype(np.float32)
        y = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)
        
        # Make the client strongly biased towards its focus classes
        for i in range(num_samples):
            # random noise
            y[i] = np.random.uniform(0, 0.1, NUM_CLASSES)
            # Add strong signal to focus classes
            focus = np.random.choice(self.focus_classes)
            if focus in LEAF_CLASSES:
                idx = LEAF_CLASSES.index(focus)
                y[i, idx] = 1.0
                
        return x, y

    def deserialize_weights(self, encoded: str) -> list:
        compressed = base64.b64decode(encoded)
        combined = zlib.decompress(compressed)

        offset = 0
        weights = []
        for name, shape in LAYER_SCHEMA:
            size = struct.unpack_from('<I', combined, offset)[0]
            offset += 4
            layer = np.frombuffer(combined[offset:offset + size], dtype=np.float32).copy().reshape(shape)
            offset += size
            weights.append(layer)
        return weights

    def serialize_weights(self, weights: list) -> str:
        byte_chunks = []
        for w in weights:
            flat = w.flatten()
            layer_bytes = flat.astype('<f4').tobytes()
            byte_chunks.append(struct.pack('<I', len(layer_bytes)) + layer_bytes)
        combined = b''.join(byte_chunks)
        compressed = zlib.compress(combined, level=6)
        return base64.b64encode(compressed).decode('ascii')

    async def register(self):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/api/register",
                json={"device_model": "MockDevice", "nickname": self.name},
                headers={"X-FGT-Token": self.token}
            )
            if resp.status_code == 200:
                self.client_id = resp.json()["client_id"]
                self.logger.info(f"Registered with client_id: {self.client_id}")
            else:
                self.logger.error(f"Failed to register: {resp.text}")
                sys.exit(1)

    async def get_model(self):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.server_url}/api/model/current")
            if resp.status_code == 200:
                encoded = resp.text
                weights = self.deserialize_weights(encoded)
                self.model.set_weights(weights)
                self.logger.info("Successfully downloaded and applied global weights.")
            else:
                self.logger.error("Failed to download model")

    async def train(self, config):
        epochs = config.get("local_epochs", 1)
        self.logger.info(f"Training for {epochs} epochs...")
        
        # Evaluate before
        loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        
        # Train
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=16, verbose=0)
        
        # Evaluate after
        new_loss, new_acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        self.logger.info(f"Training complete. Loss: {loss:.4f} -> {new_loss:.4f}, Acc: {acc:.4f} -> {new_acc:.4f}")
        
        return new_loss, new_acc

    async def submit_update(self, loss, acc):
        weights = self.model.get_weights()
        encoded = self.serialize_weights(weights)
        
        payload = {
            "client_id": self.client_id,
            "weights": encoded,
            "num_samples": len(self.x_train),
            "local_loss": float(loss),
            "local_accuracy": float(acc)
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/api/training/submit-update",
                json=payload,
                headers={"X-FGT-Token": self.token}
            )
            if resp.status_code == 200:
                self.logger.info("Successfully submitted update.")
            else:
                self.logger.error(f"Failed to submit update: {resp.text}")

    async def connect_and_listen(self):
        ws_endpoint = f"{self.ws_url}/ws/feed?client_id={self.client_id}"
        
        async with websockets.connect(ws_endpoint) as ws:
            self.logger.info("Connected to WebSocket.")
            
            # Announce connection logic here if any, wait for messages
            try:
                while True:
                    msg_text = await ws.recv()
                    msg = json.loads(msg_text)
                    msg_type = msg.get("type")
                    
                    if msg_type == "update_requested":
                        data = msg.get("data", {})
                        round_num = data.get("round", 0)
                        self.logger.info(f"Received update request for Round {round_num}")
                        
                        await self.get_model()
                        loss, acc = await self.train(data.get("config", {}))
                        await self.submit_update(loss, acc)
                        
                    elif msg_type == "training_complete":
                        self.logger.info("Training complete, disconnecting.")
                        break
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.info("WebSocket connection closed.")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--focus", required=True, help="Comma separated list of focus classes")
    args = parser.parse_args()
    
    focus_classes = [c.strip() for c in args.focus.split(",")]
    client = MockClient(name=args.name, focus_classes=focus_classes)
    
    await client.register()
    await client.connect_and_listen()

if __name__ == "__main__":
    asyncio.run(main())
