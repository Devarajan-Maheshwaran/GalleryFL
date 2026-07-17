import subprocess
import time
import httpx
import sys
import logging
import asyncio

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - Orchestrator - %(message)s")

SERVER_URL = "http://localhost:8080"
TOKEN = "dev-token-change-me"

async def wait_for_server():
    for _ in range(10):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{SERVER_URL}/api/training/status")
                if r.status_code == 200:
                    return True
        except httpx.ConnectError:
            pass
        await asyncio.sleep(1)
    return False

async def main():
    logging.info("Starting FL server...")
    server_process = subprocess.Popen([sys.executable, "run_server.py"])
    client_processes = []
    
    try:
        ready = await wait_for_server()
        if not ready:
            logging.error("Server failed to start in time.")
            sys.exit(1)
            
        logging.info("Server is up. Spawning mock clients...")
        
        client_configs = [
            ("ClientA", "portrait,food,indoor"),
            ("ClientB", "animal,plant,mountain,beach,water"),
            ("ClientC", "vehicle,gadget,clothing")
        ]
        
        client_processes = []
        client_logs = []
        for i, (name, focus) in enumerate(client_configs):
            log_f = open(f"client_{name}.log", "w")
            client_logs.append(log_f)
            p = subprocess.Popen([sys.executable, "mock_client.py", "--name", name, "--focus", focus], stdout=log_f, stderr=subprocess.STDOUT)
            client_processes.append(p)
            
        logging.info("Waiting for clients to register and connect via WS...")
        while True:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{SERVER_URL}/api/training/status")
                if r.status_code == 200:
                    status = r.json()
                    if status["connected_clients"] >= len(client_configs):
                        break
            await asyncio.sleep(2)
            
        logging.info("Starting training session via API...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{SERVER_URL}/api/training/start")
            logging.info(f"Start training response: {resp.text}")
            
        logging.info("Monitoring training status...")
        while True:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{SERVER_URL}/api/training/status")
                if r.status_code == 200:
                    status = r.json()
                    logging.info(f"Round: {status['current_round']}/{status['max_rounds']} | Connected: {status['connected_clients']}")
                    if not status["is_training"] and status["current_round"] > 0:
                        logging.info("Training session completed!")
                        break
            await asyncio.sleep(5)
            
        logging.info("Verifying final model version...")
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{SERVER_URL}/api/training/status")
            status = r.json()
            logging.info(f"Final Model Version: {status['model_version']}")
            assert status['model_version'] > 1, "Model version did not increment!"
            
        logging.info("E2E FL Mock Session Succeeded!")
        
    finally:
        logging.info("Terminating all processes...")
        for p in client_processes:
            p.terminate()
        server_process.terminate()

if __name__ == "__main__":
    asyncio.run(main())
