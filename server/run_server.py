import uvicorn
from config import ServerConfig
import os

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    config = ServerConfig.load_or_default()
    uvicorn.run("main:app", host=config.host, port=config.port, reload=True)
