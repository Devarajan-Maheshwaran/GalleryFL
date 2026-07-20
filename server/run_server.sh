#!/bin/bash
cd "$(dirname "$0")"
echo "========================================="
echo "FGT - Federated Gallery Tags Server"
echo "========================================="

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt > /dev/null 2>&1

echo ""
echo "Server will be available on LAN IP:"
hostname -I 2>/dev/null || ip route get 1 2>/dev/null | awk '{print $7}' || python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()"

echo ""
echo "Starting Dashboard in browser..."
if command -v xdg-open > /dev/null; then
  xdg-open http://localhost:8000/dashboard/ &
elif command -v open > /dev/null; then
  open http://localhost:8000/dashboard/ &
fi

echo "Starting Uvicorn..."
export FGT_PERSIST_CONFIG=1
python run_server.py
