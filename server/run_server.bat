@echo off
echo =========================================
echo FGT - Federated Gallery Tags Server
echo =========================================

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt >nul 2>nul

echo.
echo Server will be available on LAN IP:
python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()"

echo.
echo Opening Dashboard...
start http://localhost:8000/dashboard/

echo Starting Uvicorn...
uvicorn main:app --host 0.0.0.0 --port 8000
