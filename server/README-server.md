# FGT Server

This is the server component for **FGT - Federated Gallery Tags**.

## Quick Start

### Windows
Double-click `run_server.bat` or run it from the command line:
```cmd
run_server.bat
```

### Linux / macOS
Make the script executable and run it:
```bash
chmod +x run_server.sh
./run_server.sh
```

### What it does
The startup script will automatically:
1. Create a Python virtual environment (`venv`).
2. Install all required dependencies from `requirements.txt`.
3. Print your local LAN IP address (needed for the Android app).
4. Start the dashboard in your default web browser.
5. Launch the backend FastAPI server on port 8000.

## Folder Structure
- `dashboard/`: Frontend static files for the dashboard.
- `models/`: Weights, schema, and current state for federated learning.
- `output/`: Any generated exports or reports.
- `taxonomy.json`: Configures the classification categories used by the model.
