"""Windows executable entry point for the GalleryFL central aggregator."""

from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import shutil
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "GalleryFL Aggregator"
MUTEX_NAME = "Local\\GalleryFLAggregator.SingleInstance"
MODEL_FILES = (
    "base_model.tflite",
    "baseline_metrics.json",
    "head_weights.npz",
    "initial_head_weights.npz",
    "model_schema.json",
    "model_version.txt",
    "thresholds.json",
    "validation_features.npz",
)


def _asset_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()


def _data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / "GalleryFL" / "Aggregator"


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".new")
    shutil.copy2(source, temp)
    os.replace(temp, destination)


def _read_version(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _prepare_runtime(asset_dir: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "output").mkdir(parents=True, exist_ok=True)
    bundled_models = asset_dir / "models"
    state_models = data_dir / "models"
    state_models.mkdir(parents=True, exist_ok=True)

    config_source = asset_dir / "config.json"
    config_target = data_dir / "config.json"
    if not config_target.exists():
        _atomic_copy(config_source, config_target)

    bundled_version = _read_version(bundled_models / "model_version.txt")
    state_version = _read_version(state_models / "model_version.txt")
    promote_bundled_head = not (state_models / "head_weights.npz").exists() or bundled_version > state_version

    if promote_bundled_head and (state_models / "head_weights.npz").exists():
        backup_dir = data_dir / "backups" / time.strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in ("head_weights.npz", "model_version.txt", "model_schema.json"):
            existing = state_models / name
            if existing.exists():
                shutil.copy2(existing, backup_dir / name)

    # Immutable validation/backbone artifacts are refreshed from the signed ZIP.
    always_refresh = {
        "base_model.tflite",
        "baseline_metrics.json",
        "initial_head_weights.npz",
        "model_schema.json",
        "thresholds.json",
        "validation_features.npz",
    }
    for name in MODEL_FILES:
        source = bundled_models / name
        destination = state_models / name
        if not source.is_file():
            raise FileNotFoundError(f"Bundled model artifact is missing: {name}")
        if name in always_refresh or promote_bundled_head:
            _atomic_copy(source, destination)


def _acquire_single_instance() -> object:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        raise OSError("Could not create the GalleryFL single-instance mutex")
    if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        webbrowser.open("http://localhost:8000/dashboard/")
        ctypes.windll.user32.MessageBoxW(
            None,
            "GalleryFL Aggregator is already running. The dashboard was opened in your browser.",
            APP_NAME,
            0x40,
        )
        raise SystemExit(0)
    return handle


def _open_dashboard_when_ready(port: int) -> None:
    url = f"http://127.0.0.1:{port}/api/model/status"
    for _ in range(120):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(f"http://localhost:{port}/dashboard/")
                    return
        except Exception:
            time.sleep(0.25)


def main() -> int:
    multiprocessing.freeze_support()
    if os.name != "nt":
        raise RuntimeError("GalleryFL-Aggregator.exe is a Windows-only release")
    ctypes.windll.kernel32.SetConsoleTitleW(APP_NAME)
    mutex = _acquire_single_instance()
    del mutex  # The Windows handle remains valid for the process lifetime.

    assets = _asset_dir()
    data = _data_dir()
    _prepare_runtime(assets, data)
    os.environ["FGT_ASSET_DIR"] = str(assets)
    os.environ["FGT_DATA_DIR"] = str(data)
    os.environ["FGT_MODEL_DIR"] = str(data / "models")
    os.environ["FGT_CONFIG_PATH"] = str(data / "config.json")
    os.environ["FGT_PERSIST_CONFIG"] = "1"
    os.chdir(data)

    from config import ServerConfig
    from main import app
    import uvicorn

    config = ServerConfig.load_or_default()
    print("=" * 61)
    print(" GalleryFL Central Aggregator")
    print("=" * 61)
    print(f" Data directory : {data}")
    print(f" Dashboard      : http://localhost:{config.port}/dashboard/")
    print(f" LAN bind       : {config.host}:{config.port}")
    print(" Keep this window open. Press Ctrl+C to stop the server.")
    print("=" * 61)
    if os.environ.get("FGT_NO_BROWSER") != "1":
        threading.Thread(target=_open_dashboard_when_ready, args=(config.port,), daemon=True).start()
    uvicorn.run(app, host=config.host, port=config.port, reload=False, log_level="info")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        message = f"GalleryFL Aggregator failed to start:\n\n{exc}"
        print(message, file=sys.stderr)
        if os.name == "nt":
            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
        raise
