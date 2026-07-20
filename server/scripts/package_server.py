"""Build the public, server-only GalleryFL aggregator release ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT_DIR / "server"
OUTPUT_ZIP = ROOT_DIR / "GalleryFL-aggregator.zip"
CHECKSUM_FILE = ROOT_DIR / "GalleryFL-aggregator.zip.sha256"

CORE_FILES = (
    "config.py",
    "fl_coordinator.py",
    "fl_math.py",
    "main.py",
    "metrics.py",
    "model_eval.py",
    "model_manager.py",
    "requirements.txt",
    "START_GALLERYFL_SERVER.bat",
    "START_GALLERYFL_SERVER.ps1",
    "run_server.bat",
    "run_server.py",
    "run_server.sh",
    "security.py",
    "tag_demand.py",
    "taxonomy.json",
    "taxonomy_parser.py",
    "ws_manager.py",
)

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

README_TEXT = """GalleryFL Public Central Aggregator
===================================

This package contains only the central FastAPI federated-learning aggregator,
dashboard, deployed seven-parent model head, frozen TFLite backbone, and the
held-out validation artifact required by the model commit guard. It does not
contain Android source, tests, retraining scripts, credentials, or development
caches.

Requirements
------------
- Python 3.11 or 3.12 recommended
- All participating phones and the server on the same trusted LAN
- At least two connected Android clients for a federated round

Windows — easiest start
-----------------------
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and select
   "Add Python to PATH" during installation.
2. Extract this ZIP completely.
3. Double-click START_GALLERYFL_SERVER.bat.
4. Keep the console window open. The dashboard opens automatically.

PowerShell alternative
----------------------
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\\START_GALLERYFL_SERVER.ps1

Linux/macOS
-----------
  chmod +x run_server.sh
  ./run_server.sh

Dashboard
---------
  http://localhost:8000/dashboard/

The server generates a fresh access token when no FGT_SERVER_TOKEN environment
variable is supplied. To provide a stable token before starting:

Windows PowerShell:
  $env:FGT_SERVER_TOKEN = "YOUR_PRIVATE_TOKEN"

Linux/macOS:
  export FGT_SERVER_TOKEN="YOUR_PRIVATE_TOKEN"

Security and privacy boundaries
-------------------------------
- Do not expose this development server directly to the public internet.
- Raw gallery images and image features are not sent by FL clients.
- Individual DP-protected model updates are visible to the trusted aggregator.
- Cryptographic secure aggregation is not implemented.
- Rotate the access token before sharing a live server address.

Model task
----------
Single-label seven-parent softmax classification in this exact order:
people, places, activities, objects, documents, nature, events.

See SRS.md for the complete as-implemented architecture and limitations.
"""


def _collect_entries() -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for name in CORE_FILES:
        path = SERVER_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Required core file is missing: {path}")
        entries[name] = path.read_bytes()

    config = json.loads((SERVER_DIR / "config.json").read_text(encoding="utf-8"))
    config.pop("server_token", None)
    entries["config.json"] = (json.dumps(config, indent=2) + "\n").encode()

    for path in sorted((SERVER_DIR / "dashboard").rglob("*")):
        if path.is_file():
            entries[path.relative_to(SERVER_DIR).as_posix()] = path.read_bytes()

    for name in MODEL_FILES:
        path = SERVER_DIR / "models" / name
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")
        entries[f"models/{name}"] = path.read_bytes()

    entries["README.txt"] = README_TEXT.encode()
    entries["SRS.md"] = (ROOT_DIR / "SRS.md").read_bytes()

    schema = json.loads(entries["models/model_schema.json"])
    baseline = json.loads(entries["models/baseline_metrics.json"])
    release_info = {
        "release": "GalleryFL Public Central Aggregator",
        "model_version": int(entries["models/model_version.txt"].decode().strip()),
        "task": schema["task"],
        "num_classes": schema["num_classes"],
        "labels": schema["labels"],
        "baseline_macro_f1": baseline["macro_f1"],
        "baseline_accuracy": baseline["accuracy"],
        "secure_aggregation": False,
    }
    entries["RELEASE_INFO.json"] = (json.dumps(release_info, indent=2) + "\n").encode()

    sums = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in sorted(entries.items())]
    entries["SHA256SUMS.txt"] = ("\n".join(sums) + "\n").encode()
    return entries


def package_server() -> tuple[Path, Path]:
    entries = _collect_entries()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            if name == "run_server.sh":
                info.external_attr = 0o100755 << 16
            archive.writestr(info, data)

    digest = hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest()
    CHECKSUM_FILE.write_text(f"{digest}  {OUTPUT_ZIP.name}\n", encoding="utf-8")
    print(f"Created {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"SHA-256: {digest}")
    return OUTPUT_ZIP, CHECKSUM_FILE


if __name__ == "__main__":
    package_server()
