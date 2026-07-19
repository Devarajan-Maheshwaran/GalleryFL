import os
import zipfile
import shutil
from pathlib import Path

def package_server():
    print("Packaging FGT Server...")
    
    server_dir = Path("server")
    output_zip = "FGT-server.zip"
    
    # Files and directories to include
    includes = [
        "dashboard",
        "models",
        "output",
        "config.json",
        "config.py",
        "fl_coordinator.py",
        "main.py",
        "metrics.py",
        "model_manager.py",
        "retrain",
        "requirements.txt",
        "run_server.bat",
        "run_server.py",
        "run_server.sh",
        "scripts",
        "security.py",
        "tag_demand.py",
        "taxonomy.json",
        "taxonomy_parser.py",
        "verify_fl_loop.py",
        "ws_manager.py",
        "README-server.md"
    ]
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item in includes:
            item_path = server_dir / item
            if not item_path.exists():
                print(f"Warning: {item_path} does not exist, skipping.")
                continue
                
            if item_path.is_file():
                print(f"Adding file: {item}")
                zf.write(item_path, arcname=item)
            elif item_path.is_dir():
                for file_path in item_path.rglob("*"):
                    # Exclude pycache and pytest_cache
                    if "__pycache__" in file_path.parts or ".pytest_cache" in file_path.parts:
                        continue
                    # Exclude large regenerable eval artifacts and live logs.
                    if file_path.name in ("fl_probe.npz", "audit.log"):
                        continue
                    print(f"Adding file: {file_path.relative_to(server_dir)}")
                    zf.write(file_path, arcname=file_path.relative_to(server_dir))

    print(f"Successfully created {output_zip} in the current directory.")

if __name__ == "__main__":
    package_server()
