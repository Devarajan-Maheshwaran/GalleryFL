import zipfile
from pathlib import Path


def package_server() -> None:
    server_dir = Path("server")
    output_zip = Path("FGT-server.zip")
    includes = [
        "dashboard", "models", "config.json", "config.py",
        "fl_coordinator.py", "fl_math.py", "main.py", "metrics.py",
        "model_eval.py", "model_manager.py", "requirements.txt",
        "run_server.bat", "run_server.py", "run_server.sh", "security.py",
        "tag_demand.py", "taxonomy.json", "taxonomy_parser.py", "ws_manager.py",
    ]
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in includes:
            path = server_dir / item
            if path.is_file():
                archive.write(path, arcname=item)
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in file_path.parts:
                        archive.write(file_path, arcname=file_path.relative_to(server_dir))
    print(f"Created {output_zip}")


if __name__ == "__main__":
    package_server()
