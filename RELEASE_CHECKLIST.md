# Release Checklist

## Verifications Completed
- **No Mock Artifacts**: Removed `server/mock_client.py` and `server/run_mock_fl.py` from the main tree and excluded from the release packaging.
- **Tests Isolated**: Moved all test scripts (`test_alignment.py`, `test_integration.py`, `test_reports.py`) to the `server/tests/` directory and excluded them from the release build.
- **Server Startup Scripts**: Updated `run_server.bat` and created `run_server.sh` to seamlessly handle virtual environments, dependency installation, and LAN IP resolution.
- **Android UI/Copy**: Ensured all user-facing strings are polished. Replaced "GalleryFL" and "demo" strings with the official product name: "FGT - Federated Gallery Tags".
- **Dynamic Classification Head**: Removed hardcoded `NUM_CLASSES` assumptions in the Android client (`MainActivity.kt`). The `ClassificationHead` now dynamically determines the class count from the global weights provided by the server.
- **Model Schema Alignment**: Verified the TensorFlow Lite serialization schema matches exactly between the server and Android (`w1`, `b1`, `w2`, `b2`).
- **APK Gradle Config**: Verified that `.tflite` assets are explicitly set to not compress in `build.gradle.kts`.

## Included in `FGT-server.zip`
- `dashboard/` (Frontend HTML/CSS/JS)
- `models/` (Baseline weights, schema, `.tflite` model)
- `output/` (Directory for exports)
- `config.py`
- `fl_coordinator.py`
- `main.py`
- `metrics.py`
- `model_manager.py`
- `prep_model.py`
- `security.py`
- `taxonomy_parser.py`
- `ws_manager.py`
- `taxonomy.json`
- `requirements.txt`
- `run_server.bat`
- `run_server.sh`
- `README-server.md`

## Excluded from `FGT-server.zip`
- `__pycache__/` and `.pytest_cache/`
- Mock and test scripts
- Raw dataset/conversion scripts (`convert_dataset.py`, `coco_to_fgt_mapping.py` are left in source repo but not needed in the zipped release)

## Publishing the Release
To complete the release:
1. Run `python scripts/package_server.py` to generate `FGT-server.zip`.
2. Build the Android release APK in Android Studio (`Build > Generate Signed Bundle / APK`).
3. Create a new Release in GitHub.
4. Upload `FGT-server.zip` and the generated APK (rename it to `FGT-client-release.apk`) to the Release Assets.
5. The download links in `README.md` will become fully functional!
