# FGT - Federated Gallery Tags
### Self-hosted federated photo intelligence

FGT is a self-hosted platform for privacy-preserving, on-device machine learning. It allows you to analyze and tag your mobile gallery photos without your images ever leaving your device. A local federated learning server orchestrates model updates, ensuring that while the collective AI model improves, your raw photos remain completely private.

## Downloads
Download the latest pre-compiled releases directly from GitHub (these links will be active once the release is published):
- [Download FGT Server ZIP](https://github.com/<owner>/<repo>/releases/latest/download/FGT-server.zip)
- [Download Android APK](https://github.com/<owner>/<repo>/releases/latest/download/FGT-client-release.apk)

---

## Features
- **Local-Only Inference:** All image scanning and tag generation happens directly on your Android device.
- **Federated Learning:** Only small, aggregated model updates (gradients) are sent to the server. Your photos never touch the network.
- **Self-Hosted Dashboard:** Monitor the health of your federated network, track participating devices, and observe model improvements in real-time.
- **Full Control:** Export the final, trained model directly from the dashboard, or download a comprehensive training report.
- **Transparent Undo/Revert:** Fully revert model updates from the app to maintain absolute control over the classification results.

## Architecture Overview
FGT consists of two main components:
1. **FGT Server (Python/FastAPI):** Orchestrates the federated learning rounds, aggregates client updates, and hosts the real-time monitoring dashboard.
2. **FGT Client (Android):** Connects to the server over a local LAN, uses the global model to scan your local photo gallery, performs on-device training, and submits weight updates back to the server.

---

## Privacy & Permissions

### Privacy First
FGT is designed to keep your data private:
- No photos or metadata are ever uploaded.
- The server only receives numerical weight updates for the ML model.
- You can review the exact size of the payload before sending it.

### Android Permissions
To function, the Android app requires:
- **Storage/Photos Access:** To read and analyze images stored on your device.
- **Network Access:** To connect to the local FGT server for fetching the latest model and submitting updates.

---

## Quick Start

### 1. Server Setup
Download and extract the `FGT-server.zip`.

**On Windows:**
Double-click `run_server.bat` or run it from the command prompt:
```cmd
run_server.bat
```

**On Linux/macOS:**
```bash
chmod +x run_server.sh
./run_server.sh
```

The script will automatically create a virtual environment, install dependencies, print the LAN IP address of your server, and open the dashboard at `http://localhost:8000/dashboard/`.

![Dashboard Screenshot](docs/dashboard.png)

### 2. Android Client Setup
Download and install the `FGT-client-release.apk` on your Android device.

1. Open the app and grant the necessary photo access permissions.
2. On the **Connect** screen, enter the LAN IP address provided by the server script (e.g., `http://192.168.1.100:8000`).
3. Enter the Access Code displayed on the server dashboard (or the default).
4. Tap **Join Training**.

![Android Screenshot](docs/android-home.png)

### 3. Usage
Once connected:
- **Scan Gallery:** The app will analyze your local photos using the current model.
- **Train:** Based on the analysis and your adjustments, the app computes updates and sends them to the server when a training round begins.
- **Export:** In the server dashboard, use the **Export Model** or **Export Report** buttons to download the trained `.tflite` model or the full metrics history.

---

## Building from Source

If you prefer to build the APK yourself instead of downloading the pre-compiled release:

1. Open the `android` folder in Android Studio.
2. Build a debug APK: `Build > Build Bundle(s) / APK(s) > Build APK(s)`.
3. To build a signed release APK, configure your keystore via Gradle properties or environment variables, and run:
   ```bash
   ./gradlew assembleRelease
   ```
*(Note: Do not commit your private keystore credentials to the repository.)*
