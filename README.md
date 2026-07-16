# GalleryFL: Federated Gallery Tags

GalleryFL is a complete, end-to-end implementation of a Federated Learning (FL) system designed to train image classification models directly on edge devices (Android) while preserving user privacy. 

Traditional machine learning requires centralizing large datasets, which poses significant privacy risks when dealing with personal photos. GalleryFL solves this by sending the model to the data instead of sending the data to the model. Devices train the model locally using their own galleries and only transmit the learned weight updates (gradients) to a central server.

## Architecture

The system consists of three main components:

1. **FastAPI Coordination Server**
   - Coordinates FL training rounds and manages client connections via WebSockets.
   - Aggregates local updates using a Trimmed Mean algorithm to defend against malicious or outlier updates.
   - Implements strict validation checks (norm constraints and shape verification) for incoming updates.
   - Provides a real-time web dashboard to monitor training metrics (loss, accuracy, client contributions).

2. **Android Client Application**
   - Built with Jetpack Compose and Kotlin Coroutines.
   - Implements local training using a pre-trained MobileNetV3 feature extractor and a trainable classification head.
   - Utilizes Differential Privacy (DP) via Gaussian noise injection and Gradient Clipping (L2 Norm) before transmitting any data.
   - Communicates seamlessly with the server via OkHttp WebSockets and Retrofit REST APIs.

3. **Python Simulator Client**
   - A robust `mock_client.py` for testing and simulating large-scale federated networks without needing physical Android devices.
   - Simulates local training loops, applies FedProx-style proximal terms, and serializes gradients exactly like the Android client.

## Security and Privacy Features

- **Data Locality:** Raw images and metadata never leave the device.
- **Differential Privacy (DP):** Gaussian noise is injected into weight updates to prevent inference attacks that might attempt to reconstruct local data.
- **Gradient Clipping:** Bounds the maximum impact any single client can have on the global model, ensuring stability.
- **Anomaly Detection:** The server evaluates incoming weight updates against historical norms using a 3-sigma rule, rejecting poisoned or corrupted updates.

## Setup Instructions

### Server
1. Navigate to the `server` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the base model:
   ```bash
   python prep_model.py
   ```
4. Run the server:
   ```bash
   python run_server.py
   ```
5. Access the dashboard at `http://localhost:8080/dashboard`.

### Android Client
1. Open the `android` folder in Android Studio.
2. Build and run the application on an emulator or physical device.
3. In the application, enter the server URL (e.g., `http://10.0.2.2:8080` for emulator) and join the training session.

## Testing

The server includes a suite of integration tests.
```bash
cd server
pytest test_integration.py
```
