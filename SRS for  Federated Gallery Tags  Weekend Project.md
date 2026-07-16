# SRS for "Federated Gallery Tags" Weekend Project

## 1. Overview

"Federated Gallery Tags" (FGT) is a weekend-build, privacy-preserving federated learning demo that trains a shared image-tagging model across multiple Android devices using Flower, without ever uploading raw images to a central server. The final model auto-organizes each user’s gallery into human-meaningful folders and tags, exposing an open, self-hosted alternative to opaque OEM gallery AI.[^1][^2]

The system consists of:
- A **PC-based Federated Aggregator** (Flower server running on the user’s laptop/desktop).
- One or more **Android Client Apps** (friends’ phones) that perform on-device training and inference with local gallery images.
- Two distributable artifacts: `FGT-server.zip` (PC) and `FGT-client.zip` containing signed APK(s) for phones.

## 2. Innovation Goals and Constraints

### 2.1 Innovation Goals

- Demonstrate **real federated learning** on everyday data (photo galleries), with a self-hosted server visible and controllable by the user.[^2][^1]
- Provide **open, modifiable models and pipelines** versus proprietary OEM gallery AI and show the comparison in gains in accuracy and performance[^3][^4]
- Use **non-IID personal data** (different lifestyle/photo patterns) to collaboratively train a single global model without centralizing images.[^2]
- Offer a **simple, fun UX**: "Run a federated AI experiment with your friends on the same Wi‑Fi" that auto-organizes their galleries.
- Add **post-FL organization behavior**: moving images into appropriate folders or applying tags recognized by the default gallery.

### 2.2 Non-Goals

- Competing with production-grade Google Photos / Apple Photos quality or scale.[^4][^3]
- Implementing advanced cryptographic FL (secure aggregation, homomorphic encryption) beyond basic Flower protocols.[^1][^2]
- Handling cloud deployment; scope is LAN-based FL with one local server.

### 2.3 Constraints

- **Time**: Implementation must be feasible within a single weekend (Friday evening–Sunday evening).
- **Complexity**: Reuse Flower’s Android example and TFLite tooling to avoid building FL infrastructure from scratch.[^5][^6][^7]
- **Accessibility**: End users should be able to run the server by unzipping a folder and executing a script, and install the APK without Android Studio.

## 3. System Context and Actors

### 3.1 Actors

- **Server Operator (You/Host)**
  - Starts the Flower server on a PC.
  - Shares Wi‑Fi, IP address, and port with friends.
  - Monitors rounds, performance metrics, and manages releases.

- **Mobile Participant (Friend/User)**
  - Installs the FGT Android app (APK).
  - Grants gallery access to the app.
  - Opts in to federated training and auto-organization behavior.

- **GitHub/LinkedIn Audience**
  - Clones and runs the demo.
  - Evaluates innovation and quality based on README, screenshots, and behavior.

### 3.2 External Systems

- **Flower Framework** (Python library and Android example) providing FL server and Android client infrastructure.[^6][^8][^7]
- **Android OS Media Store** for reading/writing image metadata and reflecting tags/folders in default gallery applications.[^9][^10]
- **Default OEM Gallery Apps** (Samsung Gallery, Google Photos, OEM variants) that will display reorganized folders or tags created by FGT.[^11][^12][^10]

## 4. Functional Requirements

### 4.1 Federated Aggregator (PC Server)

#### FR-S1: Start/Stop Federated Server

- The system shall provide a script `run_server.sh` (Linux/macOS) and `run_server.bat` (Windows) inside `FGT-server.zip` to:
  - Create or activate a Python virtual environment.
  - Install dependencies (`flwr`, `tensorflow` / `tflite-runtime`, etc.).
  - Start a Flower server listening on a configurable host/IP and port.[^7][^6]

#### FR-S2: Register and Manage Clients

- The server shall accept connections from multiple Android clients implementing the Flower Android protocol, identified by client IDs.[^7]
- The server shall maintain an in-memory registry of connected clients and display:
  - Number of active clients.
  - Client IDs.
  - Current round participation status.

#### FR-S3: Federated Training Coordination

- The server shall coordinate federated training rounds using a strategy such as `FedAvgAndroid`:
  - Send initial global model to selected clients.[^7]
  - Receive updated weights after local training.
  - Aggregate updates into a new global model.
  - Repeat for N rounds or until convergence criteria.

#### FR-S4: Metrics Logging

- The server shall log per-round metrics:
  - Number of participating clients.
  - Training loss/accuracy returned from clients (on local validation subsets).
  - Global validation performance on a small host-side validation set.

#### FR-S5: Export Final Global Model

- After training completes, the server shall export the final global model as a TFLite file into `output/model/global_model.tflite` inside the server directory.
- This model shall be compatible with the Android client app for inference.

### 4.2 Android Client App

#### FR-C1: Server Connection Configuration

- The app shall provide a "Connect" screen with:
  - Input fields for server IP and port.
  - Optional client nickname/ID.
  - "Test Connection" button that attempts to reach the server and shows status.

#### FR-C2: Permission and Gallery Access

- The app shall request runtime permission to read photos from the gallery (e.g., READ_MEDIA_IMAGES/READ_EXTERNAL_STORAGE depending on API level).[^10][^9]
- The app shall allow the user to:
  - Choose one or more folders or albums to include in the training/analysis.
  - Limit the number of images used (e.g., up to N per session) to keep training time reasonable.

#### FR-C3: Local Dataset Creation

- The app shall construct a local dataset from selected images:
  - Resize images to a fixed resolution (e.g., 224×224).
  - Convert to tensors compatible with the TFLite model.
  - Maintain a mapping between image URIs and predicted tags.

#### FR-C4: Local Training Loop

- The app shall implement a local training loop compatible with Flower’s Android example:[^5][^7]
  - Initialize model parameters from server-provided TFLite weights.
  - Run K epochs of training over the local dataset using an optimizer (e.g., SGD or Adam).
  - Compute local loss/accuracy on a validation subset.
  - Serialize updated weights and metrics to send back to the server.

#### FR-C5: Federated Session Participation

- The app shall provide a "Join Federated Training" button:
  - When pressed, the app connects to the Flower server and participates in training rounds until completion or user cancellation.
  - The app shall display progress logs: current round, local loss/accuracy, global round number.

#### FR-C6: Inference and Tagging

- After receiving a final or updated global model from the server, the app shall:
  - Run inference on the chosen images to produce tag probabilities for configured tag classes (e.g., Selfie, Friends, Gym, Travel, Pets, Food, Important Documents) and Internal taggings and groupings inside each of them with option to not include them or delete them manually
  (example: APP -> Friends ->Friend1...friend'n' with rename functions).
  - Select the most confident tag per image or top-K tags.

#### FR-C7: Auto-Organization Behavior

- The app shall provide an "Organize My Gallery" function:
  - For each image, apply:
    - Tag metadata update (where supported via Media Store or EXIF-like mechanisms).[^9][^10]
    - Optional file move into a tag-specific folder under `/Pictures/FGT/<TagName>`.
  - The default gallery should reflect these changes as:
    - New folders/albums named after tags (e.g., `FGT_Gym`, `FGT_Travel`).
    - Tag-based search where compatible with OEM gallery behavior.[^12][^11][^10]

#### FR-C8: Revert/Control Mechanism

- The app shall provide:
  - A preview mode showing which images will be moved or tagged before changes are applied.
  - A "Revert" or "Undo" option for a reasonable number of recent operations (e.g., move images back from FGT folders).

### 4.3 UI and UX Requirements

#### FR-UI1: Home Screen

- Elements:
  - Title: "Federated Gallery Tags".
  - Short description: "Organize your photos with privacy-preserving federated AI".
  - Buttons: "Connect to Server", "Analyze Gallery", "Join Federated Training", "Organize Gallery".
  - Status bar panel showing:
    - Server connection status.
    - Global model version (e.g., Round #).

#### FR-UI2: Connect Screen

- Elements:
  - Text fields: Server IP, Port, Client ID.
  - "Test Connection" button with animated spinner (simple rotation or pulsating dot) while attempting connection.
  - Status message: success/failure.

#### FR-UI3: Gallery Analysis Screen

- Elements:
  - Folder/album selection UI (checkbox list or Android picker).
  - Button: "Scan & Tag Preview".
  - Animated circular progress indicator while scanning.
  - Output visualizations:
    - Animated pie chart of tag distribution (e.g., Selfie 30%, Gym 10%, Travel 25%). Simple animation could be segments growing from 0 to full size.
    - Tag list with counts and iconography (camera icon for selfies, dumbbell for gym, airplane for trips).

#### FR-UI4: Federated Training Screen

- Elements:
  - Text area or console-like log showing training messages ("Round 1: local loss=... global accuracy=...").
  - Animated line chart or bar chart representing global accuracy over rounds (updated in near-real-time with simple transitions).
  - "Start" / "Pause" / "Leave Session" buttons.
  - Summary box: "You contributed N images; Global accuracy improved from A to B".

#### FR-UI5: Organize Gallery Screen

- Elements:
  - List of proposed operations: image thumbnail + suggested tag + target folder.
  - Checkboxes to include/exclude specific operations.
  - Button: "Apply Changes".
  - Animated progress bar while file moves/tags are applied.
  - Post-operation message: "FGT created X new folders and moved Y images".

### 4.4 Packaging and Distribution Requirements

#### FR-P1: Server Zip

- `FGT-server.zip` shall contain:
  - `/server` directory with Flower server code, model definitions, and configs.
  - `run_server.sh` and `run_server.bat` scripts.
  - `requirements.txt` or equivalent dependency description.
  - A minimal `README_server.md` with quickstart instructions.

#### FR-P2: Client Zip

- `FGT-client.zip` shall contain:
  - One or more signed APK files (e.g., `FGT-client-release.apk`).
  - A `README_client.md` describing installation steps and permissions.

#### FR-P3: GitHub Releases

- The project repository shall define GitHub Releases including both zip packages to simplify sharing and social-media-driven adoption.

## 5. Non-Functional Requirements

### 5.1 Performance

- Training rounds on a typical mid-range Android phone (with limited dataset size, e.g., 200–500 images) should complete within a few minutes per round.
- Inference for tagging a batch of up to 1000 images should complete within a time frame acceptable to casual users (e.g., under a few minutes). On-device ML tutorials emphasize efficient TFLite models for reasonable on-device latency.[^9]

### 5.2 Usability

- UI shall avoid technical jargon; FL details can be optionally shown in an "Advanced" section.
- Animated outputs (charts, progress indicators) shall be simple and non-distracting.
- Error messages shall clearly explain issues (no server, bad IP, insufficient permissions) and next steps.

### 5.3 Privacy

- No raw images or per-image embeddings shall leave the Android device; only model weights/gradients and aggregate metrics are transmitted.[^1][^2]
- Networking shall be limited to local Wi‑Fi/LAN; internet connectivity is not required.

### 5.4 Reliability

- If the server goes offline mid-round, clients shall handle disconnects gracefully and allow rejoining.
- If a client drops during training, the server’s aggregation strategy shall still proceed with other clients.[^6][^7]

### 5.5 Portability

- Server scripts target common desktop OSes (Windows, macOS, Linux) with minimal dependencies and an option for Dockerization.
- Android client targets API levels that allow gallery access and ML runtimes, according to current on-device ML guidelines.[^9]

## 6. Innovation Angles vs OEM Galleries

### 6.1 OEM Gallery Behaviors

- Apple Photos: uses on-device ML for face recognition and photo curation, and for some features mixes local processing with privacy-preserving server-side matching (e.g., landmark recognition using homomorphic encryption and differential privacy).[^13][^14][^4]
- Google Photos: employs ML models (cloud and on-device) to label and organize photos; Google can, in principle, access photos for ML purposes even if features emphasize privacy.[^15][^3]
- Samsung Gallery and similar OEM apps: implement face recognition and people tagging, letting users correct or add faces; models and pipelines remain proprietary and not user-programmable.[^11][^12]

### 6.2 Differentiating Features

- **Self-Hosted Federated Training**: FGT explicitly exposes a local FL server the user runs, making the collaborative training process visible and controllable rather than hidden.[^6][^2][^1]
- **Open Model and Pipeline**: Model architecture, training loops, and data flows are implemented in open-source code, allowing inspection and modification.[^8][^6]
- **Small-Group Collaborative Training**: A classroom or friend group can jointly train a model on their diverse photo distributions, demonstrating non-IID FL behavior experimentally.[^2]
- **Programmable Auto-Organization**: Users can customize folder naming, tag schemas, and organization rules via configuration files or in-app settings, unlike fixed OEM behaviors.[^10][^9]

### 6.3 Innovation Add-Ons (Optional)

- **FL vs Non-FL Comparison Mode**: Provide toggles to run: (a) a centrally trained baseline model and (b) the federated model, exposing accuracy differences on local validation sets.
- **Per-User Personalization Layer**: Add a lightweight personal adaptation head on top of the global model, trained only on each user’s images, highlighting a known FL technique for handling non-IID data.[^2]
- **Transparent Privacy Dashboard**: Visualize what is sent (weights, metrics) versus what stays local (images, tags) to reinforce privacy-preserving design.[^1]

## 7. System Architecture (High-Level)

### 7.1 Components

- **Server Component**
  - Flower server.
  - Global model manager (saving, versioning TFLite models).
  - Metrics logger.

- **Client Component**
  - Android UI layer.
  - Gallery data loader.
  - Local trainer (TFLite training or fine-tuning-compatible engine).
  - Tag inference engine.
  - Organizer module (Media Store updates and file ops).

### 7.2 Data Flow

1. Server boots with initial global model.
2. Clients connect, receive global model weights.
3. Clients locally train on gallery images, produce updated weights + metrics.
4. Server aggregates updates into new global model.
5. After N rounds, server exports final global model.
6. Clients use the final global model to tag and organize galleries.

## 8. UI Flow Summary

1. **Connect**: User enters server IP/port → tests connection.
2. **Analyze**: User selects folders → app scans and generates tag distribution visualization.
3. **Join FL**: User taps "Join Federated Training" → app participates in FL rounds and shows training logs and a simple accuracy-over-rounds chart.
4. **Organize**: User previews proposed moves/tags → taps "Apply Changes" → progress animation and success summary.

## 9. Acceptance Criteria

- A user can download `FGT-server.zip`, run a script, and see the Flower server ready with a global model.
- At least two Android phones with `FGT-client.apk` can connect, participate in federated training rounds, and receive updated global models.
- The app can tag and reorganize selected gallery images into tag-based folders, observable in the default gallery.
- No raw image data leaves devices; only weights/metrics are transmitted.
- UI provides clear, simple animations for connection tests, analysis progress, training, and organization without overwhelming users.

---

## References

1. [Federated learning: what it is and how it works](https://cloud.google.com/discover/what-is-federated-learning) - Federated learning is a privacy-preserving AI and machine learning technique. Learn about its model,...

2. [What is federated learning in image search?](https://milvus.io/ai-quick-reference/what-is-federated-learning-in-image-search) - Federated learning (FL) is a machine learning approach where models are trained across decentralized...

3. [Google Photos' AI Models](https://skyld.io/google-photos-model-extraction) - In this study, we examined AI model extraction in Google Photos, particularly focusing on its behavi...

4. [Recognizing People in Photos Through Private On-Device ...](https://machinelearning.apple.com/research/recognizing-people-photos) - Photos uses a number of machine learning algorithms, running privately on-device, to help curate and...

5. [Federated Learning on Android devices with Flower](https://flower.ai/blog/2021-12-15-federated-learning-on-android-devices-with-flower/) - Flower can be used to federate machine learning training on Android devices. pipelines written in Py...

6. [flwrlabs/flower - A Friendly Federated AI Framework](https://github.com/flwrlabs/flower) - Flower ( flwr ) is a framework for building federated AI systems. This series of tutorials introduce...

7. [Flower Android Example (TensorFlowLite)](https://flower.ai/docs/examples/android.html) - This example demonstrates a federated learning setup with Android clients in a background thread. Th...

8. [Flower: A Friendly Federated Learning Framework](https://arxiv.org/pdf/2007.14390.pdf) - by DJ Beutel · 2020 · Cited by 2055 — Python-based Flower clients are implemented for Nvidia Jetson ...

9. [Image Classification | On-Device ML](https://developers.google.com/learn/pathways/on-device-ml-2) - Learn to build custom image classification models, and improve the skills you gained in the Get star...

10. [I need gallery with face recognition & OCR (No cloud)](https://www.reddit.com/r/androidapps/comments/1s318f4/i_need_gallery_with_face_recognition_ocr_no_cloud/) - It's open source, works fully offline, and supports things like tagging and basic search. It doesn't...

11. [How to Fix Face Tags Not Working in Samsung Gallery | Fix it ...](https://www.youtube.com/watch?v=DgMIFAvD3RY) - 1. Open Samsung Gallery app and tap on the search icon. 2. Tap the three-dot menu → Settings 3. Enab...

12. [Samsung gallery face recognition update](https://eu.community.samsung.com/t5/samsung-lounge/samsung-gallery-face-recognition-update/td-p/9123675) - Adding faces to the pictures in samsung gallery when the AI cannot recognise a face or recognised a ...

13. [Apple now uses AI to analyze your photos by default - without consent](https://discuss.techlore.tech/t/apple-now-uses-ai-to-analyze-your-photos-by-default-without-consent/11763) - Apple last year deployed a mechanism for identifying landmarks and places of interest in images stor...

14. [Apple's AI Philosophy: Privacy-First, On-Device Intelligence](https://statusneo.com/apple-ai-pioneering-a-new-era-of-privacy-with-game-changing-on-device-intelligence/) - Apple's Face ID isn't just about unlocking your phone. It's powered by machine learning that learns ...

15. [Does google use my google photos for machine learning ...](https://support.google.com/photos/thread/226496087/does-google-use-my-google-photos-for-machine-learning-purposes?hl=en) - Theoretically, Google can access the photos. Nevertheless, no one will do so without obtaining expli...

NOTE: I CARE ABOUT COMPLETE PROFESSIONAL WORKING OF THE PROJECT, and quality of the output(better than existing gallery tag models used in latest androids, apple, google) and the clear visibility of the output and performance gains

UI/UX should be fun and implement Neumorphism (Soft UI) with warm colours; strictly no emojis and leave out comments and other not wanted content for now
