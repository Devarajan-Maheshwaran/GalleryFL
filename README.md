# GalleryFL

GalleryFL is a privacy-oriented Android gallery organizer backed by a local-network federated-learning server. Images remain on Android devices. The server distributes a compact seven-parent classification head, coordinates local fine-tuning, aggregates bounded client updates, and serves a live monitoring dashboard.

## Current model

The deployed classifier predicts exactly seven broad parent categories:

1. People
2. Places
3. Activities
4. Objects
5. Documents
6. Nature
7. Events

Architecture:

```text
224×224 RGB image
  → frozen two-output TFLite backbone
  → 1024-D projection
  → Dense(256, ReLU)
  → Dense(7)
  → softmax / argmax
```

The production head is `server/models/head_weights.npz`. Its held-out COCO-parent-proxy result is:

- Macro F1: **0.3055**
- Accuracy: **0.3407**
- Model version: **2**

This is above chance and usable as a bootstrap, but it is not a high-accuracy autonomous gallery model. COCO is a weak source for Documents and Events, so those automatic categories are disabled by conservative confidence gates until trusted corrections are collected. GalleryFL is designed to improve cautiously from high-confidence unlabelled photos and, more importantly, explicit user corrections.

## Federated-learning behavior

The runtime implements a custom Android-friendly FL protocol over REST and WebSocket. Flower is used as a reference verifier, not as the production transport.

Each FL round:

1. At least two connected clients receive the global head and round configuration.
2. Each phone extracts frozen 1024-D features locally.
3. Explicit corrections become full-weight categorical targets.
4. Unlabelled photos are used only when confidence exceeds both the global pseudo-label gate and the class-specific gate.
5. Rejections without a replacement category are excluded because they are not valid single-label targets.
6. The phone freezes the first dense layer and fine-tunes only `w2/b2` with categorical cross-entropy, FedProx, fixed per-example clipping, and optional Gaussian local DP.
7. Only weights and aggregate metadata are sent; images never leave the phone.
8. The server clips each client delta and performs effective-sample-weighted FedAvg. Human labels count fully; pseudo-labels count at reduced weight.
9. A held-out validation artifact commits a pseudo-label-only round only when it strictly improves macro F1. Human-correction rounds receive only a small fixed baseline tolerance for domain adaptation.

Important limitation: unlabelled data alone cannot reveal a semantic category when the initial model is wrong. Confidence-gated self-training can refine a good boundary but can also reinforce mistakes. The correction picker in Image Detail is therefore the trusted learning signal.

## Verified FL results

Reference tests used Flower 1.32 FedAvg and five strongly Non-IID clients (Dirichlet alpha 0.25). Client labels were hidden during local training; clients used confidence-gated pseudo-labels with local DP.

```text
Baseline macro F1: 0.299729
Final macro F1:    0.302369
Baseline accuracy: 0.326846
Final accuracy:    0.330872
Accepted rounds:   5 / 10
```

A ten-seed re-verification then executed 100 candidate rounds under the same unlabelled Non-IID and DP-enabled conditions. Nine of ten runs finished with positive macro-F1 change; all ten were non-regressing. Mean macro-F1 change was **+0.000885**, mean accuracy change was **+0.000940**, and Flower/project output aggregation differed by at most **5.96e-8**. Without the commit guard, eight of ten runs regressed and mean macro-F1 change was **-0.002542**.

A separate live REST/WebSocket run completed eight rounds with four clients, accepted every submission, advanced model versions, persisted metrics, and broadcast all round events. The production model was restored after the verification run.

These checks prove the implementation and strict improvement gate work on the committed validation domain. They do not guarantee improvement on every private gallery distribution.

## Repository layout

```text
GalleryFL/
├── android/                 Android app (Kotlin, Compose, LiteRT, Room)
├── server/
│   ├── dashboard/           Browser monitoring UI
│   ├── models/              Backbone, head, schema, thresholds, validation artifact
│   ├── main.py              FastAPI endpoints
│   ├── fl_coordinator.py    Round orchestration and guarded aggregation
│   ├── fl_math.py           Bounded sample-weighted FedAvg
│   ├── model_eval.py        Framework-free seven-class evaluation
│   ├── model_manager.py     Versioned model state and wire serialization
│   └── security.py          Validation, rate limiting, trust, audit log
├── README.md
└── SRS.md
```

## Requirements

### Server

- Windows 10/11, Linux, or macOS
- Python 3.11 or 3.12 recommended
- Two or more Android clients for a true federated round
- All devices on the same LAN

### Android

- Android Studio with JDK 21
- Android SDK Platform 36
- Android 10/API 29 or newer device
- USB debugging for command-line installation

## Run the server on Windows PowerShell

```powershell
Set-Location "C:\path\to\GalleryFL"

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\server\requirements.txt

Set-Location .\server
python .\run_server.py
```

Open the dashboard in another PowerShell window:

```powershell
Start-Process "http://localhost:8000/dashboard/"
```

## Build and install Android

```powershell
Set-Location "C:\path\to\GalleryFL\android"

$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:Path += ";$env:ANDROID_HOME\platform-tools"

.\gradlew.bat clean assembleDebug
adb devices
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

For an emulator, use `http://10.0.2.2:8000`. For a physical phone, use the PC's LAN IPv4 address and the access token shown by the server, for example:

```text
192.168.1.10:8000@ABCD2345
```

If Windows Firewall blocks port 8000, run PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "GalleryFL Server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## Core API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/register` | Register or resume a client |
| `GET` | `/api/model/current` | Download a full head or one-version delta |
| `GET` | `/api/model/schema` | Read the canonical seven-class tensor contract |
| `POST` | `/api/training/start` | Start an FL session |
| `POST` | `/api/training/submit-update` | Submit a trained client head |
| `POST` | `/api/training/skip-update` | Respond when a client has insufficient usable data |
| `GET` | `/api/training/status` | Session, model, and client status |
| `GET` | `/api/metrics/history` | Persisted round metrics |
| `GET` | `/api/metrics/comparison` | Deployment baseline versus current global head |
| `GET` | `/api/taxonomy` | Parent/leaf taxonomy metadata |
| `POST` | `/api/taxonomy/signal` | Record local tag-demand signals |
| `WS` | `/ws/feed` | Heartbeats and round events |

Authenticated POST requests use `X-FGT-Token`.

## Privacy and security boundaries

- Images and extracted gallery metadata stay on the device.
- The backbone is frozen; only the small output layer is locally optimized.
- Per-example gradients are clipped to a fixed public bound.
- If epsilon is positive, Gaussian noise is added with conservative per-round budget composition.
- Client deltas are bounded again at the server.
- Shapes, versions, finite values, duplicate updates, stale rounds, and anomalous norms are validated.
- The server still receives individual DP-protected updates. Cryptographic secure aggregation is **not** implemented and must not be claimed.

See `SRS.md` for the full architecture, algorithms, requirements, data contracts, and acceptance criteria.
