# Federated Gallery Tags (FGT) — System and Federated Learning Architecture SRS

**Document version:** 3.0

**Status:** As-implemented specification, verified against repository HEAD

**System:** GalleryFL Android application and Federated Gallery Tags server

**Primary deployment:** Private local-area network

**Verification baseline:** Git commit `a9a5632` plus the conformance corrections recorded in this revision

---

## 1. Purpose

This Software Requirements Specification defines the behavior, interfaces, model contract, federated-learning algorithm, privacy boundaries, operational requirements, and acceptance criteria for GalleryFL.

GalleryFL organizes an Android photo gallery into seven broad semantic parents while keeping source images on the phone. A LAN server coordinates federated fine-tuning of a small classification head. The system is intended as a privacy-oriented prototype and research-quality application baseline, not as a claim of production-grade visual recognition or formal end-to-end anonymity.

## 2. Scope

The core system contains:

- An Android gallery application.
- A frozen TFLite visual feature extractor.
- A trainable seven-parent classification head.
- Local semi-supervised and correction-driven fine-tuning.
- Local example-level differential privacy when enabled.
- A FastAPI federated coordinator.
- Bounded sample-weighted FedAvg aggregation.
- Versioned model distribution and persistence.
- A browser dashboard for live status and metrics.
- On-device gallery grouping, organization, EXIF tagging, and undo support.

The core system does not include:

- Uploading gallery images to the server.
- Cloud storage.
- Cryptographic secure aggregation.
- A guarantee that unlabelled examples provide correct semantic supervision.
- Fine-grained 34-leaf model outputs; leaf tags remain taxonomy metadata only.
- On-server backbone training.

### 2.1 Implementation verification against the supplied architecture addendum

The supplied architecture addendum was treated as a design input and checked against the current repository. Historical claims that no longer match the implementation are corrected below; the remainder of this SRS is normative for the shipped code.

| Area from supplied addendum | Repository evidence | As-implemented verdict |
|---|---|---|
| Federated orchestration | `server/main.py`, `fl_coordinator.py`, `model_manager.py` | **Conforms:** custom FastAPI coordinator; Flower is not the production transport. |
| FL communication | Android Retrofit/OkHttp and `FGTWebSocketClient.kt`; FastAPI REST/WS routes | **Conforms:** REST carries control/model data and WebSocket carries round events and heartbeats. |
| Split mobile model | Identical server/Android `base_model.tflite`; `FeatureExtractor.kt`; four-tensor head | **Conforms:** frozen LiteRT backbone with 1024-D projection and separately trainable head. |
| Task semantics | `model_schema.json`, `ClassificationHead.kt`, `TaxonomyConfig.kt` | **Corrected:** the deployed task is seven-parent **single-label multiclass**, using softmax/argmax—not 34-output multi-label sigmoid. The 34 leaves are metadata. |
| Trainable scope | `LocalTrainer.kt`, `backwardOutputOnly` | **Corrected:** FL freezes `w1/b1` and trains only `w2/b2` (1,799 parameters), rather than backpropagating the complete head. |
| Unlabelled learning | `PseudoLabelGenerator.kt`, `MainActivity.kt` | **Conforms with safeguards:** only high-confidence top-1 pseudo-labels are used; they receive reduced weight, and uncertain/rejected-only examples are excluded. |
| Trusted supervision | `RecordTagFeedbackUseCase.kt`, `ImageDetailScreen.kt` | **Upgraded:** an explicit seven-parent correction picker persists full-weight categorical targets. |
| Non-IID optimizer | `LocalTrainer.kt` | **Conforms:** categorical cross-entropy plus FedProx against downloaded global output weights. The local step is a full usable-set gradient, not the mini-batch process described historically. |
| Aggregation | `server/fl_math.py`, `_robust_fedavg_aggregate` | **Corrected:** clipped effective-sample-weighted FedAvg is current. Trimmed Mean is not in the shipped implementation. |
| Aggregation weights | Human/pseudo counts and trust score in `fl_coordinator.py` | **Upgraded:** human examples count fully, pseudo-labels use configured reduced weight, and tag popularity cannot bias model aggregation. |
| Differential privacy | `DPNoiseInjector.kt`, `LocalTrainer.kt`, `metrics.py` | **Upgraded:** fixed per-example clipping, `C/n` average sensitivity, secure Gaussian randomness, and conservative epoch/round composition replace the historical mis-calibrated whole-update path. |
| Candidate validation | `model_eval.py`, `validation_features.npz`, `baseline_metrics.json` | **Upgraded:** framework-free macro-F1/accuracy evaluation is populated and runs before a candidate is committed. |
| Flower positioning | README/SRS verification evidence | **Conforms:** Flower 1.32 is a numerical reference verifier only; production remains the custom Android protocol. |
| Android application stack | `android/app/build.gradle.kts` | **Corrected:** Kotlin, Compose, LiteRT, Retrofit, OkHttp, Moshi, Room, DataStore, Coil, MediaStore, and ExifInterface are present. Hilt and Paging are not dependencies. |
| Dashboard stack | `server/dashboard/` | **Corrected:** the dashboard uses self-contained Canvas drawing, not Chart.js. |
| Gallery organization | `MainActivity.kt`, `OrganizeExecutor.kt` | **Corrected:** organization and undo are wired into the UI; this is no longer an unexposed component. |
| Benchmark state | `models/baseline_metrics.json`, runtime model evaluator | **Corrected:** held-out metrics are populated; TensorFlow is not required by the live server evaluator. |

### 2.2 Normative architecture decision

The current single-label seven-parent contract is intentional. It supersedes older multi-label/34-leaf descriptions because the production head, Android folder policy, correction UI, model schema, FL loss, aggregation validator, and held-out metrics all use the same seven ordered parent classes. Reintroducing multi-label leaf outputs would require a coordinated schema version, new training data, new head artifacts, Android loss/inference changes, server evaluator changes, and a migration of persisted client weights.

## 3. Definitions

| Term | Definition |
|---|---|
| Parent class | One of People, Places, Activities, Objects, Documents, Nature, Events |
| Leaf tag | A detailed taxonomy item such as selfie, beach, receipt, or wedding; not a model output |
| Backbone | Frozen MobileNetV3-Small-derived TFLite image network that emits a 1024-D projection and a 7×7×1024 spatial map |
| Head | Four tensors: `w1`, `b1`, `w2`, `b2` |
| Global model | Current server-owned head |
| Local update | Client head after one or more local optimization steps |
| Non-IID | Client galleries have different class distributions and image characteristics |
| Pseudo-label | A confidence-gated class selected by the current model for an unlabelled photo |
| Trusted correction | A category explicitly selected by the user |
| FedProx | Local proximal regularization toward the downloaded global parameters |
| Local DP | Per-example clipped gradient plus Gaussian noise applied before an update leaves the device |
| Quality guard | Server validation that blocks unacceptable held-out macro-F1 regression |

## 4. Stakeholders

- **Gallery owner:** wants useful grouping without sending photos away.
- **Server operator:** runs the LAN coordinator and monitors rounds.
- **Developer/researcher:** evaluates Non-IID FL, DP utility, and model behavior.
- **Maintainer:** preserves the seven-class schema and Android/server wire compatibility.

## 5. Technology stack

### 5.1 Android

- Kotlin
- Jetpack Compose / Material 3
- Android SDK 36, minimum SDK 29
- Google LiteRT/TFLite interpreter and GPU delegate
- Frozen MobileNetV3-Small-derived two-output backbone
- Pure Kotlin seven-parent head and output-layer optimizer
- Room persistence
- DataStore preferences
- Retrofit, OkHttp, Moshi
- Kotlin coroutines
- MediaStore and EXIF interfaces

### 5.2 Server

- Python 3.11+
- FastAPI
- Uvicorn
- NumPy
- Pydantic
- REST and WebSocket transport
- Self-contained HTML/CSS/JavaScript dashboard with Canvas-based charts

### 5.3 FL verification reference

Flower is not the runtime protocol because the Android application already has a compact REST/WebSocket transport and a strict binary tensor contract. Flower FedAvg is used as a reference implementation when verifying numerical aggregation compatibility and Non-IID simulations.

## 6. System context

```text
┌──────────────────────── Android client A ───────────────────────┐
│ MediaStore → TFLite backbone → 1024-D features → 7-class head  │
│ user correction / confident pseudo-label → local DP + FedProx  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ weights + metadata only
                               │ REST / WebSocket over LAN
┌──────────────────────────────▼───────────────────────────────────┐
│ FastAPI server                                                   │
│ validation → bounded FedAvg → held-out quality guard → version  │
│ dashboard / metrics / model persistence                          │
└──────────────────────────────┬───────────────────────────────────┘
                               │ global head or one-version delta
┌──────────────────────────────▼───────────────────────────────────┐
│ Android clients B...N                                           │
└──────────────────────────────────────────────────────────────────┘
```

## 7. Model requirements

### 7.1 Class order

The following order is normative everywhere:

```text
0 people
1 places
2 activities
3 objects
4 documents
5 nature
6 events
```

### 7.2 Head tensor schema

| Tensor | Shape | Type | Meaning |
|---|---:|---|---|
| `w1` | 1024×256 | float32 | Frozen-runtime dense projection |
| `b1` | 256 | float32 | First-layer bias |
| `w2` | 256×7 | float32 | Federated output weights |
| `b2` | 7 | float32 | Federated output bias |

The deployed inference function is:

```text
hidden = ReLU(features × w1 + b1)
logits = hidden × w2 + b2 + optional_local_bias
probabilities = stable_softmax(logits)
prediction = argmax(probabilities)
```

### 7.3 Backbone contract

- Architecture lineage: MobileNetV3-Small-style inverted residual/squeeze-excitation backbone; the shipped TFLite graph contains blocks through `expanded_conv_10` plus the GalleryFL projection outputs.
- Input: float32 `[1,224,224,3]`.
- Pixel transform: `pixel / 127.5 - 1.0`.
- Required projection output: float32 `[1,1024]`.
- Optional spatial output: float32 `[1,7,7,1024]`.
- Server and Android backbone artifacts shall be byte-identical.
- Output selection shall use tensor shape rather than assuming output index.

### 7.4 Current baseline quality

The shipped head was trained using a COCO Minitrain 10K seven-parent proxy mapping.

| Metric | Value |
|---|---:|
| Held-out macro F1 | 0.3055 |
| Held-out accuracy | 0.3407 |
| People F1 | 0.3208 |
| Places F1 | 0.3664 |
| Activities F1 | 0.3742 |
| Objects F1 | 0.3769 |
| Documents F1 | 0.1516 |
| Nature F1 | 0.3945 |
| Events F1 | 0.1541 |

This baseline is a bootstrap, not a promise of acceptable accuracy on private galleries. Documents and Events are not adequately represented by COCO semantics. Their automatic confidence gates are disabled until trusted gallery corrections exist.

### 7.5 Confidence gates

Normative initial gates:

| Class | Gate |
|---|---:|
| People | 0.90 |
| Places | 0.80 |
| Activities | 0.80 |
| Objects | 0.80 |
| Documents | 1.01 (disabled) |
| Nature | 0.70 |
| Events | 1.01 (disabled) |

A gate is an optional post-argmax abstention rule. It must never be interpreted as independent multi-label thresholding.

## 8. Android functional requirements

### FR-A01 Gallery access

The application shall request platform-appropriate image permissions and enumerate recent images through MediaStore.

### FR-A02 Local feature extraction

The application shall resize and normalize each image locally and execute the canonical TFLite backbone without transmitting pixels.

### FR-A03 Model synchronization

The application shall download either a full head or a one-version delta, validate tensor counts and sizes, persist the head in app-private storage, and persist its version.

### FR-A04 Inference

The application shall compute seven-class softmax probabilities, apply local bias offsets, choose one top class, and apply that class's abstention gate.

### FR-A05 Conservative grouping

The application shall avoid creating a folder from a single uncertain image. Parent grouping shall require the configured minimum number of accepted images.

### FR-A06 Trusted correction

Image Detail shall expose a correction picker containing all seven parents. Selecting a parent shall:

- record rejection of a different predicted parent;
- record confirmation of the selected parent;
- persist the selected parent as the trusted target for that image;
- update local demand statistics;
- make the correction available to the next FL round.

### FR-A07 Rejection semantics

A rejection without a replacement class shall not be converted into an all-zero categorical target. It shall be excluded from local categorical training until a replacement is selected.

### FR-A08 Pseudo-label selection

For an image without feedback, the application may create a pseudo-label only when:

```text
max_probability >= max(server_pseudo_label_threshold, class_confidence_gate)
```

The default global pseudo-label threshold shall be 0.80. A pseudo-label shall carry reduced weight, default 0.25.

### FR-A09 Insufficient-data response

If a client has fewer than the configured usable examples, it shall call `/api/training/skip-update` rather than train on fabricated zeros or send unchanged weights.

### FR-A10 Local training scope

The local trainer shall freeze `w1/b1` and optimize only `w2/b2`. This requirement limits compute, stabilizes the centralized representation, and reduces the number of coordinates affected by DP noise from approximately 264,000 to 1,799.

### FR-A11 Gallery organization

The application shall support organization into `Pictures/FGT/<parent>`, EXIF tag writing, operation logging, and undo through the implemented organization components.

### FR-A12 Local state

Model weights, model version, feedback, scan results, media state, and operation logs shall survive ordinary app exit.

## 9. Local optimization requirements

### 9.1 Objective

For each usable example `i`:

```text
loss_i = sample_weight_i × categorical_cross_entropy(y_i, softmax(logits_i))
```

Human correction weight is 1.0. Pseudo-label weight is configurable in `[0,1]` and defaults to 0.25.

### 9.2 FedProx

For federated output parameter `w` and downloaded parameter `w_global`:

```text
w ← w - learning_rate × (gradient + μ × (w - w_global))
```

The server shall provide `μ`; default is 0.01.

### 9.3 Fixed clipping

Each per-example output-layer gradient shall be clipped to a public fixed L2 bound `C`:

```text
clipped_g = g × min(1, C / ||g||₂)
```

The clip bound must not be selected from a private, non-DP gradient quantile.

### 9.4 Local differential privacy

When round epsilon is positive, Gaussian noise shall be added to the averaged clipped gradient:

```text
σ = C × sqrt(2 ln(1.25 / δ_step)) / ε_step / n
```

For `E` local epochs, conservative basic composition shall use:

```text
ε_step = ε_round / E
δ_step = δ_round / E
```

Noise shall use a cryptographically strong random source. The default round budget is epsilon 3.0 and delta 1e-5. Reported cumulative privacy shall use conservative addition across released rounds; it shall not use an unsupported square-root shortcut.

### 9.5 No-DP mode

A non-positive epsilon may disable noise for controlled experiments. The UI/report must not claim DP when disabled.

## 10. Server functional requirements

### FR-S01 Registration

The server shall register a new client or resume a supplied client identifier after token validation.

### FR-S02 Minimum federation

A training session shall require at least two online clients. One device is personalization, not a federation.

### FR-S03 Round request

Each `update_requested` event shall include:

- round;
- local epochs;
- learning rate;
- FedProx μ;
- DP epsilon/delta;
- per-example clip norm;
- pseudo-label threshold and weight;
- minimum usable local samples.

### FR-S04 Update validation

The server shall reject:

- unregistered clients;
- stale model versions;
- wrong rounds;
- duplicates;
- invalid base64/zlib payloads;
- wrong tensor lengths or shapes;
- NaN or infinite values;
- effectively unchanged updates;
- grossly anomalous deltas.

### FR-S05 Skip handling

The coordinator shall treat a valid skip as a client response. If all clients respond but fewer than the minimum number provide usable updates, the session shall stop with `insufficient_usable_updates` instead of hanging.

### FR-S06 Robust aggregation

The server shall aggregate client deltas, not apply last-write-wins behavior.

For each client:

```text
delta_k = client_weights_k - global_weights
delta_k = L2_clip(delta_k, server_delta_clip_norm)
```

The aggregation scalar shall be:

```text
effective_count_k = trust_k × (human_count_k + pseudo_weight × pseudo_count_k)
```

The new head shall be:

```text
new_global = old_global + weighted_mean(delta_k, effective_count_k)
```

Tag-demand popularity shall not change model aggregation weights.

### FR-S07 Flower compatibility

With clipping inactive, the project's weighted FedAvg result shall numerically match Flower FedAvg for the same client parameters and example counts.

### FR-S08 Quality guard

Before committing a candidate, the server shall evaluate it on the shipped held-out validation features.

- A pseudo-label-only round must not reduce current held-out macro F1.
- A round containing trusted corrections may drop no more than the configured amount below the fixed deployment baseline; default tolerance is 0.01.
- Rejected candidates shall not increment the model version.
- The guard measures the proxy validation domain and does not replace real gallery-domain evaluation.

### FR-S09 Versioning

An accepted global update shall increment the model version and atomically persist the four tensors and version.

### FR-S10 Metrics

The server shall persist round, timestamp, client count, effective sample information, held-out loss, held-out macro F1, and per-client contribution metadata.

### FR-S11 Dashboard

The dashboard shall display server connectivity, online clients, model version, round progress, real held-out metric history, privacy settings, and baseline/current comparison. Missing metrics shall be displayed as unavailable, never fabricated.

## 11. Wire protocol

### 11.1 Tensor serialization

Normative order: `w1`, `b1`, `w2`, `b2`.

For each tensor:

1. Flatten in C/input-major order.
2. Encode float32 little-endian bytes.
3. Prefix with a little-endian 32-bit byte length.
4. Concatenate tensors.
5. Compress with zlib/Deflater.
6. Encode as Base64 without semantic transformation.

### 11.2 Model delta

The server may return a one-version delta only when the requesting version is exactly current minus one and a last delta exists. Otherwise it shall return the full head.

### 11.3 Authentication

Authenticated endpoints shall require `X-FGT-Token`. Token regeneration shall revoke registrations and disconnect active clients.

## 12. API requirements

| Method | Endpoint | Authentication | Requirement |
|---|---|---|---|
| POST | `/api/register` | token | register/resume client |
| GET | `/api/model/current` | none on trusted LAN | full or delta head |
| GET | `/api/model/schema` | none | canonical tensor contract |
| GET | `/api/model/status` | none | model availability/version |
| POST | `/api/training/start` | dashboard/LAN | start session |
| POST | `/api/training/stop` | dashboard/LAN | stop session |
| POST | `/api/training/submit-update` | token | submit full locally trained head |
| POST | `/api/training/skip-update` | token | report insufficient usable data |
| GET | `/api/training/status` | none | session state and clients |
| GET | `/api/metrics/history` | none | persisted round metrics |
| GET | `/api/metrics/comparison` | none | baseline/current parent F1 |
| GET | `/api/metrics/summary` | none | session totals |
| GET | `/api/metrics/leaderboard` | none | contributions |
| GET | `/api/taxonomy` | none | detailed taxonomy |
| POST | `/api/taxonomy/signal` | token | local demand event |
| WS | `/ws/feed` | token for clients | heartbeats and round events |

## 13. Data requirements

### 13.1 Server model artifacts

Required files:

- `models/base_model.tflite`
- `models/head_weights.npz`
- `models/initial_head_weights.npz`
- `models/model_schema.json`
- `models/model_version.txt`
- `models/thresholds.json`
- `models/validation_features.npz`
- `models/baseline_metrics.json`

### 13.2 Android persisted data

- Four head tensors and model version in app-private storage.
- Feedback record keyed by image ID.
- Scan results.
- Media favorite/archive/trash state.
- Organization operation log.
- Per-class demand, confirmation, and rejection counters.

### 13.3 Data minimization

No raw image, thumbnail, URI, EXIF block, filename, or 1024-D image feature shall be transmitted in an FL update.

## 14. Non-functional requirements

### NFR-01 Privacy

Source images shall remain on device. Logs shall not contain image content. Local DP claims shall include actual epsilon/delta configuration.

### NFR-02 Honesty

The product shall not describe individual-update transport as secure aggregation. It shall not claim guaranteed learning from unlabelled data or guaranteed real-gallery accuracy.

### NFR-03 Reliability

Model writes shall be atomic. Stale and duplicate updates shall not alter global state. A zero-update round shall terminate cleanly.

### NFR-04 Performance

The backbone shall use LiteRT acceleration when available. Local FL shall train only the output layer. Feature extraction and training shall run off the Android main thread.

### NFR-05 Portability

The server shall run on Windows PowerShell, Linux, and macOS with Python 3.11+. Android shall build with SDK 36 and JDK 21.

### NFR-06 Maintainability

The repository shall retain only core source, required model artifacts, this SRS, and the main README. Temporary verification, test, retraining, cache, and planning files shall not ship in the final repository.

### NFR-07 Accessibility

Interactive Android and dashboard controls shall have meaningful labels/content descriptions and visible loading/error states.

## 15. Security requirements

- Validate every tensor shape and finite value.
- Validate decompressed payload lengths and reject trailing bytes.
- Use delta norm for anomaly detection.
- Bound every accepted client delta before aggregation.
- Key rate limiting by client and round.
- Reject clients not registered for the active round.
- Store no GitHub or external service credentials in source.
- Treat the LAN server as trusted-but-curious when local DP is enabled.
- Document absence of secure aggregation.

## 16. Federated-learning verification evidence

### 16.1 Flower numerical compatibility

The project's sample-weighted FedAvg was compared with Flower 1.32 reference FedAvg. With clipping inactive, results matched within float tolerance. Bounded-delta tests confirmed the project limits a maliciously large client update before averaging.

### 16.2 Flower Non-IID unlabelled simulation

Scenario:

- Five clients.
- Dirichlet alpha 0.25 class skew.
- Client-side labels hidden.
- Confidence-gated pseudo-labels only.
- Pseudo-label weight 0.25.
- Fixed clip norm 1.0.
- Local DP epsilon 3.0, delta 1e-5 per round.
- Flower reference FedAvg after project delta clipping.
- Held-out non-regression gate.

Result:

| Metric | Baseline | Final | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.299729 | 0.302369 | +0.002639 |
| Accuracy | 0.326846 | 0.330872 | +0.004027 |

Five of ten candidate rounds improved and were committed. Regressing rounds were rejected. This validates cautious semi-supervised refinement on the committed proxy domain; the small gain demonstrates why explicit corrections remain necessary.

### 16.3 Live transport run

A four-client live FastAPI/REST/WebSocket run completed eight rounds. All clients registered, downloaded and decoded the head, performed local optimization, serialized updates, submitted successfully, triggered aggregation, received round events, and completed the session. The production model was restored afterward.

### 16.4 Central model baseline

The deployed model's held-out macro F1 is 0.3055 versus the historical broken pipeline's 0.033. The runtime validation path reproduces 0.3055 without TensorFlow.

### 16.5 SRS-to-implementation conformance audit

The version 3.0 audit executed direct assertions over the shipped artifacts and source contract. It passed all of the following checks:

- exactly `README.md` and `SRS.md` remain as Markdown documentation;
- seven labels and their order agree across taxonomy, schema, Android, and model evaluator;
- production tensor shapes are `(1024,256)`, `(256,)`, `(256,7)`, `(7,)`;
- Android and server backbone SHA-256 values are identical: `24d30971cb279c7aeaf5700dafc8c3aa558a172a4fcd8436149f033a24687e50`;
- runtime evaluation reproduces macro F1 `0.3054973780` and accuracy `0.3406704362` on 4,952 samples;
- thresholds use argmax/post-argmax abstention semantics;
- the FL configuration requires at least two clients and reduced pseudo-label weight;
- Android source contains softmax, output-only backpropagation, pseudo/human sample accounting, fixed DP clipping, and skip handling;
- server source contains the required routes, clipped weighted FedAvg, candidate evaluation, and commit guard;
- Android dependencies match the declared stack and contain no Hilt, Paging, or residual test dependencies;
- self-contained dashboard code contains no Chart.js runtime dependency;
- model serialization/deserialization is byte-symmetric;
- weighted aggregation and client-delta clipping produce the expected numerical results.

Audit result: **IMPLEMENTATION_CONFORMANCE_PASS**.

## 17. Known limitations

1. Real gallery images are domain-shifted relative to COCO.
2. Documents and Events are weak in the current bootstrap model.
3. Pseudo-labeling cannot correct a confidently wrong teacher without an external signal.
4. The held-out guard evaluates the proxy domain, not every user's private gallery.
5. Local DP may reduce utility for clients with very few accepted examples.
6. Secure aggregation is not implemented.
7. At least two devices are required for a federated round.
8. A single-server LAN deployment has no high-availability design.

## 18. Acceptance criteria

### AC-01 Model contract

Server and Android use the same backbone hash, class order, tensor order, shapes, dtype, byte order, compression, and version semantics.

### AC-02 Central quality

The deployed held-out macro F1 is at least 0.20 and all seven classes are reported.

### AC-03 Local data safety

No image or image feature is present in client update requests.

### AC-04 Non-IID FL

At least two clients with different class distributions can complete multiple rounds without stale-version, rate-limit, shape, or serialization failure.

### AC-05 Unlabelled safety

Low-confidence examples and negative-only feedback are excluded. Pseudo-label-only candidates cannot reduce held-out macro F1.

### AC-06 Correction learning

A user-selected correction becomes a one-hot trusted target in the next eligible round and receives full sample weight.

### AC-07 DP correctness

The local trainer uses fixed per-example clipping, secure Gaussian randomness, average-gradient sensitivity `C/n`, and explicit composition across local epochs and rounds.

### AC-08 Aggregation correctness

With clipping disabled, server aggregation matches Flower weighted FedAvg. With clipping enabled, each client's full delta norm is bounded.

### AC-09 Round liveness

Clients may submit or skip. A round shall aggregate when enough usable updates exist and shall terminate cleanly otherwise.

### AC-10 Quality commit

A rejected candidate shall not overwrite the head or increment model version.

### AC-11 Runtime startup

The server loads required artifacts and exposes status, schema, dashboard, and model endpoints without importing a training framework.

### AC-12 Android build

The Android project targets SDK 36 and compiles with JDK 21 in an environment with sufficient Gradle/Kotlin compiler memory.

## 19. Operational configuration

Default server FL values:

```json
{
  "min_clients": 2,
  "max_rounds": 10,
  "local_epochs": 1,
  "learning_rate": 0.05,
  "mu": 0.01,
  "dp_epsilon": 3.0,
  "dp_delta": 0.00001,
  "max_grad_norm": 1.0,
  "pseudo_label_threshold": 0.8,
  "pseudo_label_weight": 0.25,
  "min_local_samples": 8,
  "server_delta_clip_norm": 1.0,
  "max_global_f1_drop": 0.01
}
```

Changing privacy or learning values changes the privacy/utility trade-off and requires revalidation.

## 20. Traceability summary

| Requirement area | Primary implementation |
|---|---|
| API and authentication | `server/main.py` |
| Round lifecycle | `server/fl_coordinator.py` |
| Aggregation | `server/fl_math.py` |
| Validation metrics | `server/model_eval.py` |
| Model persistence/wire format | `server/model_manager.py` |
| Update security | `server/security.py` |
| Dashboard | `server/dashboard/` |
| Android orchestration | `MainActivity.kt` |
| Head math | `ClassificationHead.kt` |
| Local optimization/DP | `LocalTrainer.kt`, `DPNoiseInjector.kt` |
| Pseudo-labeling | `PseudoLabelGenerator.kt` |
| Corrections | `RecordTagFeedbackUseCase.kt`, `ImageDetailScreen.kt` |
| Feature extraction | `FeatureExtractor.kt` |
| Model taxonomy | `TaxonomyConfig.kt`, `server/taxonomy.json` |
| Gallery organization | `OrganizeExecutor.kt` and related local components |

---

**End of SRS**
