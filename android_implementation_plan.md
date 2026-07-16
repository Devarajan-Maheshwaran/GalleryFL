# FGT Android Client - Implementation Plan

## Part 1: Architecture & Backend Logic

---

### 1. Project Setup

#### Gradle Configuration

```kotlin
// build.gradle.kts (app-level)
android {
    namespace = "com.fgt.galleryfl"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.fgt.galleryfl"
        minSdk = 29        // Android 10+ (Scoped Storage baseline)
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
    }

    buildFeatures {
        compose = true
    }

    // Prevent model file compression
    androidResources {
        noCompress += "tflite"
    }
}
```

#### Dependencies

| Category | Library | Version | Purpose |
|:---|:---|:---|:---|
| **Compose** | `androidx.compose.material3:material3` | 1.3+ | Material3 theming |
| | `androidx.compose.ui:ui` | 1.9+ | Core Compose (dropShadow support) |
| | `androidx.compose.foundation:foundation` | 1.9+ | Layouts, gestures |
| | `androidx.navigation:navigation-compose` | 2.8+ | Type-safe navigation |
| | `androidx.compose.animation:animation` | 1.9+ | Transitions |
| **ML** | `com.google.ai.edge.litert:litert` | latest | LiteRT runtime |
| | `com.google.ai.edge.litert:litert-gpu` | latest | GPU delegate |
| **Network** | `com.squareup.retrofit2:retrofit` | 2.11+ | REST client |
| | `com.squareup.retrofit2:converter-moshi` | 2.11+ | JSON parsing |
| | `com.squareup.okhttp3:okhttp` | 4.12+ | WebSocket + HTTP |
| | `com.squareup.okhttp3:logging-interceptor` | 4.12+ | Debug logging |
| **Image** | `io.coil-kt.coil3:coil-compose` | 3.0+ | Async image loading |
| **DI** | `com.google.dagger:hilt-android` | 2.52+ | Dependency injection |
| **DB** | `androidx.room:room-runtime` | 2.7+ | Local database |
| | `androidx.room:room-ktx` | 2.7+ | Coroutine extensions |
| **Paging** | `androidx.paging:paging-compose` | 3.3+ | Lazy gallery loading |
| **Charts** | `com.patrykandpatrick.vico:compose-m3` | 2.0+ | Compose-native charts |
| **EXIF** | `androidx.exifinterface:exifinterface` | 1.3+ | EXIF metadata |

---

### 2. Architecture Layers (MVVM + Clean)

```
┌─────────────────────────────────────────────────┐
│                    UI Layer                      │
│   Compose Screens + ViewModels                   │
│   (Observes StateFlow, dispatches intents)       │
├─────────────────────────────────────────────────┤
│                  Domain Layer                    │
│   Use Cases + Domain Models                      │
│   (Pure Kotlin, no Android deps)                 │
├─────────────────────────────────────────────────┤
│                   Data Layer                     │
│   Repositories + Data Sources                    │
│   Network (Retrofit/WS) | Local (Room/MediaStore)│
│   ML (LiteRT inference + training)               │
└─────────────────────────────────────────────────┘
```

**Data flow:** Screen -> ViewModel -> UseCase -> Repository -> DataSource -> (Network | MediaStore | LiteRT | Room)

---

### 3. ML Engine

#### 3.1 FeatureExtractor

```
File: data/ml/FeatureExtractor.kt
```

**Purpose:** Runs frozen MobileNetV3-Small backbone to extract 1024-dim feature vectors from images.

**Implementation:**
- Load `base_model.tflite` from `assets/models/`
- Input: `Bitmap` scaled to 224x224x3, normalized to [-1, 1]
- Output: `FloatArray(1024)` — feature vector
- Uses `Interpreter` with GPU delegate if available, falls back to CPU
- Thread-safe: single interpreter instance per `FeatureExtractor`, synchronized access
- Batch processing: accepts `List<Bitmap>`, processes sequentially on `Dispatchers.Default`

```kotlin
class FeatureExtractor(context: Context) {
    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer   // 1 x 224 x 224 x 3 x 4 bytes
    private val outputBuffer: ByteBuffer  // 1 x 1024 x 4 bytes

    fun extractFeatures(bitmap: Bitmap): FloatArray
    fun extractFeaturesBatch(images: List<Pair<Uri, Bitmap>>): List<Pair<Uri, FloatArray>>
    fun close()
}
```

#### 3.2 ClassificationHead

```
File: data/ml/ClassificationHead.kt
```

**Purpose:** Lightweight trainable head (1024 -> 256 -> num_classes) that sits on top of the frozen feature extractor.

**Implementation:**
- Pure Kotlin/math implementation (no TF dependency for head):
  - Layer 1: Dense 1024 -> 256, ReLU activation
  - Layer 2: Dense 256 -> num_classes, Sigmoid activation (multi-label)
- Weight storage: `Array<FloatArray>` for each layer's weights and biases
- Forward pass: matrix multiply + activation
- Backward pass: compute gradients via chain rule (sigmoid derivative -> ReLU derivative)
- `getWeights(): List<FloatArray>` — extract weights for FL transmission
- `setWeights(weights: List<FloatArray>)` — load weights from server
- `predict(features: FloatArray): FloatArray` — output tag probabilities

```kotlin
class ClassificationHead(numClasses: Int) {
    // Weights
    private var w1: Array<FloatArray>  // [1024 x 256]
    private var b1: FloatArray         // [256]
    private var w2: Array<FloatArray>  // [256 x numClasses]
    private var b2: FloatArray         // [numClasses]

    fun forward(features: FloatArray): FloatArray
    fun backward(features: FloatArray, targets: FloatArray, lr: Float): Gradients
    fun getWeights(): List<FloatArray>
    fun setWeights(weights: List<FloatArray>)
}
```

> [!NOTE]
> Implementing the head in pure Kotlin avoids TFLite on-device training API complexity. The head is tiny (~262K parameters) so matrix ops on CPU are negligible. This approach gives us full control over gradient extraction for FL.

#### 3.3 LocalTrainer

```
File: data/ml/LocalTrainer.kt
```

**Purpose:** Runs the on-device training loop with FedProx proximal term.

**FedProx Loss:**
```
L_local = BCE(y_pred, y_true) + (mu/2) * ||w_local - w_global||^2
```

**Training loop per round:**
1. Receive global weights from server -> set on `ClassificationHead`
2. Save a copy of global weights as `w_global_ref`
3. For each local epoch (K epochs):
   a. Shuffle training data
   b. For each mini-batch:
      - Extract features via `FeatureExtractor` (cached from initial scan)
      - Forward pass through `ClassificationHead`
      - Compute BCE loss + FedProx proximal term
      - Backward pass to get gradients
      - Update weights with SGD: `w -= lr * (grad + mu * (w - w_global_ref))`
4. Compute local validation accuracy on held-out 20% split
5. Return updated weights + metrics

```kotlin
class LocalTrainer(
    private val featureExtractor: FeatureExtractor,
    private val head: ClassificationHead,
    private val config: TrainingConfig
) {
    suspend fun train(
        dataset: List<LabeledImage>,
        globalWeights: List<FloatArray>,
        onProgress: (epoch: Int, loss: Float) -> Unit
    ): TrainingResult

    data class TrainingResult(
        val updatedWeights: List<FloatArray>,
        val numSamples: Int,
        val localLoss: Float,
        val localAccuracy: Float,
        val perClassAccuracy: Map<String, Float>
    )
}
```

**Device-aware adaptations:**
- Query `ActivityManager.memoryClass` to set batch size (8 for low-RAM, 16 for 4GB+, 32 for 8GB+)
- Check `BatteryManager` — reduce epochs if battery < 20%
- Check thermal status via `PowerManager` — reduce batch size if throttling

#### 3.4 DPNoiseInjector

```
File: data/ml/DPNoiseInjector.kt
```

**Purpose:** Adds calibrated Gaussian noise to gradients before transmission.

```kotlin
class DPNoiseInjector(private val epsilon: Float, private val delta: Float) {
    // Calibrate sigma based on sensitivity and privacy budget
    // sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    fun addNoise(gradients: List<FloatArray>, sensitivity: Float): List<FloatArray>
}
```

#### 3.5 GradientClipper

```
File: data/ml/GradientClipper.kt
```

**Purpose:** Bounds the L2 norm of weight updates to prevent poisoning and limit DP sensitivity.

```kotlin
class GradientClipper(private val maxNorm: Float) {
    fun clip(gradients: List<FloatArray>): List<FloatArray> {
        val norm = computeL2Norm(gradients)
        if (norm > maxNorm) {
            val scale = maxNorm / norm
            return gradients.map { layer -> layer.map { it * scale }.toFloatArray() }
        }
        return gradients
    }
}
```

#### 3.6 PseudoLabelGenerator

```
File: data/ml/PseudoLabelGenerator.kt
```

**Purpose:** Uses the pre-trained base model to auto-label images before FL training begins.

**Flow:**
1. User selects gallery folders
2. For each image: extract features -> run classification head (with pre-trained weights) -> get tag probabilities
3. Apply threshold (0.5 default) to produce multi-label tags
4. Store as `PseudoLabel(uri, tags: Map<String, Float>, confirmed: Boolean)`
5. Present in review UI for correction

#### 3.7 GradCAMGenerator

```
File: data/ml/GradCAMGenerator.kt
```

**Purpose:** Generates attention heatmaps showing which image regions influenced each tag prediction.

**Simplified approach (no full GradCAM, as we have a frozen backbone):**
1. After feature extraction, take the spatial feature maps (before GlobalAveragePooling) — shape `[7, 7, C]`
2. For a given class, multiply each spatial feature channel by the corresponding weight from the classification head's first layer
3. Sum across channels -> 7x7 activation map
4. Upsample to 224x224 via bilinear interpolation
5. Normalize to [0, 1] and apply warm colormap (transparent -> gold -> terracotta -> deep brown)
6. Overlay on original image with alpha blending

```kotlin
class GradCAMGenerator(
    private val featureExtractor: FeatureExtractor,
    private val head: ClassificationHead
) {
    fun generateHeatmap(bitmap: Bitmap, classIndex: Int): Bitmap
}
```

#### 3.8 AestheticScorer

```
File: data/ml/AestheticScorer.kt
```

**Purpose:** Selects the best cover photo for each auto-generated album.

**Simple scoring (no additional model needed):**
```
score = 0.4 * sharpness + 0.3 * brightness_balance + 0.2 * face_presence + 0.1 * aspect_ratio_fit
```

- **Sharpness:** Laplacian variance of grayscale image
- **Brightness balance:** Distance from mean brightness to 128 (center)
- **Face presence:** Binary (1 if face detected via Android `FaceDetector`)
- **Aspect ratio:** Penalty for extreme aspect ratios

```kotlin
class AestheticScorer {
    fun scoreImage(bitmap: Bitmap): Float  // 0.0 to 1.0
    fun selectBestCover(images: List<Pair<Uri, Bitmap>>): Uri
}
```

---

### 4. FL Client Protocol (Network Layer)

#### 4.1 REST API Client

```
File: data/network/FGTApiService.kt
```

```kotlin
interface FGTApiService {
    @POST("api/register")
    suspend fun register(@Body request: RegisterRequest): RegisterResponse

    @GET("api/model/current")
    suspend fun getCurrentModel(): ResponseBody  // Raw bytes (compressed TFLite weights)

    @GET("api/model/delta/{fromVersion}")
    suspend fun getModelDelta(@Path("fromVersion") version: Int): ResponseBody

    @POST("api/training/submit-update")
    suspend fun submitUpdate(@Body update: ClientUpdateRequest): UpdateResponse

    @GET("api/training/status")
    suspend fun getTrainingStatus(): TrainingStatusResponse

    @GET("api/metrics/history")
    suspend fun getMetricsHistory(): MetricsHistoryResponse

    @GET("api/metrics/leaderboard")
    suspend fun getLeaderboard(): LeaderboardResponse

    @GET("api/metrics/comparison")
    suspend fun getComparison(): ComparisonResponse

    @GET("api/export/report")
    suspend fun getTrainingReport(): TrainingReportResponse
}
```

#### 4.2 Weight Serializer

```
File: data/network/WeightSerializer.kt
```

**Critical: Endianness enforcement.**

```kotlin
object WeightSerializer {
    // Android -> Server: FloatArray -> Little-Endian ByteBuffer -> zlib compress -> Base64
    fun serialize(weights: List<FloatArray>): String {
        val buffers = weights.map { layer ->
            ByteBuffer.allocate(layer.size * 4)
                .order(ByteOrder.LITTLE_ENDIAN)  // CRITICAL
                .apply { layer.forEach { putFloat(it) } }
                .array()
        }
        val combined = combineLayers(buffers)  // Prepend each layer's size as Int32
        val compressed = Deflater().deflate(combined)  // zlib
        return Base64.encodeToString(compressed, Base64.NO_WRAP)
    }

    // Server -> Android: Base64 -> zlib decompress -> Little-Endian ByteBuffer -> FloatArray
    fun deserialize(encoded: String, layerSizes: List<Int>): List<FloatArray>
}
```

#### 4.3 WebSocket Client

```
File: data/network/FGTWebSocketClient.kt
```

```kotlin
class FGTWebSocketClient(private val okHttpClient: OkHttpClient) {
    private var webSocket: WebSocket? = null
    private val _events = MutableSharedFlow<FLEvent>(replay = 1)
    val events: SharedFlow<FLEvent> = _events

    fun connect(serverUrl: String)
    fun disconnect()

    // Event types received from server
    sealed class FLEvent {
        data class RoundStarted(val round: Int, val totalRounds: Int) : FLEvent()
        data class UpdateRequested(val round: Int) : FLEvent()
        data class RoundCompleted(val round: Int, val metrics: RoundMetrics) : FLEvent()
        data class TrainingComplete(val finalMetrics: FinalMetrics) : FLEvent()
        data class ClientUpdate(val clientId: String, val event: String) : FLEvent()
        data class Error(val message: String) : FLEvent()
    }
}
```

#### 4.4 Device Profiler

```
File: data/network/DeviceProfiler.kt
```

Sends device capabilities to server for intelligent client selection.

```kotlin
data class DeviceProfile(
    val deviceModel: String,         // Build.MODEL
    val androidVersion: Int,          // Build.VERSION.SDK_INT
    val totalRamMb: Int,              // ActivityManager.totalMem
    val availableRamMb: Int,          // ActivityManager.availMem
    val batteryPercent: Int,           // BatteryManager
    val isThermalThrottling: Boolean,  // PowerManager.thermalStatus
    val hasGpuDelegate: Boolean        // LiteRT GPU delegate availability
)
```

---

### 5. Gallery Engine

#### 5.1 GalleryRepository

```
File: data/local/GalleryRepository.kt
```

**Responsibilities:**
- Query albums (distinct `BUCKET_DISPLAY_NAME` from MediaStore)
- Query images within selected albums (with Paging 3)
- Provide `Flow<PagingData<GalleryImage>>` for lazy UI loading

```kotlin
class GalleryRepository(private val context: Context) {
    fun getAlbums(): Flow<List<Album>>
    fun getImagesInAlbums(albumIds: List<Long>): Flow<PagingData<GalleryImage>>
    fun getImageBitmap(uri: Uri, targetSize: Int = 224): Bitmap
    fun getImageCount(albumIds: List<Long>): Int
}

data class Album(
    val id: Long,
    val name: String,
    val coverUri: Uri,
    val imageCount: Int
)

data class GalleryImage(
    val id: Long,
    val uri: Uri,
    val displayName: String,
    val albumName: String,
    val dateAdded: Long,
    val width: Int,
    val height: Int
)
```

#### 5.2 AlbumCreator

```
File: data/local/AlbumCreator.kt
```

**Approach:** Copy tagged images into new FGT albums via MediaStore.

```kotlin
class AlbumCreator(private val context: Context) {
    /**
     * Creates a copy of the image in Pictures/FGT/<tagName>/
     * This appears as a new album in the default gallery.
     */
    suspend fun createTagAlbum(tagName: String, imageUris: List<Uri>): AlbumResult

    /**
     * Sets the best cover image for an album based on aesthetic scoring.
     */
    suspend fun setAlbumCover(albumPath: String, coverUri: Uri)

    data class AlbumResult(
        val albumPath: String,
        val copiedCount: Int,
        val failedUris: List<Uri>,
        val newUris: List<Uri>  // URIs of copies (for revert)
    )
}
```

**MediaStore insert pattern:**
```kotlin
val values = ContentValues().apply {
    put(MediaStore.Images.Media.DISPLAY_NAME, "FGT_${originalName}")
    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/FGT/$tagName")
}
val newUri = contentResolver.insert(
    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
)
// Copy bytes from original URI to new URI
```

#### 5.3 ExifTagWriter

```
File: data/local/ExifTagWriter.kt
```

```kotlin
class ExifTagWriter(private val context: Context) {
    /**
     * Writes tag metadata as JSON into EXIF USER_COMMENT.
     * Only works on images created by this app (FGT copies).
     */
    suspend fun writeTags(uri: Uri, tags: List<TagResult>) {
        val pfd = context.contentResolver.openFileDescriptor(uri, "rw")
        pfd?.use {
            val exif = ExifInterface(it.fileDescriptor)
            val tagJson = Json.encodeToString(tags)
            exif.setAttribute(ExifInterface.TAG_USER_COMMENT, tagJson)
            exif.saveAttributes()
        }
    }
}

@Serializable
data class TagResult(
    val category: String,      // Parent category (e.g., "Places")
    val subcategory: String,   // Leaf tag (e.g., "beach")
    val confidence: Float      // 0.0 to 1.0
)
```

#### 5.4 RevertManager (Room Database)

```
File: data/local/RevertDatabase.kt
```

**Schema:**

```kotlin
@Entity(tableName = "operation_log")
data class OperationLogEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,            // Groups operations from one "Apply" action
    val operationType: String,        // "COPY_TO_ALBUM" | "WRITE_EXIF"
    val originalUri: String,          // Source image URI
    val newUri: String?,              // Created copy URI (for COPY_TO_ALBUM)
    val previousExifData: String?,    // Previous EXIF USER_COMMENT (for WRITE_EXIF)
    val tagData: String,              // JSON of applied tags
    val timestamp: Long,              // System.currentTimeMillis()
    val reverted: Boolean = false     // Whether this operation has been undone
)

@Dao
interface OperationLogDao {
    @Query("SELECT * FROM operation_log WHERE reverted = 0 ORDER BY timestamp DESC")
    fun getActiveOperations(): Flow<List<OperationLogEntry>>

    @Query("SELECT DISTINCT sessionId FROM operation_log WHERE reverted = 0 ORDER BY timestamp DESC")
    fun getActiveSessions(): Flow<List<String>>

    @Insert
    suspend fun insert(entry: OperationLogEntry)

    @Query("UPDATE operation_log SET reverted = 1 WHERE sessionId = :sessionId")
    suspend fun revertSession(sessionId: String)
}
```

**Revert logic:**
1. Query all operations for the session
2. For `COPY_TO_ALBUM`: delete the copied file via `contentResolver.delete(newUri)`
3. For `WRITE_EXIF`: restore previous EXIF data
4. Mark session as reverted in Room

#### 5.5 ExportManifest

```
File: data/local/ExportManifest.kt
```

```kotlin
class ExportManifest {
    fun generateCSV(operations: List<OperationLogEntry>): String
    fun generateJSON(operations: List<OperationLogEntry>): String
    fun saveToDownloads(context: Context, content: String, filename: String): Uri
}
```

---

### 6. Domain Layer

#### 6.1 Use Cases

| Use Case | Input | Output | Logic |
|:---|:---|:---|:---|
| `ConnectToServerUseCase` | IP, port, nickname | `ConnectionResult` | Test connection, register, get model version |
| `ScanGalleryUseCase` | List of album IDs | `Flow<ScanProgress>` | Load images, extract features, generate pseudo-labels |
| `ReviewLabelsUseCase` | User corrections | Updated `LabeledImage` list | Merge user corrections with pseudo-labels |
| `JoinTrainingUseCase` | Training config | `Flow<TrainingProgress>` | Full FL round loop: download model -> train -> clip -> noise -> upload |
| `OrganizeGalleryUseCase` | List of `OrganizeOp` | `Flow<OrganizeProgress>` | Create albums, copy files, write EXIF, log to Room |
| `RevertOperationsUseCase` | Session ID | `RevertResult` | Undo all operations in a session |
| `ExportReportUseCase` | Report data | File URI | Generate and save training report |
| `GetPrivacyInfoUseCase` | None | `PrivacyInfo` | Compute what data was sent, DP budget used |

---

## Part 2: Android Frontend / UI

---

### 7. Neumorphic Compose Design System

#### 7.1 Color Palette

```kotlin
// ui/theme/Color.kt
object FGTColors {
    // Backgrounds (match web dashboard exactly)
    val BgBase = Color(0xFFF0E4D7)
    val BgSurface = Color(0xFFEAD9C8)
    val BgSidebar = Color(0xFFE5D3C1)

    // Neumorphic shadows
    val ShadowLight = Color(0xFFFDFAF6)
    val ShadowDark = Color(0xFFC9B9A5)
    val ShadowLightStrong = Color(0xFFFFFFFF)
    val ShadowDarkStrong = Color(0xFFB5A48E)

    // Accents
    val AccentPrimary = Color(0xFFD4845A)     // Terracotta
    val AccentGold = Color(0xFFE8B87A)        // Warm gold
    val AccentDeep = Color(0xFF8B5E3C)        // Deep brown
    val AccentRose = Color(0xFFC97B7B)        // Muted rose
    val AccentSage = Color(0xFF7A9E7E)        // Sage green

    // Text
    val TextPrimary = Color(0xFF3D2B1F)
    val TextSecondary = Color(0xFF7A6355)
    val TextMuted = Color(0xFFA89485)

    // Semantic
    val Success = Color(0xFF6B8F5E)
    val Warning = Color(0xFFC4954A)
    val Error = Color(0xFFB85C4A)
}
```

#### 7.2 Shadow Modifiers

Using Compose 1.9+ native `dropShadow` and `innerShadow`:

```kotlin
// ui/theme/NeuModifiers.kt

// Raised (default resting state for cards, buttons)
fun Modifier.neuRaised(
    cornerRadius: Dp = 12.dp
): Modifier = this
    .dropShadow(
        shape = RoundedCornerShape(cornerRadius),
        color = FGTColors.ShadowDark,
        blur = 12.dp,
        offsetX = 6.dp,
        offsetY = 6.dp
    )
    .dropShadow(
        shape = RoundedCornerShape(cornerRadius),
        color = FGTColors.ShadowLight,
        blur = 12.dp,
        offsetX = (-6).dp,
        offsetY = (-6).dp
    )

// Pressed (active/clicked state)
fun Modifier.neuPressed(
    cornerRadius: Dp = 12.dp
): Modifier = this
    .innerShadow(
        shape = RoundedCornerShape(cornerRadius),
        color = FGTColors.ShadowDark,
        blur = 8.dp,
        offsetX = 4.dp,
        offsetY = 4.dp
    )
    .innerShadow(
        shape = RoundedCornerShape(cornerRadius),
        color = FGTColors.ShadowLight,
        blur = 8.dp,
        offsetX = (-4).dp,
        offsetY = (-4).dp
    )

// Concave (inset fields, input containers)
fun Modifier.neuConcave(
    cornerRadius: Dp = 12.dp
): Modifier = this
    .innerShadow(
        shape = RoundedCornerShape(cornerRadius),
        color = FGTColors.ShadowDark,
        blur = 6.dp,
        offsetX = 3.dp,
        offsetY = 3.dp
    )
    .innerShadow(
        shape = RoundedCornerShape(cornerRadius),
        color = FGTColors.ShadowLight,
        blur = 6.dp,
        offsetX = (-3).dp,
        offsetY = (-3).dp
    )
```

#### 7.3 Typography

```kotlin
// ui/theme/Type.kt
val InterFont = FontFamily(
    Font(R.font.inter_regular, FontWeight.Normal),
    Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold),
    Font(R.font.inter_bold, FontWeight.Bold),
)

val OutfitFont = FontFamily(
    Font(R.font.outfit_regular, FontWeight.Normal),
    Font(R.font.outfit_medium, FontWeight.Medium),
)

val JetBrainsMonoFont = FontFamily(
    Font(R.font.jetbrains_mono_regular, FontWeight.Normal),
)

val FGTTypography = Typography(
    headlineLarge = TextStyle(fontFamily = InterFont, fontWeight = FontWeight.Bold, fontSize = 28.sp),
    headlineMedium = TextStyle(fontFamily = InterFont, fontWeight = FontWeight.SemiBold, fontSize = 20.sp),
    headlineSmall = TextStyle(fontFamily = InterFont, fontWeight = FontWeight.SemiBold, fontSize = 16.sp),
    bodyLarge = TextStyle(fontFamily = OutfitFont, fontWeight = FontWeight.Normal, fontSize = 16.sp),
    bodyMedium = TextStyle(fontFamily = OutfitFont, fontWeight = FontWeight.Normal, fontSize = 14.sp),
    bodySmall = TextStyle(fontFamily = OutfitFont, fontWeight = FontWeight.Normal, fontSize = 12.sp),
    labelMedium = TextStyle(fontFamily = OutfitFont, fontWeight = FontWeight.Medium, fontSize = 12.sp),
)
```

#### 7.4 Theme

```kotlin
// ui/theme/Theme.kt
@Composable
fun FGTTheme(content: @Composable () -> Unit) {
    val colorScheme = lightColorScheme(
        primary = FGTColors.AccentPrimary,
        onPrimary = Color.White,
        secondary = FGTColors.AccentGold,
        tertiary = FGTColors.AccentDeep,
        background = FGTColors.BgBase,
        surface = FGTColors.BgBase,  // MUST match background for neumorphism
        onBackground = FGTColors.TextPrimary,
        onSurface = FGTColors.TextPrimary,
        error = FGTColors.Error,
    )

    MaterialTheme(
        colorScheme = colorScheme,
        typography = FGTTypography,
        content = content
    )
}
```

---

### 8. Reusable Composable Components

#### 8.1 NeuSurface

Base neumorphic surface. All other components build on this.

```kotlin
@Composable
fun NeuSurface(
    modifier: Modifier = Modifier,
    shape: NeuShape = NeuShape.Raised,     // Raised | Pressed | Concave | Flat
    cornerRadius: Dp = 12.dp,
    content: @Composable () -> Unit
)
```

#### 8.2 NeuButton

```kotlin
@Composable
fun NeuButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    variant: NeuButtonVariant = NeuButtonVariant.Primary,  // Primary | Secondary | Danger
    enabled: Boolean = true,
    loading: Boolean = false,                               // Shows spinner, disables click
    icon: ImageVector? = null                                // Leading icon
)
```

**Interaction states:**
- Default: `neuRaised` shadow
- Pressed (`interactionSource.collectIsPressedAsState()`): animate to `neuPressed` + `scale(0.98f)`
- Disabled: no shadow, `alpha(0.5f)`
- Loading: indeterminate circular progress inside button, text hidden

**Animation:** `animateFloatAsState` for shadow transition, 200ms `tween`

#### 8.3 NeuCard

```kotlin
@Composable
fun NeuCard(
    modifier: Modifier = Modifier,
    title: String? = null,
    badge: String? = null,                  // "Live", "Round 3", etc.
    badgeColor: Color = FGTColors.Success,
    content: @Composable ColumnScope.() -> Unit
)
```

#### 8.4 NeuTextField

```kotlin
@Composable
fun NeuTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    modifier: Modifier = Modifier,
    keyboardType: KeyboardType = KeyboardType.Text,
    trailingIcon: @Composable (() -> Unit)? = null,
    isError: Boolean = false,
    errorMessage: String? = null
)
```

**Appearance:** `neuConcave` background. On focus: left border accent in `AccentPrimary`.

#### 8.5 NeuProgressBar

```kotlin
@Composable
fun NeuProgressBar(
    progress: Float,        // 0f to 1f
    modifier: Modifier = Modifier,
    label: String? = null,  // "Round 6 / 10"
    showPercentage: Boolean = true
)
```

**Track:** `neuConcave`, 8dp height, pill shape.
**Fill:** `neuRaised`, gradient `AccentGold -> AccentPrimary`. Width animated via `animateFloatAsState`.

#### 8.6 NeuToggle

```kotlin
@Composable
fun NeuToggle(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    label: String,
    modifier: Modifier = Modifier
)
```

**Track:** `neuConcave`, 48x24dp. On-state has faint `AccentPrimary` tint.
**Thumb:** `neuRaised`, 20dp circle. Animated `translateX` with `spring()` animation.

#### 8.7 NeuChip

```kotlin
@Composable
fun NeuChip(
    label: String,
    category: TagCategory,   // Determines left border color
    modifier: Modifier = Modifier,
    selected: Boolean = false,
    onClick: (() -> Unit)? = null
)
```

#### 8.8 NeuStatusDot

```kotlin
@Composable
fun NeuStatusDot(
    status: ConnectionStatus,  // Online | Offline | Training
    modifier: Modifier = Modifier,
    size: Dp = 10.dp
)
```

**Online:** `Success` color, infinite pulse animation (`scaleIn` 1f -> 1.4f, `alpha` 1f -> 0f, 2000ms).
**Training:** `AccentGold`, faster pulse (1000ms).
**Offline:** `Error`, static, no animation.

#### 8.9 NeuBottomBar

```kotlin
@Composable
fun NeuBottomBar(
    currentRoute: String,
    onNavigate: (String) -> Unit,
    modifier: Modifier = Modifier
)
```

**Appearance:** Full-width `neuRaised` bar at bottom. 5 items. Selected item has `neuPressed` background + `AccentPrimary` tint.

**Items:**
| Icon | Label | Route |
|:---|:---|:---|
| Home | Home | `home` |
| Link | Connect | `connect` |
| ImageSearch | Analyze | `analyze` |
| ModelTraining | Train | `train` |
| AutoAwesome | Organize | `organize` |

#### 8.10 NeuTopBar

```kotlin
@Composable
fun NeuTopBar(
    title: String,
    modifier: Modifier = Modifier,
    navigationIcon: @Composable (() -> Unit)? = null,
    actions: @Composable (RowScope.() -> Unit)? = null,
    statusDot: ConnectionStatus? = null
)
```

#### 8.11 NeuSnackbar

```kotlin
@Composable
fun NeuSnackbar(
    message: String,
    actionLabel: String? = null,   // "Undo"
    onAction: (() -> Unit)? = null,
    duration: SnackbarDuration = SnackbarDuration.Short
)
```

**Appearance:** `neuRaised`, warm background, `AccentPrimary` action text. Slides up from bottom.

#### 8.12 NeuDialog

```kotlin
@Composable
fun NeuDialog(
    title: String,
    message: String,
    confirmText: String = "Confirm",
    dismissText: String = "Cancel",
    onConfirm: () -> Unit,
    onDismiss: () -> Unit
)
```

**Overlay:** warm tint scrim `rgba(61, 43, 31, 0.3)`.
**Dialog:** `neuRaised` with strong shadows, scale-in animation.

---

### 9. Screen Specifications

#### 9.1 Home Screen

```
Route: "home"
ViewModel: HomeViewModel
```

**Layout:**
```
+------------------------------------------+
|  [NeuTopBar]                             |
|  Federated Gallery Tags    [StatusDot]   |
+------------------------------------------+
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] Server Status             | |
|  | Connected to 192.168.1.5:8080      | |
|  | Model v3  |  Round 3/10  |  3 peers| |
|  +------------------------------------+ |
|                                          |
|  +----------------+  +----------------+ |
|  | [NeuCard]      |  | [NeuCard]      | |
|  | Analyze        |  | Train          | |
|  | 342 images     |  | Round 3/10     | |
|  | scanned        |  | 78% accuracy   | |
|  +----------------+  +----------------+ |
|                                          |
|  +----------------+  +----------------+ |
|  | [NeuCard]      |  | [NeuCard]      | |
|  | Organize       |  | Privacy        | |
|  | 5 albums ready |  | Score: High    | |
|  +----------------+  +----------------+ |
|                                          |
+------------------------------------------+
|  [NeuBottomBar]                          |
+------------------------------------------+
```

**Cards are tappable** — navigate to the corresponding screen.
**Status card** shows real-time data from WebSocket.

---

#### 9.2 Connect Screen

```
Route: "connect"
ViewModel: ConnectViewModel
```

**Layout:**
```
+------------------------------------------+
|  [NeuTopBar] Connect to Server           |
+------------------------------------------+
|                                          |
|  +------------------------------------+ |
|  | [NeuTextField] Server IP            | |
|  | 192.168.1.__                       | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuTextField] Port                 | |
|  | 8080                               | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuTextField] Your Nickname        | |
|  | (optional)                         | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuButton] Test Connection         | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [ConnectionStatus]                  | |
|  | [AnimatedContent based on state]    | |
|  |   Idle -> "Enter server details"   | |
|  |   Testing -> Pulsing dots anim     | |
|  |   Success -> checkmark + model v#  | |
|  |   Failed -> error msg + retry      | |
|  +------------------------------------+ |
|                                          |
+------------------------------------------+
```

**States managed by `ConnectUiState`:**
```kotlin
sealed class ConnectUiState {
    object Idle : ConnectUiState()
    object Testing : ConnectUiState()
    data class Connected(val modelVersion: Int, val clientCount: Int) : ConnectUiState()
    data class Failed(val error: String) : ConnectUiState()
}
```

---

#### 9.3 Gallery Analysis Screen

```
Route: "analyze"
ViewModel: AnalyzeViewModel
```

**Two phases:**

**Phase A — Folder Selection:**
```
+------------------------------------------+
|  [NeuTopBar] Analyze Gallery             |
+------------------------------------------+
|                                          |
|  Select folders to analyze:              |
|                                          |
|  +------------------------------------+ |
|  | [x] Camera (1,245 images)           | |
|  | [ ] Downloads (89 images)           | |
|  | [x] Screenshots (312 images)        | |
|  | [ ] WhatsApp Images (567 images)    | |
|  +------------------------------------+ |
|                                          |
|  Max images: [NeuTextField: 500]         |
|                                          |
|  [NeuButton] Scan and Tag Preview        |
+------------------------------------------+
```

**Phase B — Results (after scan):**
```
+------------------------------------------+
|  [NeuTopBar] Analysis Results            |
+------------------------------------------+
|                                          |
|  +------------------------------------+ |
|  | [Vico Pie/Donut Chart]              | |
|  | Tag Distribution                    | |
|  | People 30% | Places 25% | ...      | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] Tag Breakdown             | |
|  |                                     | |
|  | People     ========== 152    30%    | |
|  | Places     =======    128    25%    | |
|  | Activities =====       89    18%    | |
|  | Nature     ====        65    13%    | |
|  | Objects    ===         40     8%    | |
|  | Documents  ==          18     4%    | |
|  | Events     =           10     2%    | |
|  +------------------------------------+ |
|                                          |
|  [NeuButton] Review Labels               |
|  (opens swipe review UI)                 |
|                                          |
+------------------------------------------+
```

**Scan progress:** `NeuProgressBar` with image count (Scanning image 142 / 500), animated.

**Swipe Review UI (Bottom Sheet):**
- Full-screen image with predicted tags shown as `NeuChip`s
- Swipe right = confirm labels
- Swipe left = reject (remove from training)
- Tap a chip to remove that specific tag
- Tap "+" to add a tag from taxonomy picker
- Swipe up = skip (keep pseudo-labels as-is)

---

#### 9.4 Federated Training Screen

```
Route: "train"
ViewModel: TrainingViewModel
```

**Layout:**
```
+------------------------------------------+
|  [NeuTopBar] Federated Training          |
+------------------------------------------+
|                                          |
|  +------------------------------------+ |
|  | [NeuProgressBar]                    | |
|  | Round 3 / 10                        | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] Training Console          | |
|  | (scrollable monospace log)          | |
|  |                                     | |
|  | [12:34:01] Round 3 started          | |
|  | [12:34:02] Training on 342 images   | |
|  | [12:34:45] Local loss: 0.423        | |
|  | [12:34:46] Applying DP noise (e=1)  | |
|  | [12:34:46] Submitting update...     | |
|  | [12:35:02] Round 3 complete         | |
|  | [12:35:02] Global accuracy: 0.782   | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [Vico Line Chart]                   | |
|  | Accuracy over Rounds                | |
|  | (terracotta line, gold baseline)    | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] Your Contribution         | |
|  | Images: 342  Rounds: 3/10          | |
|  | Accuracy improvement: +12.3%        | |
|  +------------------------------------+ |
|                                          |
|  +----+ +------+ +------------------+   |
|  |Join| |Pause | |Leave Session     |   |
|  +----+ +------+ +------------------+   |
|                                          |
+------------------------------------------+
```

**Training console:** `LazyColumn` with monospace text items (`JetBrainsMono`). Auto-scrolls to bottom. Max 200 entries (older removed). Items fade in via `AnimatedVisibility`.

**Chart:** Vico `CartesianChart` with warm line colors, smooth bezier interpolation, animated data point addition.

---

#### 9.5 Organize Screen

```
Route: "organize"
ViewModel: OrganizeViewModel
```

**Two phases:**

**Phase A — Preview:**
```
+------------------------------------------+
|  [NeuTopBar] Organize Gallery            |
+------------------------------------------+
|                                          |
|  Proposed Operations (23 images):        |
|                                          |
|  +------------------------------------+ |
|  | [Thumbnail] beach_sunset.jpg        | |
|  | Tags: Places > Beach (92%)          | |
|  | Album: FGT/Places/Beach             | |
|  | [x] Include    [Heatmap icon]       | |
|  +------------------------------------+ |
|  | [Thumbnail] group_dinner.jpg        | |
|  | Tags: People > Group (87%),         | |
|  |       Objects > Food (74%)          | |
|  | Album: FGT/People/Group             | |
|  | [x] Include    [Heatmap icon]       | |
|  +------------------------------------+ |
|  | ... (LazyColumn)                    | |
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] Summary                   | |
|  | 7 new albums | 23 images            | |
|  +------------------------------------+ |
|                                          |
|  [NeuButton] Apply Changes              |
|                                          |
+------------------------------------------+
```

**Heatmap icon tap:** Opens overlay showing GradCAM attention heatmap for that image's primary tag. Rendered as a semi-transparent warm colormap overlay on the full image in a dialog.

**Phase B — Progress:**
- `NeuProgressBar` with "Copying image 8 / 23"
- Success: `NeuSnackbar` with "Created 7 albums with 23 images" + "Undo" action

**Undo flow:**
1. User taps "Undo" in Snackbar (within 10s timeout)
2. `RevertOperationsUseCase` deletes all copied files, restores EXIF
3. Show confirmation "Reverted 23 operations"

**Post-apply:** Show `NeuCard` with "Revert History" listing past sessions, each with a "Revert" button.

---

#### 9.6 Privacy Dashboard Screen

```
Route: "privacy"
ViewModel: PrivacyViewModel
```

**Layout:**
```
+------------------------------------------+
|  [NeuTopBar] Privacy Dashboard           |
+------------------------------------------+
|                                          |
|  +------------------------------------+ |
|  | [Data Flow Diagram]                 | |
|  |                                     | |
|  | YOUR DEVICE        FGT SERVER       | |
|  | +----------+       +----------+     | |
|  | | Images   |       | Global   |     | |
|  | | Tags     | ----> | Model    |     | |
|  | | Personal | Weights| Metrics  |     | |
|  | | Data     | Only   |          |     | |
|  | +----------+ <---- +----------+     | |
|  |    STAYS     Updated    AGGREGATES  | |
|  |    HERE      Model      ONLY        | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] Privacy Score             | |
|  |                                     | |
|  | [NeuProgressBar] =========== High   | |
|  |                                     | |
|  | DP Epsilon: 1.0 (strong privacy)    | |
|  | Gradient Clipping: 1.0 L2 norm      | |
|  | Noise added: 247 parameters         | |
|  +------------------------------------+ |
|                                          |
|  +------------------------------------+ |
|  | [NeuCard] What Was Transmitted      | |
|  |                                     | |
|  | Weights sent:    3 rounds x 262KB   | |
|  | Metrics sent:    loss, accuracy     | |
|  | Images sent:     0 (zero)           | |
|  | Personal data:   0 (zero)           | |
|  +------------------------------------+ |
|                                          |
+------------------------------------------+
```

**Data flow diagram:** Custom `Canvas` composable drawing the boxes and animated dashed arrows (using `PathEffect.dashPathEffect` with animated phase offset).

---

#### 9.7 Settings Screen

```
Route: "settings" (accessible from top bar gear icon)
ViewModel: SettingsViewModel
```

**Sections:**

| Section | Controls |
|:---|:---|
| **Training** | Local epochs (slider 1-10), Learning rate (text field), Batch size (auto/manual) |
| **Privacy** | DP enabled (toggle), Epsilon (slider 0.1-10), Gradient clipping norm (slider 0.1-5) |
| **Taxonomy** | View/edit tag categories (expandable tree), enable/disable categories |
| **Gallery** | Max images per session (slider 100-2000), Include screenshots (toggle) |
| **About** | Version, licenses, server URL display |

---

### 10. Navigation

```kotlin
// ui/navigation/FGTNavGraph.kt

// Type-safe routes
@Serializable object HomeRoute
@Serializable object ConnectRoute
@Serializable object AnalyzeRoute
@Serializable object TrainRoute
@Serializable object OrganizeRoute
@Serializable object PrivacyRoute
@Serializable object SettingsRoute

@Composable
fun FGTNavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = HomeRoute,
        enterTransition = { fadeIn(tween(300)) + slideInHorizontally { it / 4 } },
        exitTransition = { fadeOut(tween(200)) },
        popEnterTransition = { fadeIn(tween(300)) + slideInHorizontally { -it / 4 } },
        popExitTransition = { fadeOut(tween(200)) }
    ) {
        composable<HomeRoute> { HomeScreen(navController) }
        composable<ConnectRoute> { ConnectScreen(navController) }
        composable<AnalyzeRoute> { AnalyzeScreen(navController) }
        composable<TrainRoute> { TrainingScreen(navController) }
        composable<OrganizeRoute> { OrganizeScreen(navController) }
        composable<PrivacyRoute> { PrivacyDashboardScreen(navController) }
        composable<SettingsRoute> { SettingsScreen(navController) }
    }
}
```

**Bottom bar:** Visible on Home, Analyze, Train, Organize. Hidden on Connect, Privacy, Settings (these use back navigation).

---

### 11. Micro-Animations

| Element | Trigger | Animation | Spec |
|:---|:---|:---|:---|
| NeuButton | Press | Shadow raised -> pressed, scale 0.98 | `tween(200ms)` |
| NeuButton | Release | Shadow pressed -> raised, scale 1.0 | `spring(dampingRatio=0.6)` |
| NeuCard | Appear | `fadeIn` + `slideInVertically(20dp)` | `tween(300ms)` |
| StatusDot | Online | Scale pulse 1 -> 1.4, alpha 1 -> 0 | `infiniteRepeatable(2000ms)` |
| StatusDot | Training | Same as online, faster | `infiniteRepeatable(1000ms)` |
| ProgressBar Fill | Value change | Width animation | `animateFloatAsState(spring)` |
| Toggle Thumb | State change | `translateX` 0 -> 24dp | `spring(stiffness=Medium)` |
| Console Log Item | Appear | `fadeIn` + `slideInVertically(8dp)` | `tween(150ms)` |
| Chart Data | New point | Vico default bezier interpolation | 500ms |
| Scan Progress | Count update | Number rolls (old -> new) | `animateIntAsState` |
| Heatmap Overlay | Open | Dialog scale 0.9 -> 1.0 + fadeIn | `tween(200ms)` |
| Swipe Review | Swipe | Card translates + rotates + fades | `Animatable` with decay |
| Snackbar | Show | `slideInVertically` from bottom | `tween(250ms)` |
| Bottom Bar Item | Select | Background `neuPressed` crossfade | `tween(200ms)` |

---

### 12. Accessibility

| Concern | Implementation |
|:---|:---|
| **Content Descriptions** | All icons have `contentDescription`. Status dots have descriptive text. Charts have `semantics { contentDescription = "..." }`. |
| **TalkBack** | Screens announce their title. Operation previews read image name + proposed tags. Training console reads latest log entry. |
| **Contrast** | Text on `BgBase` meets WCAG AA. `TextPrimary` (#3D2B1F) on `BgBase` (#F0E4D7) = ~8.2:1 ratio. |
| **Touch Targets** | All interactive elements minimum 48x48dp tap target. |
| **Reduced Motion** | Check `LocalReducedMotion.current`. Disable pulse animations, set all durations to 0ms. |
| **Font Scaling** | Use `sp` for all text. Test at 200% font scale. Layouts use `weight` and `fillMaxWidth` to accommodate. |

---

### 13. ViewModel State Pattern

Every screen follows the same state pattern:

```kotlin
// Example: TrainingViewModel
@HiltViewModel
class TrainingViewModel @Inject constructor(
    private val joinTrainingUseCase: JoinTrainingUseCase,
    private val wsClient: FGTWebSocketClient
) : ViewModel() {

    private val _uiState = MutableStateFlow(TrainingUiState())
    val uiState: StateFlow<TrainingUiState> = _uiState.asStateFlow()

    data class TrainingUiState(
        val phase: TrainingPhase = TrainingPhase.Idle,
        val currentRound: Int = 0,
        val totalRounds: Int = 0,
        val progress: Float = 0f,
        val consoleLog: List<LogEntry> = emptyList(),
        val accuracyHistory: List<Float> = emptyList(),
        val contribution: ContributionSummary? = null,
        val error: String? = null
    )

    sealed class TrainingPhase {
        object Idle : TrainingPhase()
        object Joining : TrainingPhase()
        object Training : TrainingPhase()
        object Submitting : TrainingPhase()
        object WaitingForRound : TrainingPhase()
        object Complete : TrainingPhase()
    }

    fun joinTraining() { /* ... */ }
    fun pauseTraining() { /* ... */ }
    fun leaveSession() { /* ... */ }
}
```

Screens observe `uiState` via `collectAsStateWithLifecycle()` and render declaratively.
