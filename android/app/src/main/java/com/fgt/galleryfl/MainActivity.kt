package com.fgt.galleryfl

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.Image
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import com.fgt.galleryfl.data.local.*
import com.fgt.galleryfl.data.ml.*
import com.fgt.galleryfl.data.network.*
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import com.fgt.galleryfl.ui.components.*
import com.fgt.galleryfl.ui.screens.*
import com.fgt.galleryfl.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.util.UUID
import java.util.concurrent.TimeUnit

private data class ConnectionDetails(val serverUrl: String, val token: String)

private fun parseConnectionDetails(input: String, fallbackUrl: String): ConnectionDetails {
    val trimmed = input.trim()
    // Support new simple UX: pure 8-char token (server URL from prefs/default)
    // or legacy "IP:PORT@TOKEN" for compatibility
    val atIdx = trimmed.indexOf('@')
    return if (atIdx > 0) {
        val endpoint = trimmed.substring(0, atIdx).trim()
        val token = trimmed.substring(atIdx + 1).trim()
        val url = when {
            endpoint.isBlank() -> fallbackUrl
            endpoint.startsWith("http://") || endpoint.startsWith("https://") -> endpoint.removeSuffix("/")
            else -> "http://${endpoint.removeSuffix("/")}"
        }
        ConnectionDetails(url, token)
    } else {
        // pure token case
        val token = trimmed.uppercase()
        ConnectionDetails(fallbackUrl, token)
    }
}

class MainActivity : ComponentActivity() {
    private val httpClient = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
    private val wsClient = FGTWebSocketClient(httpClient)
    private val defaultServerUrl = "http://10.0.2.2:8000"
    private lateinit var clientId: String
    private lateinit var featureExtractor: FeatureExtractor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        val prefs = getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE)
        clientId = prefs.getString("client_id", null) ?: UUID.randomUUID().toString().also {
            prefs.edit().putString("client_id", it).apply()
        }
        
        featureExtractor = FeatureExtractor(this)
        setContent {
            GalleryFLTheme {
                MainScreen(wsClient, httpClient, defaultServerUrl, clientId, featureExtractor)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        wsClient.disconnect()
        featureExtractor.close()
    }
}

@Composable
fun MainScreen(
    wsClient: FGTWebSocketClient,
    httpClient: OkHttpClient,
    defaultServerUrl: String,
    clientId: String,
    featureExtractor: FeatureExtractor
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // === PERSISTED STATE (hoisted early so remember initializers can use them) ===
    // This is the core "internal memory" for FL connection + model + outputs.
    val prefs = context.getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE)
    val savedServerUrl = prefs.getString("last_server_url", defaultServerUrl) ?: defaultServerUrl
    val savedToken = prefs.getString("last_access_token", "") ?: ""
    val savedRegisteredId = prefs.getString("last_registered_client_id", null)

    // Backward compat
    val savedCode = remember {
        if (savedToken.isNotBlank()) savedToken else (prefs.getString("last_access_code", "") ?: "")
    }

    var selectedTab by remember { mutableIntStateOf(0) }
    var showFLDialog by remember { mutableStateOf(false) }
    var selectedImage by remember { mutableStateOf<GalleryImage?>(null) }
    var searchQuery by remember { mutableStateOf("") }

    // FL State - initialized from persisted internal memory so connection + model survive simple exit
    var currentServerUrl by remember { mutableStateOf(savedServerUrl) }
    var accessCode by remember { mutableStateOf(savedToken.ifBlank { "fgt-pass" }) }

    // === CRITICAL INTERNAL MEMORY: Load from disk at composition time ===
    // This is the fix for "model and output lost when simply exiting the app".
    // We eagerly restore the model (activeWeights) and will restore albums as soon
    // as photos are available. These values come from disk, not just remember{}.
    val initialWeights = remember {
        try {
            val w = ModelStateStore.loadWeights(context)
            if (w != null && w.size == 4) {
                android.util.Log.d("FGT_Persist", "RESTORE initialWeights (composition): layers=${w.size}")
                w
            } else null
        } catch (e: Exception) {
            android.util.Log.d("FGT_Persist", "RESTORE initialWeights failed: ${e.message}")
            null
        }
    }
    val initialVersion = remember {
        try { ModelStateStore.loadModelVersion(context) } catch (_: Exception) { 0 }
    }

    var activeWeights by remember { mutableStateOf<List<FloatArray>?>(initialWeights) }
    var currentModelVersion by remember { mutableIntStateOf(initialVersion) }

    var statusText by remember { mutableStateOf("Ready to sync") }
    var isConnected by remember { mutableStateOf(false) }
    var isTraining by remember { mutableStateOf(false) }
    var currentRound by remember { mutableIntStateOf(0) }
    var registeredClientId by remember { mutableStateOf<String?>(savedRegisteredId) }

    // Photos State
    var photos by remember { mutableStateOf<List<GalleryImage>>(emptyList()) }

    // === CRITICAL INTERNAL MEMORY: smartAlbums (Scan & Group "output") ===
    // The reported problem: once Scan produces output (smartAlbums) or the model loads,
    // everything disappears on normal app exit.
    //
    // Strategy (multiple redundant restores from persistent storage):
    // - Model: loaded at composition time via remember { ModelStateStore... }
    // - Albums (the "output"): restored unconditionally from Room (ScanResultDao)
    //   as soon as photos are available (permission block + LaunchedEffects).
    // - The DB + ModelStateStore are the real "internal memory".
    var smartAlbums by remember { mutableStateOf<List<GalleryAlbum>>(emptyList()) }
    var isScanning by remember { mutableStateOf(false) }

    // Organize / Undo state
    var isOrganizing by remember { mutableStateOf(false) }
    var canUndoOrganize by remember { mutableStateOf(false) }

    val filteredPhotos by remember {
        derivedStateOf {
            if (searchQuery.isBlank()) {
                photos
            } else {
                val matchingAlbumTags = smartAlbums
                    .filter { it.tagName.contains(searchQuery, ignoreCase = true) }
                    .flatMap { it.images.map { img -> img.id } }
                    .toSet()
                photos.filter { it.id in matchingAlbumTags || it.displayName.contains(searchQuery, ignoreCase = true) }
            }
        }
    }

    var hasPermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
            } else {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
            }
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        hasPermission = permissions.values.all { it }
    }

    val deleteLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            // Successfully deleted, refresh list
            scope.launch {
                photos = GalleryRepository(context).fetchRecentImages(limit = 500)
            }
        }
    }

    // === INITIAL LOAD + AGGRESSIVE INTERNAL MEMORY RESTORE ===
    // This block is the main defense against "everything lost on simple app exit".
    // We load photos, then immediately hydrate BOTH the model and the Scan output (smartAlbums)
    // from their persistent stores (ModelStateStore + Room).
    LaunchedEffect(hasPermission) {
        if (hasPermission) {
            val all = GalleryRepository(context).fetchRecentImages(limit = 500)
            val trashed = try {
                withContext(Dispatchers.IO) { MediaStateStore.getTrashed(context).toSet() }
            } catch (_: Exception) { emptySet() }
            val loadedPhotos = if (trashed.isEmpty()) all else all.filter { it.uri.toString() !in trashed }
            photos = loadedPhotos

            // 1. Always attempt to restore the model (weights + version) from disk.
            // This is critical "internal memory" — the loaded classification head must
            // survive a normal app exit.
            try {
                val w = ModelStateStore.loadWeights(context)
                if (w != null && w.size == 4) {
                    activeWeights = w
                    currentModelVersion = ModelStateStore.loadModelVersion(context)
                    android.util.Log.d("FGT_Persist", "RESTORE model from disk (permission): v=$currentModelVersion, layers=${w.size}")
                }
            } catch (e: Exception) {
                android.util.Log.d("FGT_Persist", "RESTORE model failed: ${e.message}")
            }

            // 2. Force-restore Scan & Group "output" (smartAlbums) from Room.
            // This is the PRIMARY fix for the bug:
            // "once the app gives an output, it is all lost when the app is simply exited".
            //
            // We ALWAYS restore from the persistent DB (Room / ScanResultDao)
            // as soon as photos are available. We assign unconditionally (even if
            // the list is empty). The DB is the single source of truth for albums
            // and survives normal app exit / process death.
            if (loadedPhotos.isNotEmpty()) {
                try {
                    val restoredAlbums = restoreScanResults(context, loadedPhotos)
                    smartAlbums = restoredAlbums   // persisted truth (unconditional)
                    android.util.Log.d("FGT_Persist", "RESTORE smartAlbums (permission): ${restoredAlbums.size} albums")
                } catch (e: Exception) {
                    android.util.Log.d("FGT_Persist", "RESTORE smartAlbums failed: ${e.message}")
                }
            }
        }
    }

    // Restore "can undo" state from the operation log on launch.
    LaunchedEffect(Unit) {
        try {
            canUndoOrganize = AppDatabase.getDatabase(context).operationLogDao().getLatestLog() != null
        } catch (_: Exception) { /* DB not ready yet */ }
    }

    // === DEFENSIVE RE-HYDRATION (internal memory for model + output) ===
    // These ensure that even after process death or simple exit, the model
    // and Scan "output" (smartAlbums) are restored from persistent storage.

    LaunchedEffect(isConnected) {
        if ((activeWeights == null || (activeWeights?.size ?: 0) != 4) && isConnected) {
            val (w, v) = restoreModelFromDisk(context)
            if (w != null && w.size == 4) {
                activeWeights = w
                currentModelVersion = v
                android.util.Log.d("FGT_Persist", "RESTORE model (Launched isConnected): v=$v")
            }
        }
    }

    // Unconditional restore of Scan "output" (smartAlbums) — the main thing that was lost on simple exit.
    // Whenever we have photos, we force-restore from Room (the single source of truth).
    // We assign the result directly (even if the list is empty) so any stale in-memory
    // albums are replaced by the real persisted output.
    LaunchedEffect(photos) {
        if (photos.isNotEmpty()) {
            val restored = restoreSmartAlbumsFromDisk(context, photos)
            smartAlbums = restored   // always take persisted truth (unconditional)
            android.util.Log.d("FGT_Persist", "RESTORE smartAlbums (Launched photos): ${restored.size} albums")
        }
    }

    // Combined safety net for model + output.
    // For albums (the "output"), we always restore from Room — do not guard on isEmpty,
    // because we want to replace any stale in-memory value with the persisted truth.
    LaunchedEffect(activeWeights, photos.size) {
        if (photos.isNotEmpty()) {
            val restored = restoreSmartAlbumsFromDisk(context, photos)
            smartAlbums = restored   // always prefer persisted data
            android.util.Log.d("FGT_Persist", "RESTORE smartAlbums (Launched weights+size): ${restored.size} albums")
        }
        if (activeWeights == null || (activeWeights?.size ?: 0) != 4) {
            val (w, v) = restoreModelFromDisk(context)
            if (w != null && w.size == 4) {
                activeWeights = w
                currentModelVersion = v
                android.util.Log.d("FGT_Persist", "RESTORE model (Launched weights+size): v=$v")
            }
        }
    }

    // Auto-resume connection/session on launch (phone off / restart).
    // Persists server URL + token + clientId. Re-registers if needed to avoid 409.
    LaunchedEffect(Unit) {
        val p = context.getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE)
        val persistedUrl = p.getString("last_server_url", null)
        val persistedToken = p.getString("last_access_token", null)
        val persistedRegId = p.getString("last_registered_client_id", null)
        if (!persistedUrl.isNullOrBlank() && !persistedToken.isNullOrBlank()) {
            currentServerUrl = persistedUrl
            accessCode = persistedToken
            registeredClientId = persistedRegId
            scope.launch(Dispatchers.IO) {
                try {
                    val api = RetrofitClient.getApiService(persistedUrl, httpClient)
                    // Always re-register on resume to ensure server knows us (prevents 409)
                    val reg = api.register(persistedToken, RegisterRequest(Build.MODEL, "User-${clientId.take(4)}", persistedRegId ?: clientId))
                    TagSignalSender.configure(api, persistedToken, reg.client_id)
                    registeredClientId = reg.client_id
                    p.edit()
                        .putString("last_registered_client_id", reg.client_id)
                        .putString("last_access_token", persistedToken)
                        .putString("last_server_url", persistedUrl)
                        .apply()

                    val resp = api.getCurrentModel(0)
                    val sv = resp.headers()["X-Model-Version"]?.toIntOrNull() ?: reg.model_version
                    val w = WeightSerializer.deserialize(resp.body()?.string() ?: "")
                    activeWeights = w
                    currentModelVersion = sv
                    ModelStateStore.saveWeights(context, w)
                    ModelStateStore.saveModelVersion(context, sv)

                    withContext(Dispatchers.Main) {
                        wsClient.connect(persistedUrl, reg.client_id, persistedToken)
                        isConnected = true
                        statusText = "Resumed connection"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText = "Resume failed: ${e.message?.take(60)}"
                        isConnected = false
                    }
                }
            }
        }
    }

    // FL WebSocket Listeners
    DisposableEffect(Unit) {
        wsClient.onConnectionChanged = { connected, reason ->
            scope.launch {
                isConnected = connected
                if (connected) {
                    statusText = "Connected to coordinator"
                } else if (reason != null) {
                    statusText = "Reconnecting: $reason"
                }
            }
        }
        wsClient.onUpdateRequested = { round, roundConfig ->
            currentRound = round
            isTraining = true
            statusText = "Preparing private update (Round $round)..."
            scope.launch(Dispatchers.IO) {
                try {
                    val apiService = RetrofitClient.getApiService(currentServerUrl, httpClient)
                    val response = apiService.getCurrentModel(currentModelVersion)
                    val format = response.headers()["X-Model-Format"]
                    val serverVersion = response.headers()["X-Model-Version"]?.toIntOrNull() ?: 0
                    val downloadedWeights = WeightSerializer.deserialize(response.body()?.string() ?: "")
                    val globalWeights = if (format == "delta" && activeWeights != null) {
                        activeWeights!!.zip(downloadedWeights) { active, delta ->
                            require(active.size == delta.size) { "Delta tensor shape mismatch" }
                            FloatArray(active.size) { index -> active[index] + delta[index] }
                        }
                    } else {
                        downloadedWeights
                    }
                    val numClasses = globalWeights[2].size / 256
                    require(numClasses == TaxonomyConfig.NUM_CLASSES) { "Server taxonomy mismatch" }
                    activeWeights = globalWeights
                    currentModelVersion = serverVersion
                    ModelStateStore.saveWeights(context, globalWeights)
                    ModelStateStore.saveModelVersion(context, serverVersion)

                    // Idempotently refresh registration before either update or skip response.
                    val prefsNow = context.getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE)
                    val liveToken = prefsNow.getString("last_access_token", accessCode) ?: accessCode
                    val requestedClientId = registeredClientId
                        ?: prefsNow.getString("last_registered_client_id", clientId)
                        ?: clientId
                    val registration = apiService.register(
                        liveToken,
                        RegisterRequest(Build.MODEL, "User-${clientId.take(4)}", requestedClientId),
                    )
                    registeredClientId = registration.client_id
                    prefsNow.edit().putString("last_registered_client_id", registration.client_id).apply()

                    val repository = GalleryRepository(context)
                    val loadedImages = repository.fetchRecentImages(limit = 100).mapNotNull { image ->
                        repository.loadBitmap(image.uri)?.let { bitmap -> image to bitmap }
                    }
                    val feedbackStore = LocalFeedbackStore(context)
                    val resolver = ThresholdResolver(feedbackStore)
                    val biasOffsets = feedbackStore.getBiasOffsets(numClasses)
                    val localHead = ClassificationHead(numClasses).also { it.setWeightsFlat(globalWeights) }
                    val pseudoGenerator = PseudoLabelGenerator(
                        resolver,
                        localHead,
                        roundConfig.pseudoLabelThreshold,
                    )
                    val feedbackDao = AppDatabase.getDatabase(context).feedbackDao()

                    val trainFeatures = mutableListOf<FloatArray>()
                    val trainTargets = mutableListOf<FloatArray>()
                    val sampleWeights = mutableListOf<Float>()
                    var humanLabelCount = 0
                    var pseudoLabelCount = 0

                    for ((image, bitmap) in loadedImages) {
                        val features = featureExtractor.extractFeatures(bitmap).projection
                        val feedback = feedbackDao.getFeedbackForImage(image.id)
                        when {
                            feedback?.isConfirmed == true -> {
                                val target = FloatArray(numClasses)
                                target[feedback.classIndex] = 1f
                                trainFeatures.add(features)
                                trainTargets.add(target)
                                sampleWeights.add(1f)
                                humanLabelCount++
                            }
                            feedback != null -> {
                                // A rejection without a replacement is not a valid
                                // categorical target. Wait for an explicit correction.
                            }
                            else -> {
                                val pseudo = pseudoGenerator.generatePseudoLabel(features, biasOffsets)
                                if (pseudo != null) {
                                    trainFeatures.add(features)
                                    trainTargets.add(pseudo.targets)
                                    sampleWeights.add(roundConfig.pseudoLabelWeight)
                                    pseudoLabelCount++
                                }
                            }
                        }
                    }

                    val submitToken = prefsNow.getString("last_access_token", liveToken) ?: liveToken
                    if (trainFeatures.size < roundConfig.minLocalSamples) {
                        apiService.skipUpdate(
                            submitToken,
                            SkipUpdateRequest(
                                client_id = registration.client_id,
                                round = round,
                                base_model_version = currentModelVersion,
                                reason = "only_${trainFeatures.size}_usable_examples",
                            ),
                        )
                        withContext(Dispatchers.Main) {
                            statusText = "Round $round skipped: add corrections or more confident photos"
                            isTraining = false
                        }
                        return@launch
                    }

                    val trainer = LocalTrainer(ClassificationHead(numClasses), mu = roundConfig.mu)
                    val result = trainer.train(
                        featuresList = trainFeatures,
                        targetsList = trainTargets,
                        sampleWeights = sampleWeights,
                        globalWeights = globalWeights,
                        epochs = roundConfig.localEpochs,
                        lr = roundConfig.learningRate,
                        dpEpsilon = roundConfig.dpEpsilon,
                        dpDelta = roundConfig.dpDelta,
                        maxGradNorm = roundConfig.maxGradNorm,
                    )
                    apiService.submitUpdate(
                        submitToken,
                        ClientUpdateRequest(
                            client_id = registration.client_id,
                            weights = WeightSerializer.serialize(result.updatedWeights),
                            num_samples = result.numSamples,
                            human_labeled_samples = humanLabelCount,
                            pseudo_labeled_samples = pseudoLabelCount,
                            local_loss = result.localLoss,
                            local_accuracy = result.localAccuracy,
                            round = round,
                            base_model_version = currentModelVersion,
                        ),
                    )
                    withContext(Dispatchers.Main) {
                        statusText = "Private update sent for Round $round ($humanLabelCount corrected, $pseudoLabelCount high-confidence)"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText = "Sync failed: ${e.message}"
                        isTraining = false
                    }
                }
            }
        }
        wsClient.onRoundCompleted = { round ->
            isTraining = false
            statusText = "Round $round finished"
        }
        wsClient.onTrainingComplete = {
            isTraining = false
            statusText = "Model fully synced"
        }
        onDispose { wsClient.disconnect() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AppBackdrop(Modifier.fillMaxSize())
        Scaffold(
            containerColor = Color.Transparent,
            topBar = {
                if (selectedImage == null) {
                    PhotosTopBar(
                        searchQuery = searchQuery,
                        onSearchQueryChange = { searchQuery = it },
                        isTraining = isTraining,
                        onProfileClick = { showFLDialog = true }
                    )
                }
            },
            bottomBar = {
                if (selectedImage == null) {
                    PhotosBottomNav(
                        selectedTab = selectedTab,
                        onTabSelected = { selectedTab = it }
                    )
                }
            }
        ) { padding ->
            Box(modifier = Modifier.padding(padding)) {
                if (!hasPermission) {
                    PermissionScreen {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            permissionLauncher.launch(arrayOf(Manifest.permission.READ_MEDIA_IMAGES))
                        } else {
                            permissionLauncher.launch(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE))
                        }
                    }
                } else {
                    when (selectedTab) {
                        0 -> PhotosScreen(
                            photos = filteredPhotos,
                            onImageClick = { selectedImage = it }
                        )
                        1 -> ExploreScreen(
                            photos = photos,
                            smartAlbums = smartAlbums,
                            onImageClick = { selectedImage = it },
                            isScanning = isScanning,
                            onScanClick = {
                                isScanning = true
                                scope.launch(Dispatchers.IO) {
                                    try {
                                        val repo = GalleryRepository(context)
                                        val allImages = repo.fetchRecentImages(limit = 500)
                                        if (activeWeights == null) {
                                            withContext(Dispatchers.Main) {
                                                isScanning = false
                                                statusText = "Model is not synced. Sync the model to start scanning."
                                                showFLDialog = true
                                            }
                                            return@launch
                                        }
                                        // Best-effort backend check: confirm a
                                        // model is actually loaded server-side.
                                        try {
                                            val status = RetrofitClient.getApiService(currentServerUrl, httpClient).getModelStatus()
                                            if (!status.loaded || !status.head_present) {
                                                withContext(Dispatchers.Main) {
                                                    isScanning = false
                                                    statusText = "Model is not synced. Sync the model to start scanning."
                                                    showFLDialog = true
                                                }
                                                return@launch
                                            }
                                        } catch (_: Exception) { /* offline: trust local state */ }

                                        val numClasses = activeWeights!![2].size / 256
                                        val head = ClassificationHead(numClasses)
                                        head.setWeightsFlat(activeWeights!!)
                                        val feedbackStore = LocalFeedbackStore(context)
                                        val thresholds = ThresholdResolver(feedbackStore).getThresholdsForAllClasses(numClasses)
                                        val biasOffsets = feedbackStore.getBiasOffsets(numClasses)

                                        val albumMap = mutableMapOf<Int, MutableList<Pair<GalleryImage, Float>>>()
                                        val confidenceMap = mutableMapOf<Int, Float>()

                                        for (img in allImages) {
                                            val bitmap = repo.loadBitmap(img.uri) ?: continue
                                            val features = featureExtractor.extractFeatures(bitmap).projection
                                            val preds = head.forward(features, biasOffsets)

                                            var bestClass = -1
                                            var bestMargin = -1f

                                            for (c in 0 until numClasses) {
                                                val margin = preds[c] - thresholds[c]
                                                if (margin >= 0 && margin > bestMargin) {
                                                    bestMargin = margin
                                                    bestClass = c
                                                }
                                            }

                                            if (bestClass != -1) {
                                                albumMap.getOrPut(bestClass) { mutableListOf() }.add(Pair(img, preds[bestClass]))
                                                confidenceMap[bestClass] = (confidenceMap[bestClass] ?: 0f) + preds[bestClass]
                                            }
                                        }

                                        val entitiesToSave = mutableListOf<ScanResultEntity>()
                                        val finalAlbums = albumMap.map { (c, imagePairs) ->
                                            val policy = TaxonomyConfig.getPolicyForClassIndex(c)!!
                                            val sorted = imagePairs.sortedByDescending { it.second }
                                            val sortedImages = sorted.map { it.first }
                                            val avgConf = confidenceMap[c]!! / sortedImages.size
                                            val folderPath = "${policy.category.name}/${policy.tag.name}"
                                            sorted.forEachIndexed { pos, (img, conf) ->
                                                entitiesToSave.add(
                                                    ScanResultEntity(
                                                        classIndex = c,
                                                        tagName = policy.tag.name,
                                                        folderPath = folderPath,
                                                        imageUri = img.uri.toString(),
                                                        confidence = conf,
                                                        positionInAlbum = pos,
                                                        avgConfidence = avgConf,
                                                        thresholdUsed = thresholds[c]
                                                    )
                                                )
                                            }
                                            GalleryAlbum(
                                                folderPath = folderPath,
                                                tagName = policy.tag.name,
                                                images = sortedImages,
                                                averageConfidence = avgConf,
                                                thresholdUsed = thresholds[c],
                                                localPersonalizationAffected = false
                                            )
                                        }

                                        // Persist the scan so results survive app exit.
                                        try {
                                            val dao = AppDatabase.getDatabase(context).scanResultDao()
                                            dao.clear()
                                            if (entitiesToSave.isNotEmpty()) dao.insertAll(entitiesToSave)
                                        } catch (_: Exception) { /* best-effort */ }

                                        withContext(Dispatchers.Main) {
                                            smartAlbums = finalAlbums
                                            isScanning = false
                                            statusText = "Scan complete: ${finalAlbums.size} groups found"
                                        }
                                    } catch (e: Exception) {
                                        android.util.Log.e("ScanLogic", "Scanning failed", e)
                                        withContext(Dispatchers.Main) { 
                                            isScanning = false 
                                            statusText = "Scan failed: ${e.localizedMessage}"
                                        }
                                    }
                                }
                            },
                            onAlbumClick = { album ->
                                searchQuery = album.tagName
                                selectedTab = 0
                            },
                            onOrganizeClick = {
                                scope.launch(Dispatchers.IO) {
                                    try {
                                        if (activeWeights == null) {
                                            withContext(Dispatchers.Main) {
                                                statusText = "Model is not synced. Sync the model first."
                                                showFLDialog = true
                                            }
                                            return@launch
                                        }
                                        isOrganizing = true
                                        val feedbackStore = LocalFeedbackStore(context)
                                        val executor = OrganizeExecutor(
                                            context, featureExtractor, feedbackStore,
                                            AlbumCreator(context), ExifTagWriter(context),
                                            AppDatabase.getDatabase(context)
                                        )
                                        val outcome = executor.organize(activeWeights!!)
                                        withContext(Dispatchers.Main) {
                                            canUndoOrganize = true
                                            isOrganizing = false
                                            statusText = "Organized ${outcome.folders.size} albums, " +
                                                    "${outcome.totalCopied} photos into Pictures/FGT/"
                                        }
                                    } catch (e: Exception) {
                                        withContext(Dispatchers.Main) {
                                            isOrganizing = false
                                            statusText = "Organize failed: ${e.message}"
                                        }
                                    }
                                }
                            },
                            onUndoClick = {
                                scope.launch(Dispatchers.IO) {
                                    try {
                                        val feedbackStore = LocalFeedbackStore(context)
                                        val executor = OrganizeExecutor(
                                            context, featureExtractor, feedbackStore,
                                            AlbumCreator(context), ExifTagWriter(context),
                                            AppDatabase.getDatabase(context)
                                        )
                                        val done = executor.undoLast()
                                        withContext(Dispatchers.Main) {
                                            canUndoOrganize = executor.hasUndoable()
                                            statusText = if (done) "Undid last organization" else "Nothing to undo"
                                        }
                                    } catch (e: Exception) {
                                        withContext(Dispatchers.Main) { statusText = "Undo failed: ${e.message}" }
                                    }
                                }
                            },
                            canUndo = canUndoOrganize,
                            modelReady = activeWeights != null
                        )
                        2 -> SearchScreen(
                            photos = filteredPhotos,
                            smartAlbums = smartAlbums,
                            onImageClick = { selectedImage = it }
                        )
                    }
                }
            }
        }

        selectedImage?.let { image ->
        ImageDetailScreen(
            image = image,
            onBack = { selectedImage = null },
            featureExtractor = featureExtractor,
            activeWeights = activeWeights,
            onDelete = { img ->
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    val pendingIntent = android.provider.MediaStore.createDeleteRequest(
                        context.contentResolver,
                        listOf(img.uri)
                    )
                    deleteLauncher.launch(
                        androidx.activity.result.IntentSenderRequest.Builder(pendingIntent.intentSender).build()
                    )
                } else {
                    // Legacy delete
                    context.contentResolver.delete(img.uri, null, null)
                    photos = photos.filter { it.id != img.id }
                }
                selectedImage = null
            }
        )
    }
    }

    if (showFLDialog) {
        FLSyncDialog(
            statusText = statusText,
            isConnected = isConnected,
            isTraining = isTraining,
            initialCode = savedCode,
            onDismiss = { showFLDialog = false },
            onConnect = { code ->
                scope.launch(Dispatchers.IO) {
                    try {
                        val connection = parseConnectionDetails(code, defaultServerUrl)
                        val tokenOnly = if (code.contains("@")) code.substringAfter("@").trim().uppercase() else code.trim().uppercase()
                        val url = connection.serverUrl

                        val prefs = context.getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE)
                        prefs.edit()
                            .putString("last_server_url", url)
                            .putString("last_access_token", tokenOnly)
                            .apply()

                        currentServerUrl = url
                        accessCode = tokenOnly

                        val apiService = RetrofitClient.getApiService(url, httpClient)
                        val reg = apiService.register(tokenOnly, RegisterRequest(Build.MODEL, "User-${clientId.take(4)}", clientId))
                        TagSignalSender.configure(apiService, tokenOnly, reg.client_id)
                        
                        val response = apiService.getCurrentModel(0)
                        val serverVersion = response.headers()["X-Model-Version"]?.toIntOrNull() ?: reg.model_version
                        activeWeights = WeightSerializer.deserialize(response.body()?.string() ?: "")
                        currentModelVersion = serverVersion
                        ModelStateStore.saveWeights(context, activeWeights!!)
                        ModelStateStore.saveModelVersion(context, serverVersion)

                        prefs.edit()
                            .putString("last_registered_client_id", reg.client_id)
                            .apply()

                        withContext(Dispatchers.Main) {
                            wsClient.connect(url, reg.client_id, tokenOnly)
                            registeredClientId = reg.client_id
                            statusText = "Opening secure connection..."
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) { statusText = "Sync failed: ${e.message}" }
                    }
                }
            },
            onDisconnect = {
                wsClient.disconnect()
                isConnected = false
                isTraining = false
                statusText = "Ready to sync"
            }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotosTopBar(
    searchQuery: String,
    onSearchQueryChange: (String) -> Unit,
    isTraining: Boolean,
    onProfileClick: () -> Unit
) {
    TopAppBar(
        colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent),
        title = {
            OutlinedTextField(
                value = searchQuery,
                onValueChange = onSearchQueryChange,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .padding(end = 16.dp),
                placeholder = { Text("Search tags or names", style = MaterialTheme.typography.bodyMedium) },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, tint = FGTColors.TextSecondary) },
                trailingIcon = {
                    if (searchQuery.isNotEmpty()) {
                        IconButton(onClick = { onSearchQueryChange("") }) {
                            Icon(Icons.Default.Close, contentDescription = "Clear")
                        }
                    }
                },
                shape = RoundedCornerShape(24.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    unfocusedContainerColor = FGTColors.BgGlass,
                    focusedContainerColor = FGTColors.BgGlass,
                    unfocusedBorderColor = Color.Transparent,
                    focusedBorderColor = FGTColors.AccentPrimary
                )
            )
        },
        actions = {
            IconButton(onClick = onProfileClick) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = Icons.Outlined.AccountCircle,
                        contentDescription = "Profile",
                        modifier = Modifier.size(32.dp),
                        tint = FGTColors.AccentPrimary
                    )
                    if (isTraining) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(38.dp),
                            strokeWidth = 2.dp,
                            color = FGTColors.AccentPrimary
                        )
                    }
                }
            }
        }
    )
}

/**
 * Rebuild the persisted Scan & Group albums from [ScanResultEntity] rows.
 * This is the "output" that must survive simple app exit.
 */
private suspend fun restoreScanResults(
    context: Context,
    photos: List<GalleryImage>
): List<GalleryAlbum> {
    val entities = AppDatabase.getDatabase(context).scanResultDao().getAll()
    if (entities.isEmpty()) return emptyList()
    val byUri = photos.associateBy { it.uri.toString() }
    return entities
        .groupBy { it.classIndex }
        .mapNotNull { (_, rows) ->
            val images = rows.mapNotNull { byUri[it.imageUri] }
            if (images.isEmpty()) return@mapNotNull null
            val first = rows.first()
            GalleryAlbum(
                folderPath = first.folderPath,
                tagName = first.tagName,
                images = images,
                averageConfidence = first.avgConfidence,
                thresholdUsed = first.thresholdUsed,
                localPersonalizationAffected = false
            )
        }
}

/**
 * Restore the classification head (activeWeights) from disk.
 * This is the primary "internal memory" for the loaded model.
 * Called from multiple LaunchedEffects so the model is never lost on simple exit.
 */
private suspend fun restoreModelFromDisk(context: Context): Pair<List<FloatArray>?, Int> {
    return try {
        val weights = ModelStateStore.loadWeights(context)
        val version = ModelStateStore.loadModelVersion(context)
        if (weights != null && weights.size == 4) {
            Pair(weights, version)
        } else {
            Pair(null, 0)
        }
    } catch (_: Exception) {
        Pair(null, 0)
    }
}

/**
 * Restore smart albums ("output") from Room DB.
 * Unconditionally hydrates from persisted scan results whenever photos are present.
 */
private suspend fun restoreSmartAlbumsFromDisk(
    context: Context,
    photos: List<GalleryImage>
): List<GalleryAlbum> {
    return try {
        restoreScanResults(context, photos)
    } catch (_: Exception) {
        emptyList()
    }
}

@Composable
fun PhotosBottomNav(selectedTab: Int, onTabSelected: (Int) -> Unit) {
    GlassContainer(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp)
            .height(68.dp),
        cornerRadius = 34.dp,
        backgroundColor = FGTColors.BgSurface.copy(alpha = 0.72f),
        borderColor = Color.White.copy(alpha = 0.65f),
        elevation = 24.dp
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            val items = listOf(
                Icons.Default.Photo to "Photos",
                Icons.Default.Favorite to "For You",
                Icons.Default.Search to "Search"
            )
            items.forEachIndexed { index, (icon, label) ->
                val isSelected = selectedTab == index
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .clip(RoundedCornerShape(18.dp))
                        .clickable { onTabSelected(index) }
                        .background(
                            if (isSelected) Color.White.copy(alpha = 0.6f) else Color.Transparent
                        )
                        .padding(horizontal = 14.dp, vertical = 8.dp)
                ) {
                    Icon(
                        imageVector = icon,
                        contentDescription = label,
                        tint = if (isSelected) FGTColors.AccentPrimary else FGTColors.TextSecondary,
                        modifier = Modifier.size(if (isSelected) 26.dp else 22.dp)
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        label,
                        style = MaterialTheme.typography.labelSmall,
                        color = if (isSelected) FGTColors.AccentPrimary else FGTColors.TextSecondary
                    )
                }
            }
        }
    }
}

@Composable
fun FLSyncDialog(
    statusText: String,
    isConnected: Boolean,
    isTraining: Boolean,
    initialCode: String = "",
    onDismiss: () -> Unit,
    onConnect: (String) -> Unit,
    onDisconnect: () -> Unit
) {
    var codeInput by remember { mutableStateOf(initialCode) }
    var codeError by remember { mutableStateOf<String?>(null) }

    fun tryConnect() {
        val code = codeInput.trim()
        if (code.isBlank()) {
            codeError = "Enter the 8-character access token"
            return
        }
        // Accept pure 8-char token OR legacy IP@TOKEN (extract token)
        val token = if (code.contains("@")) {
            code.substringAfter("@").trim().uppercase()
        } else {
            code.uppercase()
        }
        if (!AccessCodeGenerator.isValid(token)) {
            codeError = "Access token must be 8 characters (A-Z 2-9, no 0/O/1/I/L)."
            return
        }
        codeError = null
        onConnect(code)  // pass original input (parse handles both)
    }

    Dialog(onDismissRequest = onDismiss) {
        GlassContainer(
            modifier = Modifier.fillMaxWidth(0.92f),
            cornerRadius = 28.dp,
            backgroundColor = FGTColors.BgSurface.copy(alpha = 0.92f),
            borderColor = Color.White.copy(alpha = 0.6f),
            elevation = 30.dp
        ) {
            Column(Modifier.padding(24.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Image(
                        painter = painterResource(R.drawable.ic_aperture),
                        contentDescription = null,
                        modifier = Modifier.size(30.dp),
                        colorFilter = ColorFilter.tint(FGTColors.AccentPrimary)
                    )
                    Spacer(Modifier.width(12.dp))
                    Text(
                        "Model Sync",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = FGTColors.TextPrimary
                    )
                }
                Spacer(Modifier.height(16.dp))
                Text(
                    if (isConnected) statusText else "Enter the 8-character server access token to sync the model. Your photos never leave this device.",
                    style = MaterialTheme.typography.bodyMedium, color = FGTColors.TextSecondary
                )
                if (isTraining) {
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(Modifier.fillMaxWidth(), color = FGTColors.AccentPrimary)
                }
                if (!isConnected) {
                    Spacer(Modifier.height(16.dp))
                    OutlinedTextField(
                        value = codeInput,
                        onValueChange = { codeInput = it; codeError = null },
                        label = { Text("Access Token (8 chars)") },
                        placeholder = { Text("e.g. P54RCJHM") },
                        isError = codeError != null,
                        supportingText = codeError?.let { { Text(it, color = FGTColors.Error) } }
                            ?: {
                                Text(
                                    "8-character alphanumeric token from server dashboard.",
                                    color = FGTColors.TextSecondary
                                )
                            },
                        trailingIcon = {
                            IconButton(onClick = {
                                codeInput = AccessCodeGenerator.generate()
                                codeError = null
                            }) {
                                Icon(
                                    Icons.Default.Refresh,
                                    contentDescription = "Generate access token",
                                    tint = FGTColors.AccentPrimary
                                )
                            }
                        },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
                Spacer(Modifier.height(20.dp))
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    if (!isConnected) {
                        Button(onClick = { tryConnect() }) { Text("Sync Now") }
                    } else {
                        TextButton(
                            onClick = onDisconnect,
                            colors = ButtonDefaults.textButtonColors(contentColor = FGTColors.Error)
                        ) { Text("Disconnect") }
                        Spacer(Modifier.width(8.dp))
                        TextButton(onClick = onDismiss) { Text("Done") }
                    }
                }
            }
        }
    }
}

@Composable
fun PermissionScreen(onRequest: () -> Unit) {
    EmptyState(
        title = "Allow access to your photos",
        message = "To organize your gallery automatically, FGT needs permission to see your photos. Your photos never leave this device.",
        actionLabel = "Allow Access",
        onAction = onRequest
    )
}
