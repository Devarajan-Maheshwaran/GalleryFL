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
    val parts = input.trim().split("@", limit = 2)
    val endpoint = parts.firstOrNull().orEmpty().trim()
    val token = parts.getOrNull(1)?.trim().orEmpty()
    val url = when {
        endpoint.isBlank() -> fallbackUrl
        endpoint.startsWith("http://") || endpoint.startsWith("https://") -> endpoint.removeSuffix("/")
        else -> "http://${endpoint.removeSuffix("/")}"
    }
    require(token.isNotBlank()) { "Enter the server as IP:PORT@ACCESS_CODE" }
    return ConnectionDetails(url, token)
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

    var selectedTab by remember { mutableIntStateOf(0) }
    var showFLDialog by remember { mutableStateOf(false) }
    var selectedImage by remember { mutableStateOf<GalleryImage?>(null) }
    var searchQuery by remember { mutableStateOf("") }

    // FL State
    var currentServerUrl by remember { mutableStateOf(defaultServerUrl) }
    var accessCode by remember { mutableStateOf("fgt-pass") }
    var activeWeights by remember { mutableStateOf<List<FloatArray>?>(null) }
    var statusText by remember { mutableStateOf("Ready to sync") }
    var isConnected by remember { mutableStateOf(false) }
    var isTraining by remember { mutableStateOf(false) }
    var currentRound by remember { mutableIntStateOf(0) }
    var currentModelVersion by remember { mutableIntStateOf(0) }
    var registeredClientId by remember { mutableStateOf<String?>(null) }

    // Photos State
    var photos by remember { mutableStateOf<List<GalleryImage>>(emptyList()) }
    var smartAlbums by remember { mutableStateOf<List<GalleryAlbum>>(emptyList()) }
    var isScanning by remember { mutableStateOf(false) }

    // Organize / Undo state
    var isOrganizing by remember { mutableStateOf(false) }
    var canUndoOrganize by remember { mutableStateOf(false) }

    // Remembered coordinator access code for smoother reconnect.
    val savedCode = remember {
        context.getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE).getString("last_access_code", "") ?: ""
    }

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

    // Initial Load
    LaunchedEffect(hasPermission) {
        if (hasPermission) {
            photos = GalleryRepository(context).fetchRecentImages(limit = 500)
        }
    }

    // Restore "can undo" state from the operation log on launch.
    LaunchedEffect(Unit) {
        try {
            canUndoOrganize = AppDatabase.getDatabase(context).operationLogDao().getLatestLog() != null
        } catch (_: Exception) { /* DB not ready yet */ }
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
        wsClient.onUpdateRequested = { round, lr, epochs, mu, dpEpsilon, dpDelta, maxGradNorm ->
            currentRound = round
            isTraining = true
            statusText = "Syncing knowledge (Round $round)..."
            scope.launch(Dispatchers.IO) {
                try {
                    val apiService = RetrofitClient.getApiService(currentServerUrl, httpClient)
                    val response = apiService.getCurrentModel(currentModelVersion)
                    val format = response.headers()["X-Model-Format"]
                    val serverVersion = response.headers()["X-Model-Version"]?.toIntOrNull() ?: 0
                    val encodedWeights = response.body()?.string() ?: ""
                    
                    val downloadedWeights = WeightSerializer.deserialize(encodedWeights)
                    val globalWeights = if (format == "delta" && activeWeights != null) {
                        activeWeights!!.zip(downloadedWeights) { a, b ->
                            FloatArray(a.size) { i -> a[i] + b[i] }
                        }
                    } else {
                        downloadedWeights
                    }
                    activeWeights = globalWeights
                    currentModelVersion = serverVersion

                    val numClasses = globalWeights[2].size / 256
                    val repo = GalleryRepository(context)
                    val recentImages = repo.fetchRecentImages(limit = 100)
                    
                    val validImages = recentImages.filter { repo.loadBitmap(it.uri) != null }
                    val featuresList = validImages.mapNotNull { img ->
                        repo.loadBitmap(img.uri)?.let { featureExtractor.extractFeatures(it).projection }
                    }.ifEmpty { List(5) { FloatArray(1024) { 0f } } }

                    val feedbackStore = LocalFeedbackStore(context)
                    val resolver = ThresholdResolver(feedbackStore)
                    val biasOffsets = feedbackStore.getBiasOffsets(numClasses)
                    val thresholds = resolver.getThresholdsForAllClasses(numClasses)

                    val localHead = ClassificationHead(numClasses)
                    localHead.setWeightsFlat(globalWeights)

                    val db = AppDatabase.getDatabase(context)
                    val pseudoGen = PseudoLabelGenerator(resolver, localHead)
                    
                    val targetsList = mutableListOf<FloatArray>()
                    featuresList.forEachIndexed { index, features ->
                        val labels = pseudoGen.generatePseudoLabels(features, biasOffsets)
                        if (validImages.isNotEmpty() && index < validImages.size) {
                            val imgId = validImages[index].id
                            val feedback = db.feedbackDao().getFeedbackForImage(imgId)
                            if (feedback != null) {
                                labels[feedback.classIndex] = if (feedback.isConfirmed) 1.0f else 0.0f
                            }
                        }
                        targetsList.add(labels)
                    }

                    // Per-example DP-SGD (server-calibrated epsilon/delta/C).
                    // dpEpsilon <= 0 means the server disabled DP for this session.
                    val trainer = LocalTrainer(ClassificationHead(numClasses), mu = mu)
                    val result = trainer.train(
                        featuresList, targetsList, globalWeights,
                        epochs = epochs, lr = lr,
                        dpEpsilon = dpEpsilon, dpDelta = dpDelta,
                        maxGradNorm = maxGradNorm, numSamples = featuresList.size
                    )

                    apiService.submitUpdate(accessCode, ClientUpdateRequest(
                        client_id = registeredClientId ?: error("Client is not registered"),
                        weights = WeightSerializer.serialize(result.updatedWeights),
                        num_samples = result.numSamples,
                        local_loss = result.localLoss,
                        local_accuracy = result.localAccuracy,
                        round = round,
                        base_model_version = currentModelVersion
                    ))

                    withContext(Dispatchers.Main) { statusText = "Sync complete for Round $round" }
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

    Box(modifier = Modifier.fillMaxSize().background(FGTColors.BgBase)) {
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
                            smartAlbums = smartAlbums,
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
                                                statusText = "Connect to sync model first"
                                                showFLDialog = true
                                            }
                                            return@launch
                                        }

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

                                        val finalAlbums = albumMap.map { (c, imagePairs) ->
                                            val policy = TaxonomyConfig.getPolicyForClassIndex(c)!!
                                            val sortedImages = imagePairs.sortedByDescending { it.second }.map { it.first }
                                            GalleryAlbum(
                                                folderPath = "${policy.category.name}/${policy.tag.name}",
                                                tagName = policy.tag.name,
                                                images = sortedImages,
                                                averageConfidence = confidenceMap[c]!! / sortedImages.size,
                                                thresholdUsed = thresholds[c],
                                                localPersonalizationAffected = false
                                            )
                                        }

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
                                                statusText = "Connect to sync model first"
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
                        2 -> LibraryScreen(smartAlbums)
                        3 -> SearchScreen(
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
                context.getSharedPreferences("fgt_prefs", Context.MODE_PRIVATE)
                    .edit().putString("last_access_code", code).apply()
                scope.launch(Dispatchers.IO) {
                    try {
                        val connection = parseConnectionDetails(code, defaultServerUrl)
                        currentServerUrl = connection.serverUrl
                        accessCode = connection.token

                        val apiService = RetrofitClient.getApiService(currentServerUrl, httpClient)
                        val reg = apiService.register(accessCode, RegisterRequest(Build.MODEL, "User-${clientId.take(4)}", clientId))
                        TagSignalSender.configure(apiService, accessCode, reg.client_id)
                        
                        val response = apiService.getCurrentModel(0)
                        val serverVersion = response.headers()["X-Model-Version"]?.toIntOrNull() ?: reg.model_version
                        activeWeights = WeightSerializer.deserialize(response.body()?.string() ?: "")
                        currentModelVersion = serverVersion
                        
                        withContext(Dispatchers.Main) {
                            wsClient.connect(currentServerUrl, reg.client_id, accessCode)
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
                    unfocusedContainerColor = FGTColors.BgSurface,
                    focusedContainerColor = FGTColors.BgSurface,
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

@Composable
fun PhotosBottomNav(selectedTab: Int, onTabSelected: (Int) -> Unit) {
    GlassContainer(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp)
            .height(64.dp),
        cornerRadius = 32.dp,
        backgroundColor = FGTColors.BgGlass.copy(alpha = 0.9f)
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            val items = listOf(
                Icons.Default.Photo to "Photos",
                Icons.Default.Favorite to "For You",
                Icons.Default.CollectionsBookmark to "Albums",
                Icons.Default.Search to "Search"
            )
            items.forEachIndexed { index, (icon, label) ->
                val isSelected = selectedTab == index
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .clip(CircleShape)
                        .clickable { onTabSelected(index) }
                        .padding(8.dp)
                ) {
                    Icon(
                        imageVector = icon,
                        contentDescription = label,
                        tint = if (isSelected) FGTColors.AccentPrimary else FGTColors.TextSecondary,
                        modifier = Modifier.size(if (isSelected) 28.dp else 24.dp)
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
        val parts = code.split("@")
        if (code.isBlank() || parts.size != 2 || parts[0].isBlank() || parts[1].isBlank()) {
            codeError = "Use the format IP:PORT@ACCESS_CODE"
            return
        }
        codeError = null
        onConnect(code)
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.CloudSync, null, tint = FGTColors.AccentPrimary)
            Spacer(Modifier.width(12.dp))
            Text("Model Sync")
        }},
        text = {
            Column {
                Text(
                    if (isConnected) statusText else "Enter your coordinator access code to sync the model. Your photos never leave this device.",
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
                        label = { Text("Coordinator Access Code") },
                        placeholder = { Text("IP:PORT@ACCESS_CODE") },
                        isError = codeError != null,
                        supportingText = codeError?.let { { Text(it, color = FGTColors.Error) } },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        },
        confirmButton = {
            if (!isConnected) {
                Button(onClick = { tryConnect() }) { Text("Sync Now") }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = onDisconnect, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) { Text("Disconnect") }
                    TextButton(onClick = onDismiss) { Text("Done") }
                }
            }
        }
    )
}

@Composable
fun PermissionScreen(onRequest: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(Icons.Outlined.PhotoLibrary, null, Modifier.size(80.dp), tint = FGTColors.AccentPrimary)
        Spacer(Modifier.height(24.dp))
        Text("Allow access to your photos", style = MaterialTheme.typography.headlineSmall, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold, color = FGTColors.TextPrimary)
        Spacer(Modifier.height(12.dp))
        Text("To organize your gallery automatically, FGT needs permission to see your photos. Your photos never leave this device.", textAlign = androidx.compose.ui.text.style.TextAlign.Center, color = FGTColors.TextSecondary)
        Spacer(Modifier.height(32.dp))
        Button(onClick = onRequest, modifier = Modifier.fillMaxWidth().height(48.dp)) { Text("Allow Access") }
    }
}
