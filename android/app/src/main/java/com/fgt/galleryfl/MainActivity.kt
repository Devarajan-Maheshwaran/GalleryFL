package com.fgt.galleryfl

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.fgt.galleryfl.data.local.*
import com.fgt.galleryfl.data.ml.*
import com.fgt.galleryfl.data.network.*
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import coil3.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.util.UUID

object FGTColors {
    val BgBase = Color(0xFFFFFFFF)
    val BgSurface = Color(0xFFF1F3F4)
    val AccentPrimary = Color(0xFF1A73E8) // Google Blue
    val TextPrimary = Color(0xFF202124)
    val TextSecondary = Color(0xFF5F6368)
}

data class GalleryAlbum(
    val folderPath: String,
    val tagName: String,
    val images: List<GalleryImage>,
    val averageConfidence: Float,
    val thresholdUsed: Float,
    val localPersonalizationAffected: Boolean
)

class MainActivity : ComponentActivity() {
    private val httpClient = OkHttpClient()
    private val wsClient = FGTWebSocketClient(httpClient)
    private val defaultServerUrl = "http://10.0.2.2:8000"
    private val clientId = UUID.randomUUID().toString()
    private lateinit var featureExtractor: FeatureExtractor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        featureExtractor = FeatureExtractor(this)
        setContent {
            FGTTheme {
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
fun FGTTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = FGTColors.AccentPrimary,
            surface = FGTColors.BgBase,
            onSurface = FGTColors.TextPrimary
        ),
        content = content
    )
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

    // FL State
    var currentServerUrl by remember { mutableStateOf(defaultServerUrl) }
    var accessCode by remember { mutableStateOf("fgt-pass") }
    var activeWeights by remember { mutableStateOf<List<FloatArray>?>(null) }
    var statusText by remember { mutableStateOf("Ready to sync") }
    var isConnected by remember { mutableStateOf(false) }
    var isTraining by remember { mutableStateOf(false) }
    var currentRound by remember { mutableIntStateOf(0) }
    var currentModelVersion by remember { mutableIntStateOf(0) }

    // Photos State
    var photos by remember { mutableStateOf<List<GalleryImage>>(emptyList()) }
    var hasPermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
            } else {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
            }
        )
    }

    // Explore State
    var smartAlbums by remember { mutableStateOf<List<GalleryAlbum>>(emptyList()) }
    var isScanning by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        hasPermission = permissions.values.all { it }
    }

    // Initial Load
    LaunchedEffect(hasPermission) {
        if (hasPermission) {
            photos = GalleryRepository(context).fetchRecentImages(limit = 500)
        }
    }

    // FL WebSocket Listeners
    DisposableEffect(Unit) {
        wsClient.onUpdateRequested = { round, lr, epochs ->
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

                    val trainer = LocalTrainer(ClassificationHead(numClasses))
                    val result = trainer.train(featuresList, targetsList, globalWeights, epochs = epochs, lr = lr)

                    apiService.submitUpdate(accessCode, ClientUpdateRequest(
                        client_id = clientId,
                        weights = WeightSerializer.serialize(result.updatedWeights),
                        num_samples = result.numSamples,
                        local_loss = result.localLoss,
                        local_accuracy = result.localAccuracy
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

    Scaffold(
        topBar = {
            PhotosTopBar(
                isTraining = isTraining,
                onProfileClick = { showFLDialog = true }
            )
        },
        bottomBar = {
            PhotosBottomNav(
                selectedTab = selectedTab,
                onTabSelected = { selectedTab = it }
            )
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
                    0 -> PhotosScreen(photos)
                    1 -> ExploreScreen(
                        smartAlbums = smartAlbums,
                        isScanning = isScanning,
                        onScanClick = {
                            isScanning = true
                            scope.launch(Dispatchers.IO) {
                                try {
                                    val repo = GalleryRepository(context)
                                    val allImages = repo.fetchRecentImages(limit = 200)
                                    if (activeWeights == null) {
                                        withContext(Dispatchers.Main) {
                                            isScanning = false
                                            statusText = "Connect to sync model first"
                                            showFLDialog = true
                                        }
                                        return@launch
                                    }

                                    val head = ClassificationHead(34)
                                    head.setWeightsFlat(activeWeights!!)
                                    val feedbackStore = LocalFeedbackStore(context)
                                    val thresholds = ThresholdResolver(feedbackStore).getThresholdsForAllClasses(34)
                                    val biasOffsets = feedbackStore.getBiasOffsets(34)

                                    val albumMap = mutableMapOf<Int, MutableList<Pair<GalleryImage, Float>>>()
                                    val confidenceMap = mutableMapOf<Int, Float>()

                                    for (img in allImages) {
                                        val bitmap = repo.loadBitmap(img.uri) ?: continue
                                        val features = featureExtractor.extractFeatures(bitmap).projection
                                        val preds = head.forward(features, biasOffsets)

                                        // WINNER TAKES ALL - find best class above threshold
                                        var bestClass = -1
                                        var bestMargin = -1f

                                        for (c in 0 until 34) {
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
                                    }
                                } catch (e: Exception) {
                                    withContext(Dispatchers.Main) { isScanning = false }
                                }
                            }
                        }
                    )
                    2 -> LibraryScreen(smartAlbums)
                }
            }
        }
    }

    if (showFLDialog) {
        FLSyncDialog(
            statusText = statusText,
            isConnected = isConnected,
            isTraining = isTraining,
            onDismiss = { showFLDialog = false },
            onConnect = { code ->
                scope.launch(Dispatchers.IO) {
                    try {
                        // Force use 10.0.2.2:8000 for emulator reliability regardless of input
                        val parsedToken = if (code.contains("@")) code.split("@")[1] else code
                        // Strip out any dashboard IP and force emulator localhost
                        currentServerUrl = "http://10.0.2.2:8000"
                        accessCode = if (parsedToken.isBlank()) "fgt-pass" else parsedToken

                        val apiService = RetrofitClient.getApiService(currentServerUrl, httpClient)
                        val reg = apiService.register(accessCode, RegisterRequest(Build.MODEL, "User-${clientId.take(4)}"))
                        
                        val response = apiService.getCurrentModel(0)
                        val serverVersion = response.headers()["X-Model-Version"]?.toIntOrNull() ?: reg.model_version
                        activeWeights = WeightSerializer.deserialize(response.body()?.string() ?: "")
                        currentModelVersion = serverVersion
                        
                        withContext(Dispatchers.Main) {
                            wsClient.connect(currentServerUrl, reg.client_id)
                            isConnected = true
                            statusText = "Synced with coordinator"
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) { statusText = "Sync failed: ${e.message}" }
                    }
                }
            }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotosTopBar(isTraining: Boolean, onProfileClick: () -> Unit) {
    TopAppBar(
        title = {
            Surface(
                tonalElevation = 3.dp,
                shape = RoundedCornerShape(24.dp),
                modifier = Modifier.fillMaxWidth().height(48.dp).padding(end = 16.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(horizontal = 16.dp)
                ) {
                    Icon(Icons.Default.Search, contentDescription = null, tint = FGTColors.TextSecondary)
                    Spacer(modifier = Modifier.width(12.dp))
                    Text("Search your photos", color = FGTColors.TextSecondary, style = MaterialTheme.typography.bodyMedium)
                }
            }
        },
        actions = {
            IconButton(onClick = onProfileClick) {
                Box {
                    Icon(
                        imageVector = Icons.Outlined.AccountCircle,
                        contentDescription = "Profile",
                        modifier = Modifier.size(32.dp),
                        tint = FGTColors.AccentPrimary
                    )
                    if (isTraining) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(32.dp),
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
    NavigationBar(containerColor = FGTColors.BgBase) {
        val items = listOf(
            Triple("Photos", Icons.Default.Photo, Icons.Outlined.Photo),
            Triple("Explore", Icons.Default.Search, Icons.Outlined.Search),
            Triple("Library", Icons.Default.CollectionsBookmark, Icons.Outlined.CollectionsBookmark)
        )
        items.forEachIndexed { index, (label, selectedIcon, unselectedIcon) ->
            NavigationBarItem(
                selected = selectedTab == index,
                onClick = { onTabSelected(index) },
                label = { Text(label) },
                icon = { Icon(if (selectedTab == index) selectedIcon else unselectedIcon, contentDescription = label) }
            )
        }
    }
}

@Composable
fun PhotosScreen(photos: List<GalleryImage>) {
    LazyVerticalGrid(
        columns = GridCells.Fixed(3),
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(1.dp),
        horizontalArrangement = Arrangement.spacedBy(1.dp),
        verticalArrangement = Arrangement.spacedBy(1.dp)
    ) {
        items(photos) { photo ->
            AsyncImage(
                model = photo.uri,
                contentDescription = null,
                modifier = Modifier.aspectRatio(1f),
                contentScale = ContentScale.Crop
            )
        }
    }
}

@Composable
fun ExploreScreen(
    smartAlbums: List<GalleryAlbum>,
    isScanning: Boolean,
    onScanClick: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Explore", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(16.dp))

        if (smartAlbums.isEmpty() && !isScanning) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Outlined.AutoAwesome, null, Modifier.size(64.dp), tint = FGTColors.AccentPrimary)
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Your smart albums will appear here", textAlign = TextAlign.Center, color = FGTColors.TextSecondary)
                    Spacer(modifier = Modifier.height(24.dp))
                    Button(onClick = onScanClick) { Text("Scan & Organize") }
                }
            }
        } else if (isScanning) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Running smart classification...", color = FGTColors.TextSecondary)
                }
            }
        } else {
            // HIERARCHICAL VIEW
            val categories = smartAlbums.groupBy { it.folderPath.substringBefore("/") }
            LazyColumn(verticalArrangement = Arrangement.spacedBy(24.dp)) {
                categories.forEach { (catId, albums) ->
                    item {
                        Column {
                            Text(
                                text = catId.replaceFirstChar { it.uppercase() },
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = FGTColors.TextPrimary
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            LazyVerticalGrid(
                                columns = GridCells.Fixed(2),
                                modifier = Modifier.heightIn(max = 1000.dp), // Adjust or use non-nested grid if possible
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp),
                                userScrollEnabled = false
                            ) {
                                items(albums) { album ->
                                    SmartAlbumCard(album)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SmartAlbumCard(album: GalleryAlbum) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Box(
            modifier = Modifier
                .aspectRatio(1f)
                .clip(RoundedCornerShape(16.dp))
                .background(FGTColors.BgSurface)
        ) {
            AsyncImage(
                model = album.images.firstOrNull()?.uri,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(album.tagName, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text("${album.images.size} items", style = MaterialTheme.typography.bodySmall, color = FGTColors.TextSecondary)
    }
}

@Composable
fun LibraryScreen(smartAlbums: List<GalleryAlbum>) {
    LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item { Text("Library", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold) }
        
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                LibraryActionItem("Favorites", Icons.Default.Favorite, Color(0xFFE91E63))
                LibraryActionItem("Utilities", Icons.Default.Build, Color(0xFF607D8B))
                LibraryActionItem("Archive", Icons.Default.Archive, Color(0xFF795548))
                LibraryActionItem("Trash", Icons.Default.Delete, Color(0xFFF44336))
            }
        }

        item {
            Text("Photos on device", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 16.dp))
        }

        items(smartAlbums) { album ->
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Box(Modifier.size(64.dp).clip(RoundedCornerShape(8.dp)).background(FGTColors.BgSurface)) {
                    AsyncImage(model = album.images.firstOrNull()?.uri, contentDescription = null, contentScale = ContentScale.Crop)
                }
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text(album.tagName, fontWeight = FontWeight.Bold)
                    Text("Pictures/FGT/${album.folderPath.substringAfter("/")}", style = MaterialTheme.typography.bodySmall, color = FGTColors.TextSecondary)
                }
            }
        }
    }
}

@Composable
fun LibraryActionItem(label: String, icon: ImageVector, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(icon, contentDescription = label, tint = color, modifier = Modifier.size(28.dp))
        Spacer(modifier = Modifier.height(4.dp))
        Text(label, style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
fun FLSyncDialog(
    statusText: String,
    isConnected: Boolean,
    isTraining: Boolean,
    onDismiss: () -> Unit,
    onConnect: (String) -> Unit
) {
    var codeInput by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.CloudSync, null, tint = FGTColors.AccentPrimary)
            Spacer(Modifier.width(12.dp))
            Text("Model Sync")
        }},
        text = {
            Column {
                Text(statusText, style = MaterialTheme.typography.bodyMedium, color = FGTColors.TextSecondary)
                if (isTraining) {
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                if (!isConnected) {
                    Spacer(Modifier.height(16.dp))
                    OutlinedTextField(
                        value = codeInput,
                        onValueChange = { codeInput = it },
                        label = { Text("FGT Access Code") },
                        placeholder = { Text("IP:PORT@TOKEN") }
                    )
                }
            }
        },
        confirmButton = {
            if (!isConnected) {
                Button(onClick = { onConnect(codeInput) }) { Text("Sync Now") }
            } else {
                TextButton(onClick = onDismiss) { Text("Done") }
            }
        }
    )
}

@Composable
fun PermissionScreen(onRequest: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(Icons.Outlined.PhotoLibrary, null, Modifier.size(80.dp), tint = FGTColors.AccentPrimary)
        Spacer(Modifier.height(24.dp))
        Text("Allow access to your photos", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(12.dp))
        Text("To organize your gallery automatically, FGT needs permission to see your photos. Your photos never leave this device.", textAlign = TextAlign.Center, color = FGTColors.TextSecondary)
        Spacer(Modifier.height(32.dp))
        Button(onClick = onRequest, modifier = Modifier.fillMaxWidth().height(48.dp)) { Text("Allow Access") }
    }
}
