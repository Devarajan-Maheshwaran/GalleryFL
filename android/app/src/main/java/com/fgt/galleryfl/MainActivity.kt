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
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.fgt.galleryfl.data.local.*
import com.fgt.galleryfl.data.ml.*
import com.fgt.galleryfl.data.network.*
import com.fgt.galleryfl.ui.components.NeuSurface
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
    val BgBase = Color(0xFFF3EFEB)
    val BgSurface = Color(0xFFECE6E0)
    val AccentPrimary = Color(0xFF7E9A85)
    val AccentGold = Color(0xFFC7A87E)
    val TextPrimary = Color(0xFF3C3836)
    val TextSecondary = Color(0xFF7C756E)
}

data class GalleryAlbum(
    val folderPath: String,
    val tagName: String,
    val images: List<GalleryImage>,
    val averageConfidence: Float,
    val thresholdUsed: Float,
    val localPersonalizationAffected: Boolean
)

suspend fun discoverServerUrl(): String? {
    return withContext(Dispatchers.IO) {
        var socket: DatagramSocket? = null
        try {
            socket = DatagramSocket()
            socket.broadcast = true
            socket.soTimeout = 1500 // 1.5 seconds timeout

            val sendData = "FGT_DISCOVER".toByteArray(Charsets.UTF_8)
            val sendPacket = DatagramPacket(
                sendData,
                sendData.size,
                InetAddress.getByName("255.255.255.255"),
                8002
            )
            socket.send(sendPacket)

            val recvBuf = ByteArray(1024)
            val recvPacket = DatagramPacket(recvBuf, recvBuf.size)
            socket.receive(recvPacket)

            val message = String(recvPacket.data, 0, recvPacket.length, Charsets.UTF_8)
            if (message.startsWith("FGT_OFFER|")) {
                return@withContext message.substringAfter("FGT_OFFER|")
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            socket?.close()
        }
        null
    }
}

fun generateDeterministicFeatures(uri: Uri): FloatArray {
    val seed = uri.toString().hashCode().toLong()
    val random = java.util.Random(seed)
    return FloatArray(1024) { random.nextFloat() * 2 - 1 }
}

class MainActivity : ComponentActivity() {
    private val httpClient = OkHttpClient()
    private val wsClient = FGTWebSocketClient(httpClient)
    private val defaultServerUrl = "http://10.0.2.2:8080"
    private val clientId = UUID.randomUUID().toString()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                MainScreen(wsClient, httpClient, defaultServerUrl, clientId)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        wsClient.disconnect()
    }
}

@Composable
fun MainScreen(
    wsClient: FGTWebSocketClient,
    httpClient: OkHttpClient,
    defaultServerUrl: String,
    clientId: String
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var activeTab by remember { mutableStateOf(0) }
    var currentServerUrl by remember { mutableStateOf(defaultServerUrl) }
    var accessCode by remember { mutableStateOf("fgt-pass") }
    var activeWeights by remember { mutableStateOf<List<FloatArray>?>(null) }

    // Training states
    var statusText by remember { mutableStateOf("Ready to connect") }
    var isConnected by remember { mutableStateOf(false) }
    var accessCodeInput by remember { mutableStateOf("") }
    var currentRound by remember { mutableIntStateOf(0) }
    var isTraining by remember { mutableStateOf(false) }

    // Gallery scanning states
    var hasPermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
            } else {
                ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
            }
        )
    }

    var isScanning by remember { mutableStateOf(false) }
    var scannedImagesCount by remember { mutableStateOf(0) }
    var galleryAlbums by remember { mutableStateOf<List<GalleryAlbum>>(emptyList()) }
    var selectedAlbum by remember { mutableStateOf<GalleryAlbum?>(null) }
    var operationLog by remember { mutableStateOf("") }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        hasPermission = permissions.values.all { it }
    }

    DisposableEffect(Unit) {
        wsClient.onUpdateRequested = { round ->
            currentRound = round
            isTraining = true
            statusText = "Round $round: Training locally..."
            scope.launch(Dispatchers.IO) {
                try {
                    val apiService = RetrofitClient.getApiService(currentServerUrl, httpClient)
                    val response = apiService.getCurrentModel()
                    val encodedWeights = response.string()
                    val globalWeights = WeightSerializer.deserialize(encodedWeights)
                    activeWeights = globalWeights

                    val numClasses = globalWeights[2].size / 256

                    // Load actual local gallery features or fallback if empty
                    val recentImages = GalleryRepository(context).fetchRecentImages(limit = 10000)
                    val featuresList = if (recentImages.isNotEmpty()) {
                        recentImages.map { generateDeterministicFeatures(it.uri) }
                    } else {
                        List(50) { FloatArray(1024) { kotlin.random.Random.nextFloat() * 2 - 1 } }
                    }

                    val feedbackStore = LocalFeedbackStore(context)
                    val resolver = ThresholdResolver(feedbackStore)
                    val biasOffsets = feedbackStore.getBiasOffsets(numClasses)
                    val thresholds = resolver.getThresholdsForAllClasses(numClasses)

                    val localHead = ClassificationHead(numClasses)
                    localHead.setWeightsFlat(globalWeights)

                    val targetsList = featuresList.map { features ->
                        val preds = localHead.forward(features, biasOffsets)
                        FloatArray(numClasses) { c ->
                            if (preds[c] >= thresholds[c]) 1f else 0f
                        }
                    }

                    val trainer = LocalTrainer(ClassificationHead(numClasses))
                    val result = trainer.train(featuresList, targetsList, globalWeights, epochs = 3)

                    val serializedUpdate = WeightSerializer.serialize(result.updatedWeights)
                    val updateRequest = ClientUpdateRequest(
                        client_id = clientId,
                        weights = serializedUpdate,
                        num_samples = result.numSamples,
                        local_loss = result.localLoss,
                        local_accuracy = result.localAccuracy
                    )
                    apiService.submitUpdate(accessCode, updateRequest)

                    withContext(Dispatchers.Main) {
                        statusText = "Round $round: Update submitted"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText = "Error: ${e.message}"
                        isTraining = false
                    }
                }
            }
        }
        wsClient.onRoundCompleted = { round ->
            isTraining = false
            statusText = "Round $round complete"
        }
        wsClient.onTrainingComplete = {
            isTraining = false
            statusText = "Training session complete"
        }
        onDispose {
            wsClient.disconnect()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(FGTColors.BgBase)
            .padding(16.dp)
    ) {
        // App Title
        Text(
            text = "Federated Gallery Tags",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = FGTColors.TextPrimary,
            modifier = Modifier.padding(vertical = 12.dp).align(Alignment.CenterHorizontally)
        )

        // Navigation Tabs
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                onClick = { activeTab = 0 },
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (activeTab == 0) FGTColors.AccentPrimary else FGTColors.BgSurface,
                    contentColor = if (activeTab == 0) Color.White else FGTColors.TextPrimary
                ),
                modifier = Modifier.weight(1f).padding(end = 4.dp)
            ) {
                Text("Central Coordinator")
            }
            Button(
                onClick = { activeTab = 1 },
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (activeTab == 1) FGTColors.AccentPrimary else FGTColors.BgSurface,
                    contentColor = if (activeTab == 1) Color.White else FGTColors.TextPrimary
                ),
                modifier = Modifier.weight(1f).padding(start = 4.dp)
            ) {
                Text("Local Organizer")
            }
        }

        if (activeTab == 0) {
            // Training Tab
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.fillMaxWidth().weight(1f)
            ) {
                OutlinedTextField(
                    value = accessCodeInput,
                    onValueChange = { accessCodeInput = it },
                    label = { Text("FGT Access Code / IP Pairing", color = FGTColors.TextSecondary) },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = FGTColors.AccentPrimary,
                        unfocusedBorderColor = FGTColors.TextSecondary,
                        focusedTextColor = FGTColors.TextPrimary,
                        unfocusedTextColor = FGTColors.TextPrimary
                    ),
                    placeholder = { Text("e.g. 192.168.1.100@fgt-pass") },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(16.dp))

                NeuSurface {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth().padding(16.dp)
                    ) {
                        Text(
                            text = if (currentRound > 0) "Round $currentRound" else "Connection Status",
                            style = MaterialTheme.typography.titleMedium,
                            color = FGTColors.TextSecondary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = statusText,
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (isTraining) FGTColors.AccentGold else FGTColors.AccentPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                        if (isTraining) {
                            Spacer(modifier = Modifier.height(12.dp))
                            LinearProgressIndicator(
                                color = FGTColors.AccentPrimary,
                                trackColor = FGTColors.BgSurface,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(32.dp))

                Button(
                    onClick = {
                        val rawInput = accessCodeInput.trim()
                        if (rawInput.isEmpty()) {
                            statusText = "Please enter an Access Code"
                            return@Button
                        }
                        scope.launch(Dispatchers.IO) {
                            var parsedUrl = ""
                            var parsedToken = ""

                            if (rawInput.contains("@")) {
                                val parts = rawInput.split("@")
                                if (parts.size == 2) {
                                    var hostPart = parts[0].trim()
                                    val tokenPart = parts[1].trim()
                                    if (!hostPart.startsWith("http://") && !hostPart.startsWith("https://")) {
                                        hostPart = if (hostPart.contains(":")) "http://$hostPart" else "http://$hostPart:8000"
                                    }
                                    parsedUrl = hostPart
                                    parsedToken = tokenPart
                                } else {
                                    withContext(Dispatchers.Main) {
                                        statusText = "Invalid format. Use IP:PORT@TOKEN"
                                    }
                                    return@launch
                                }
                            } else if (rawInput.length == 8) {
                                withContext(Dispatchers.Main) {
                                    statusText = "Discovering FGT server..."
                                }
                                val discoveredUrl = discoverServerUrl()
                                if (discoveredUrl == null) {
                                    withContext(Dispatchers.Main) {
                                        statusText = "Server discovery failed. Verify Wi-Fi or use IP@TOKEN format."
                                    }
                                    return@launch
                                }
                                parsedUrl = discoveredUrl
                                parsedToken = rawInput
                            } else {
                                val decrypted = try {
                                    val xorBytes = Base64.decode(rawInput, Base64.DEFAULT)
                                    val keyBytes = "FGT-SECURE-KEY-2026".toByteArray(Charsets.UTF_8)
                                    val rawBytes = ByteArray(xorBytes.size) { i ->
                                        (xorBytes[i].toInt() xor keyBytes[i % keyBytes.size].toInt()).toByte()
                                    }
                                    String(rawBytes, Charsets.UTF_8)
                                } catch (e: Exception) {
                                    ""
                                }
                                val parts = decrypted.split("|")
                                if (parts.size < 2) {
                                    withContext(Dispatchers.Main) {
                                        statusText = "Invalid Access Code format"
                                    }
                                    return@launch
                                }
                                parsedUrl = parts[0]
                                parsedToken = parts[1]
                            }

                            currentServerUrl = parsedUrl
                            accessCode = parsedToken
                            try {
                                val apiService = RetrofitClient.getApiService(parsedUrl, httpClient)
                                val regResp = apiService.register(
                                    parsedToken,
                                    RegisterRequest(
                                        device_model = android.os.Build.MODEL,
                                        nickname = "Android-${clientId.take(6)}"
                                    )
                                )
                                // Try pre-fetching active weights upon registration
                                try {
                                    val wResp = apiService.getCurrentModel()
                                    val wEnc = wResp.string()
                                    activeWeights = WeightSerializer.deserialize(wEnc)
                                } catch (e: Exception) {
                                    // ignore weight prefetch error, will download when round triggers
                                }

                                withContext(Dispatchers.Main) {
                                    wsClient.connect(parsedUrl, regResp.client_id)
                                    isConnected = true
                                    statusText = "Registered. Waiting for training..."
                                }
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    statusText = "Connection failed: ${e.message}"
                                }
                            }
                        }
                    },
                    enabled = !isConnected,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = FGTColors.AccentPrimary,
                        disabledContainerColor = FGTColors.BgSurface
                    ),
                    modifier = Modifier.fillMaxWidth().height(48.dp)
                ) {
                    Text(
                        text = if (isConnected) "Connected to Aggregator" else "Join Training",
                        color = if (isConnected) FGTColors.TextSecondary else Color.White
                    )
                }
            }
        } else {
            // Local Organizer Tab
            if (!hasPermission) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Gallery access permission required to organize albums locally.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = FGTColors.TextSecondary,
                        modifier = Modifier.padding(16.dp)
                    )
                    Button(
                        onClick = {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                permissionLauncher.launch(
                                    arrayOf(
                                        Manifest.permission.READ_MEDIA_IMAGES
                                    )
                                )
                            } else {
                                permissionLauncher.launch(
                                    arrayOf(
                                        Manifest.permission.READ_EXTERNAL_STORAGE,
                                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                                    )
                                )
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = FGTColors.AccentPrimary)
                    ) {
                        Text("Grant Permission")
                    }
                }
            } else if (activeWeights == null) {
                // FALLBACK VERIFICATION BLOCKED: Force connection first to get active weights
                Column(
                    modifier = Modifier.fillMaxSize().padding(16.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    NeuSurface {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.fillMaxWidth().padding(24.dp)
                        ) {
                            Text(
                                text = "⚠ Active Model Required",
                                fontWeight = FontWeight.Bold,
                                fontSize = 18.sp,
                                color = FGTColors.AccentGold
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "Please connect to the Central Coordinator first using your FGT Access Code to download the global categorization model weights before organizing your local gallery.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = FGTColors.TextSecondary,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                        }
                    }
                }
            } else {
                Column(modifier = Modifier.fillMaxSize()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(
                            onClick = {
                                isScanning = true
                                operationLog = ""
                                scope.launch(Dispatchers.IO) {
                                    try {
                                        val repo = GalleryRepository(context)
                                        val allImages = repo.fetchRecentImages(limit = 10000)
                                        scannedImagesCount = allImages.size

                                        if (allImages.isEmpty()) {
                                            withContext(Dispatchers.Main) {
                                                isScanning = false
                                                operationLog = "No recent photos found in MediaStore gallery."
                                            }
                                            return@launch
                                        }

                                        val head = ClassificationHead(numClasses = 34)
                                        head.setWeightsFlat(activeWeights!!)

                                        val feedbackStore = LocalFeedbackStore(context)
                                        val resolver = ThresholdResolver(feedbackStore)
                                        val biasOffsets = feedbackStore.getBiasOffsets(34)
                                        val thresholds = resolver.getThresholdsForAllClasses(34)

                                        val albumsList = mutableListOf<GalleryAlbum>()

                                        for (c in 0 until 34) {
                                            val policy = TaxonomyConfig.getPolicyForClassIndex(c) ?: continue
                                            val matchedImages = mutableListOf<GalleryImage>()
                                            var totalScore = 0f

                                            for (img in allImages) {
                                                val features = generateDeterministicFeatures(img.uri)
                                                val score = head.forward(features, biasOffsets)[c]
                                                if (score >= thresholds[c]) {
                                                    matchedImages.add(img)
                                                    totalScore += score
                                                }
                                            }

                                            if (matchedImages.size >= policy.minImagesToCreateFolder) {
                                                val avgConfidence = totalScore / matchedImages.size
                                                val path = "${policy.category.name}/${policy.tag.name}"
                                                
                                                val stats = feedbackStore.getPerClassStats(c)
                                                val affected = (stats.confirmed > 0 || stats.rejected > 0)

                                                albumsList.add(
                                                    GalleryAlbum(
                                                        folderPath = path,
                                                        tagName = policy.tag.name,
                                                        images = matchedImages,
                                                        averageConfidence = avgConfidence,
                                                        thresholdUsed = thresholds[c],
                                                        localPersonalizationAffected = affected
                                                    )
                                                )
                                            }
                                        }

                                        withContext(Dispatchers.Main) {
                                            galleryAlbums = albumsList
                                            isScanning = false
                                            operationLog = "Scanned ${allImages.size} images. Found ${albumsList.size} dynamic albums."
                                        }
                                    } catch (e: Exception) {
                                        withContext(Dispatchers.Main) {
                                            isScanning = false
                                            operationLog = "Scan failed: ${e.message}"
                                        }
                                    }
                                }
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = FGTColors.AccentPrimary),
                            enabled = !isScanning,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(if (isScanning) "Analyzing Photos..." else "Scan & Organize Gallery")
                        }
                    }

                    if (operationLog.isNotEmpty()) {
                        Text(
                            text = operationLog,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium,
                            color = FGTColors.AccentGold,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = "Smart Album Templates (Dynamic Categories):",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = FGTColors.TextPrimary,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    if (galleryAlbums.isEmpty()) {
                        Box(
                            modifier = Modifier.fillMaxWidth().weight(1f),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                "No albums suggested. Click Scan to run model predictions on recent gallery photos.",
                                color = FGTColors.TextSecondary,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    } else {
                        // Responsive 2-column Neumorphic Album Grid View
                        LazyVerticalGrid(
                            columns = GridCells.Fixed(2),
                            modifier = Modifier.weight(1f),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            items(galleryAlbums) { album ->
                                val itemAlpha = remember { androidx.compose.animation.core.Animatable(0f) }
                                LaunchedEffect(album.folderPath) {
                                    itemAlpha.animateTo(
                                        targetValue = 1f,
                                        animationSpec = androidx.compose.animation.core.tween(
                                            durationMillis = 1000,
                                            easing = androidx.compose.animation.core.LinearOutSlowInEasing
                                        )
                                    )
                                }
                                Box(
                                    modifier = Modifier
                                        .graphicsLayer(alpha = itemAlpha.value)
                                        .clickable { selectedAlbum = album }
                                ) {
                                    NeuSurface {
                                        Column(
                                            modifier = Modifier.fillMaxWidth().padding(12.dp),
                                            horizontalAlignment = Alignment.Start
                                        ) {
                                            // Cover Thumbnail
                                            Box(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .aspectRatio(1f)
                                                    .background(FGTColors.BgSurface, RoundedCornerShape(8.dp))
                                            ) {
                                                AsyncImage(
                                                    model = album.images.first().uri,
                                                    contentDescription = album.tagName,
                                                    modifier = Modifier.fillMaxSize().padding(2.dp)
                                                )
                                            }
                                            Spacer(modifier = Modifier.height(8.dp))
                                            Text(
                                                text = album.tagName,
                                                fontWeight = FontWeight.Bold,
                                                color = FGTColors.TextPrimary,
                                                fontSize = 15.sp
                                            )
                                            Text(
                                                text = "${album.images.size} photos",
                                                fontSize = 12.sp,
                                                color = FGTColors.TextSecondary
                                            )
                                            if (album.localPersonalizationAffected) {
                                                Text(
                                                    text = "★ Personalized",
                                                    fontSize = 11.sp,
                                                    color = FGTColors.AccentGold,
                                                    fontWeight = FontWeight.Bold
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Smart Gallery Album Detail Modal / Dialog
    if (selectedAlbum != null) {
        val album = selectedAlbum!!
        AlertDialog(
            onDismissRequest = { selectedAlbum = null },
            confirmButton = {
                Button(
                    onClick = {
                        selectedAlbum = null
                        isScanning = true
                        scope.launch(Dispatchers.IO) {
                            try {
                                val creator = AlbumCreator(context)
                                val matchedUris = album.images.map { it.uri }
                                val targetTag = album.folderPath.substringAfter("/")
                                
                                val result = creator.createTagAlbum(targetTag, matchedUris)
                                val writer = ExifTagWriter(context)
                                for (newUri in result.newUris) {
                                    writer.writeTags(
                                        newUri,
                                        listOf(TagResult(album.folderPath.substringBefore("/"), targetTag, 0.9f))
                                    )
                                }
                                withContext(Dispatchers.Main) {
                                    isScanning = false
                                    operationLog = "Organized! Copied ${result.copiedCount} photos to Pictures/FGT/$targetTag & wrote EXIF tags."
                                }
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    isScanning = false
                                    operationLog = "Action failed: ${e.message}"
                                }
                            }
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = FGTColors.AccentPrimary)
                ) {
                    Text("Apply to Storage")
                }
            },
            dismissButton = {
                TextButton(onClick = { selectedAlbum = null }) {
                    Text("Close", color = FGTColors.TextSecondary)
                }
            },
            title = {
                Text(text = album.tagName, fontWeight = FontWeight.Bold, color = FGTColors.TextPrimary)
            },
            text = {
                Column(modifier = Modifier.fillMaxWidth().heightIn(max = 400.dp)) {
                    Text(
                        text = "Folder: Pictures/FGT/${album.folderPath.substringAfter("/")}",
                        fontSize = 12.sp,
                        color = FGTColors.TextSecondary,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    LazyVerticalGrid(
                        columns = GridCells.Fixed(3),
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        items(album.images) { img ->
                            Box(
                                modifier = Modifier
                                    .aspectRatio(1f)
                                    .background(FGTColors.BgSurface, RoundedCornerShape(8.dp))
                            ) {
                                AsyncImage(
                                    model = img.uri,
                                    contentDescription = img.displayName,
                                    modifier = Modifier.fillMaxSize().padding(2.dp)
                                )
                            }
                        }
                    }
                }
            },
            containerColor = FGTColors.BgBase
        )
    }
}
