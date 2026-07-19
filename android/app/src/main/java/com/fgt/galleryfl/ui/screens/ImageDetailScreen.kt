package com.fgt.galleryfl.ui.screens

import android.content.Intent
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.outlined.Visibility
import androidx.compose.material.icons.outlined.VisibilityOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.AppDatabase
import com.fgt.galleryfl.data.local.GalleryImage
import com.fgt.galleryfl.data.local.GalleryRepository
import com.fgt.galleryfl.data.local.LocalFeedbackStore
import com.fgt.galleryfl.data.local.RecordTagFeedbackUseCase
import com.fgt.galleryfl.data.ml.ClassificationHead
import com.fgt.galleryfl.data.ml.FeatureExtractor
import com.fgt.galleryfl.data.ml.HeatmapGenerator
import com.fgt.galleryfl.data.network.TagSignalSender
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import com.fgt.galleryfl.ui.theme.FGTColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageDetailScreen(
    image: GalleryImage,
    onBack: () -> Unit,
    onDelete: (GalleryImage) -> Unit,
    featureExtractor: FeatureExtractor,
    activeWeights: List<FloatArray>?
) {
    val context = LocalContext.current
    val density = LocalDensity.current
    val scope = rememberCoroutineScope()

    var showInfo by remember { mutableStateOf(false) }
    var showHeatmap by remember { mutableStateOf(false) }
    var heatmap by remember { mutableStateOf<Array<FloatArray>?>(null) }
    var predictedTag by remember { mutableStateOf<String?>(null) }
    var predictedClassIndex by remember { mutableStateOf(-1) }

    // Flick-to-dismiss state.
    val dragOffsetY = remember { mutableStateOf(0f) }
    var isDismissing by remember { mutableStateOf(false) }
    val dismissPx = with(density) { 150.dp.toPx() }
    val drag = dragOffsetY.value
    val bgAlpha = (1f - (abs(drag) / 800f)).coerceIn(0f, 1f)
    val scale = 1f - min(0.25f, abs(drag) / 800f)

    // Predict the tag once when the photo opens (decoupled from the heatmap
    // toggle) so the correction chips are always available.
    LaunchedEffect(image.id) {
        if (activeWeights == null) return@LaunchedEffect
        withContext(Dispatchers.IO) {
            try {
                val repo = GalleryRepository(context)
                val bmp = repo.loadBitmap(image.uri) ?: return@withContext
                val features = featureExtractor.extractFeatures(bmp)
                val numClasses = activeWeights[2].size / 256
                val head = ClassificationHead(numClasses)
                head.setWeightsFlat(activeWeights)
                val preds = head.forward(features.projection)
                val topClass = preds.indices.maxByOrNull { preds[it] } ?: -1
                val name = if (topClass >= 0) TaxonomyConfig.leafTags.getOrNull(topClass)?.name else null
                withContext(Dispatchers.Main) {
                    predictedClassIndex = topClass
                    predictedTag = name
                    name?.let { TagSignalSender.emit(listOf(it)) }
                }
            } catch (_: Exception) {
                // Prediction is best-effort; ignore failures.
            }
        }
    }

    // Recompute the GradCAM-style heatmap only when the toggle is on.
    LaunchedEffect(showHeatmap, image.id) {
        if (!showHeatmap) {
            heatmap = null
            return@LaunchedEffect
        }
        if (activeWeights == null) return@LaunchedEffect
        withContext(Dispatchers.IO) {
            try {
                val repo = GalleryRepository(context)
                val bmp = repo.loadBitmap(image.uri) ?: return@withContext
                val features = featureExtractor.extractFeatures(bmp)
                val numClasses = activeWeights[2].size / 256
                val head = ClassificationHead(numClasses)
                head.setWeightsFlat(activeWeights)
                val preds = head.forward(features.projection)
                val topClass = preds.indices.maxByOrNull { preds[it] } ?: -1
                val hm = HeatmapGenerator(head).generateHeatmap(features.spatialMap, topClass)
                withContext(Dispatchers.Main) { heatmap = hm }
            } catch (_: Exception) {
                // Heatmap is best-effort; ignore failures.
            }
        }
    }

    Scaffold(
        containerColor = Color.Black.copy(alpha = bgAlpha.coerceIn(0f, 1f)),
        topBar = {
            TopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Black.copy(alpha = 0.5f),
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White,
                    actionIconContentColor = Color.White
                ),
                title = { Text(image.displayName, style = MaterialTheme.typography.titleMedium) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = {
                        if (activeWeights != null) showHeatmap = !showHeatmap
                        else showInfo = true
                    }) {
                        Icon(
                            imageVector = if (showHeatmap) Icons.Outlined.VisibilityOff else Icons.Outlined.Visibility,
                            contentDescription = "Toggle heatmap",
                            tint = if (showHeatmap) FGTColors.AccentPrimary else Color.White
                        )
                    }
                    IconButton(onClick = {
                        val shareIntent = Intent(Intent.ACTION_SEND).apply {
                            type = "image/*"
                            putExtra(Intent.EXTRA_STREAM, image.uri)
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        context.startActivity(Intent.createChooser(shareIntent, "Share Image"))
                    }) {
                        Icon(Icons.Default.Share, contentDescription = "Share")
                    }
                    IconButton(onClick = { showInfo = !showInfo }) {
                        Icon(Icons.Default.Info, contentDescription = "Info")
                    }
                    IconButton(onClick = { onDelete(image) }) {
                        Icon(Icons.Default.Delete, contentDescription = "Delete", tint = FGTColors.Error)
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .offset { IntOffset(0, drag.roundToInt()) }
                .scale(scale, scale)
                .pointerInput(Unit) {
                    detectVerticalDragGestures(
                        onVerticalDrag = { _, dragAmount ->
                            if (!isDismissing) dragOffsetY.value += dragAmount
                        },
                        onDragEnd = {
                            if (abs(dragOffsetY.value) > dismissPx) {
                                isDismissing = true
                                onBack()
                            } else {
                                dragOffsetY.value = 0f
                            }
                        }
                    )
                },
            contentAlignment = Alignment.Center
        ) {
            AsyncImage(
                model = image.uri,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )

            // Approximate GradCAM overlay: a 7x7 grid coloured by the
            // HeatmapGenerator scores for the predicted class.
            if (showHeatmap && heatmap != null) {
                Box(Modifier.matchParentSize()) {
                    Column(Modifier.fillMaxSize()) {
                        for (y in 0..6) {
                            Row(Modifier.weight(1f).fillMaxWidth()) {
                                for (x in 0..6) {
                                    val v = (heatmap!![y][x] * 0.75f).coerceIn(0f, 0.85f)
                                    Box(
                                        Modifier
                                            .weight(1f)
                                            .fillMaxSize()
                                            .background(Color(1f, 1f, 1f, v))
                                    )
                                }
                            }
                        }
                    }
                }
            }

            if (showHeatmap && predictedTag != null) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 8.dp),
                    color = FGTColors.AccentPrimary.copy(alpha = 0.9f),
                    contentColor = Color.White,
                    shape = MaterialTheme.shapes.small
                ) {
                    Text(
                        "Predicted: $predictedTag",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }

            // Inline smart-tag chip with correction (user rejects -> local FL
            // correction is logged and the on-device model retrains on idle).
            if (predictedTag != null) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 16.dp),
                    color = FGTColors.BgSurface.copy(alpha = 0.92f),
                    contentColor = FGTColors.TextPrimary,
                    shape = RoundedCornerShape(20.dp),
                    shadowElevation = 6.dp
                ) {
                    Row(
                        modifier = Modifier.padding(6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            predictedTag!!.replace("_", " "),
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                        )
                        IconButton(
                            onClick = {
                                scope.launch(Dispatchers.IO) {
                                    try {
                                        val store = LocalFeedbackStore(context)
                                        val dao = AppDatabase.getDatabase(context).feedbackDao()
                                        RecordTagFeedbackUseCase(store, dao)
                                            .recordRejected(image.id, predictedClassIndex)
                                        withContext(Dispatchers.Main) {
                                            predictedTag = null
                                            Toast.makeText(
                                                context,
                                                "Tag removed. GalleryFL learns from your corrections locally.",
                                                Toast.LENGTH_SHORT
                                            ).show()
                                        }
                                    } catch (_: Exception) {
                                    }
                                }
                            }
                        ) {
                            Icon(
                                Icons.Default.Close,
                                contentDescription = "Remove tag",
                                tint = FGTColors.Error
                            )
                        }
                    }
                }
            }

            if (showInfo) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth(),
                    color = Color.Black.copy(alpha = 0.8f),
                    contentColor = Color.White
                ) {
                    Column(modifier = Modifier.padding(24.dp)) {
                        Text("Info", style = MaterialTheme.typography.titleLarge)
                        Spacer(Modifier.height(8.dp))
                        Text("Path: ${image.uri.path}", style = MaterialTheme.typography.bodySmall)
                        Text("Date: ${java.util.Date(image.dateAdded * 1000)}", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}
