package com.fgt.galleryfl.ui.screens

import android.content.Context
import android.content.Intent
import android.os.VibrationEffect
import android.os.Vibrator
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryImage
import com.fgt.galleryfl.ui.theme.FGTColors
import kotlin.math.min

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotosScreen(
    photos: List<GalleryImage>,
    onImageClick: (GalleryImage) -> Unit
) {
    var columns by remember { mutableStateOf(3) }
    val gridState = rememberLazyGridState()
    val context = LocalContext.current
    var peekImage by remember { mutableStateOf<GalleryImage?>(null) }
    val sheetState = rememberModalBottomSheetState()

    fun performHaptic(ctx: Context) {
        val vibrator = ctx.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        vibrator?.vibrate(VibrationEffect.createOneShot(18, VibrationEffect.DEFAULT_AMPLITUDE))
    }
    
    // Calculate if we should show the collapsed small title
    val showSmallTitle by remember {
        derivedStateOf { gridState.firstVisibleItemIndex > 0 }
    }

    Column(modifier = Modifier.fillMaxSize().background(FGTColors.BgBase)) {
        // Large Header (iOS Style)
        AnimatedVisibility(
            visible = !showSmallTitle,
            enter = fadeIn() + expandVertically(),
            exit = fadeOut() + shrinkVertically()
        ) {
            Text(
                text = "Photos",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = FGTColors.TextPrimary,
                modifier = Modifier.padding(start = 16.dp, top = 24.dp, bottom = 16.dp)
            )
        }

        if (photos.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(32.dp)
                ) {
                    Surface(
                        shape = RoundedCornerShape(20.dp),
                        color = FGTColors.BgSurface,
                        tonalElevation = 2.dp
                    ) {
                        Text(
                            "🖼",
                            fontSize = 40.sp,
                            modifier = Modifier.padding(20.dp)
                        )
                    }
                    Spacer(Modifier.height(16.dp))
                    Text(
                        "No photos yet",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = FGTColors.TextPrimary
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "Your gallery will appear here once photos are available on this device.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = FGTColors.TextSecondary,
                        modifier = Modifier.padding(horizontal = 24.dp)
                    )
                }
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(columns),
                state = gridState,
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, _, zoom, _ ->
                            columns = (columns * zoom).toInt().coerceIn(3, 6)
                        }
                    },
                contentPadding = PaddingValues(1.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                itemsIndexed(
                    photos,
                    key = { _, item -> item.id },
                    span = { index, _ ->
                        val isHighlight = (index + 1) % 12 == 0
                        GridItemSpan(if (isHighlight) min(2, columns) else 1)
                    }
                ) { _, photo ->
                    AsyncImage(
                        model = photo.uri,
                        contentDescription = null,
                        modifier = Modifier
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(12.dp))
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onTap = { onImageClick(photo) },
                                    onLongPress = { peekImage = photo; performHaptic(context) }
                                )
                            },
                        contentScale = ContentScale.Crop
                    )
                }
            }
        }

        peekImage?.let { img ->
            ModalBottomSheet(onDismissRequest = { peekImage = null }, sheetState = sheetState) {
                Column(Modifier.padding(16.dp)) {
                    Text(img.displayName, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = FGTColors.TextPrimary)
                    Spacer(Modifier.height(12.dp))
                    Row(Modifier.fillMaxWidth().clickable {
                        val si = Intent(Intent.ACTION_SEND).apply {
                            type = "image/*"; putExtra(Intent.EXTRA_STREAM, img.uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        context.startActivity(Intent.createChooser(si, "Share"))
                        peekImage = null
                    }.padding(vertical = 14.dp)) { Text("Share", style = MaterialTheme.typography.bodyLarge, color = FGTColors.TextPrimary) }
                    Row(Modifier.fillMaxWidth().clickable { peekImage = null }.padding(vertical = 14.dp)) { Text("Add to Album", style = MaterialTheme.typography.bodyLarge) }
                    Row(Modifier.fillMaxWidth().clickable { peekImage = null }.padding(vertical = 14.dp)) { Text("Favorite", style = MaterialTheme.typography.bodyLarge) }
                    Row(Modifier.fillMaxWidth().clickable { peekImage = null }.padding(vertical = 14.dp)) { Text("Smart Tags", style = MaterialTheme.typography.bodyLarge) }
                    Row(Modifier.fillMaxWidth().clickable { peekImage = null }.padding(vertical = 14.dp)) { Text("Delete", style = MaterialTheme.typography.bodyLarge, color = FGTColors.Error) }
                }
            }
        }
    }
}
