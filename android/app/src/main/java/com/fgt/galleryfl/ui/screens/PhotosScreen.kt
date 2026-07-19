package com.fgt.galleryfl.ui.screens

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryImage
import com.fgt.galleryfl.ui.theme.FGTColors
import androidx.compose.foundation.shape.RoundedCornerShape

@Composable
fun PhotosScreen(
    photos: List<GalleryImage>,
    onImageClick: (GalleryImage) -> Unit
) {
    val gridState = rememberLazyGridState()
    
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
                columns = GridCells.Fixed(3),
                state = gridState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(1.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                items(photos, key = { it.id }) { photo ->
                    AsyncImage(
                        model = photo.uri,
                        contentDescription = null,
                        modifier = Modifier
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { onImageClick(photo) },
                        contentScale = ContentScale.Crop
                    )
                }
            }
        }
    }
}
