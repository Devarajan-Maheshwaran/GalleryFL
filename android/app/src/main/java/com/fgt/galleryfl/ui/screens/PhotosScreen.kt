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
