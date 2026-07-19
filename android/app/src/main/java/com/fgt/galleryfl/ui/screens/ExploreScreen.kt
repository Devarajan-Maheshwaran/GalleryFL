package com.fgt.galleryfl.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Undo
import androidx.compose.material3.*
import androidx.compose.animation.core.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryAlbum
import com.fgt.galleryfl.ui.components.*
import com.fgt.galleryfl.ui.theme.FGTColors

@Composable
fun ExploreScreen(
    smartAlbums: List<GalleryAlbum>,
    isScanning: Boolean,
    onScanClick: () -> Unit,
    onAlbumClick: (GalleryAlbum) -> Unit,
    onOrganizeClick: () -> Unit = {},
    onUndoClick: () -> Unit = {},
    canUndo: Boolean = false,
    modelReady: Boolean = true
) {
    Column(modifier = Modifier.fillMaxSize()) {
        Text(
            "Explore",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = FGTColors.TextPrimary,
            modifier = Modifier.padding(start = 16.dp, top = 24.dp, bottom = 8.dp)
        )

        FederatedIntelligenceIndicator(isActive = isScanning || modelReady)

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, end = 16.dp, bottom = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(onClick = onOrganizeClick, enabled = modelReady && !isScanning) {
                Icon(Icons.Outlined.AutoAwesome, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Organize")
            }
            if (canUndo) {
                OutlinedButton(onClick = onUndoClick, enabled = !isScanning) {
                    Icon(Icons.Outlined.Undo, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Undo")
                }
            }
        }

        val categories = smartAlbums.groupBy { it.folderPath.substringBefore("/") }
        
        if (smartAlbums.isEmpty() && !isScanning) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                GlassContainer(
                    modifier = Modifier.widthIn(max = 320.dp).padding(24.dp),
                    cornerRadius = 28.dp
                ) {
                    Column(
                        Modifier.padding(28.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(Icons.Outlined.AutoAwesome, null, Modifier.size(64.dp), tint = FGTColors.AccentPrimary)
                        Spacer(Modifier.height(16.dp))
                        Text("Your AI categories will appear here", color = FGTColors.TextSecondary, textAlign = TextAlign.Center)
                        Spacer(Modifier.height(24.dp))
                        Button(onClick = onScanClick) { Text("Scan & Group") }
                    }
                }
            }
        } else if (isScanning) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = FGTColors.AccentPrimary)
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Category Highlights
                item {
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        items(categories.keys.toList()) { category ->
                            GlassContainer(
                                modifier = Modifier.clickable { /* Filter logic in main */ },
                                cornerRadius = 20.dp,
                                backgroundColor = FGTColors.BgGlass,
                                elevation = 10.dp,
                                sheen = false
                            ) {
                                Text(
                                    category,
                                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                                    color = FGTColors.AccentPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }

                categories.forEach { (cat, albums) ->
                    item {
                        Text(cat, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = FGTColors.TextPrimary)
                        Spacer(Modifier.height(12.dp))
                        LazyVerticalGrid(
                            columns = GridCells.Fixed(2),
                            modifier = Modifier.heightIn(max = 2000.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                            userScrollEnabled = false
                        ) {
                            items(albums) { album ->
                                ImmersiveAlbumCard(album, onAlbumClick)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun FederatedIntelligenceIndicator(isActive: Boolean) {
    val transition = rememberInfiniteTransition(label = "intel")
    val pulse by transition.animateFloat(
        initialValue = 0.55f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1200), RepeatMode.Reverse),
        label = "pulse"
    )
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    ) {
        Icon(
            Icons.Outlined.AutoAwesome,
            contentDescription = null,
            tint = if (isActive) FGTColors.AccentPrimary else FGTColors.TextSecondary,
            modifier = Modifier.size(20.dp).alpha(if (isActive) pulse else 1f)
        )
        Spacer(Modifier.width(10.dp))
        Text(
            if (isActive) "Privacy-Engineered Sync Active" else "Local intelligence ready",
            style = MaterialTheme.typography.labelMedium,
            color = FGTColors.TextSecondary
        )
    }
}

@Composable
fun ImmersiveAlbumCard(album: GalleryAlbum, onClick: (GalleryAlbum) -> Unit) {
    GlassContainer(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick(album) },
        cornerRadius = 24.dp
    ) {
        Box {
            AsyncImage(
                model = album.images.firstOrNull()?.uri,
                contentDescription = null,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(24.dp)),
                contentScale = ContentScale.Crop
            )
            // Frosted label strip over the photo (glass over imagery).
            Box(
                Modifier
                    .align(Alignment.BottomStart)
                    .fillMaxWidth()
                    .background(FGTColors.BgGlass)
                    .padding(14.dp)
            ) {
                Column {
                    Text(
                        album.tagName,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = FGTColors.TextPrimary
                    )
                    Text(
                        "${album.images.size} items",
                        style = MaterialTheme.typography.bodySmall,
                        color = FGTColors.TextSecondary
                    )
                }
            }
        }
    }
}
