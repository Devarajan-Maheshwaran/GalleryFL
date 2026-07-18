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
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryAlbum
import com.fgt.galleryfl.ui.theme.FGTColors

@Composable
fun ExploreScreen(
    smartAlbums: List<GalleryAlbum>,
    isScanning: Boolean,
    onScanClick: () -> Unit,
    onAlbumClick: (GalleryAlbum) -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().background(FGTColors.BgBase)) {
        Text(
            "Explore",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = FGTColors.TextPrimary,
            modifier = Modifier.padding(start = 16.dp, top = 24.dp, bottom = 8.dp)
        )

        val categories = smartAlbums.groupBy { it.folderPath.substringBefore("/") }
        
        if (smartAlbums.isEmpty() && !isScanning) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Outlined.AutoAwesome, null, Modifier.size(64.dp), tint = FGTColors.AccentPrimary)
                    Spacer(Modifier.height(16.dp))
                    Text("Your AI categories will appear here", color = FGTColors.TextSecondary)
                    Spacer(Modifier.height(24.dp))
                    Button(onClick = onScanClick) { Text("Scan & Group") }
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
                            Surface(
                                shape = RoundedCornerShape(20.dp),
                                color = FGTColors.AccentPrimary.copy(alpha = 0.1f),
                                modifier = Modifier.clickable { /* Filter logic in main */ }
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
fun ImmersiveAlbumCard(album: GalleryAlbum, onClick: (GalleryAlbum) -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick(album) },
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = FGTColors.BgSurface)
    ) {
        Column {
            AsyncImage(
                model = album.images.firstOrNull()?.uri,
                contentDescription = null,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(24.dp)),
                contentScale = ContentScale.Crop
            )
            Column(modifier = Modifier.padding(16.dp)) {
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
