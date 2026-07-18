package com.fgt.galleryfl.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Archive
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryAlbum
import com.fgt.galleryfl.ui.theme.FGTColors

@Composable
fun LibraryScreen(smartAlbums: List<GalleryAlbum>) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(FGTColors.BgBase)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        item {
            Text(
                "Library",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = FGTColors.TextPrimary
            )
        }
        
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                LibraryUtilityItem("Favorites", Icons.Default.Favorite, Color(0xFFEF4444))
                LibraryUtilityItem("Utilities", Icons.Default.Build, Color(0xFF6366F1))
                LibraryUtilityItem("Archive", Icons.Default.Archive, Color(0xFF10B981))
                LibraryUtilityItem("Trash", Icons.Default.Delete, Color(0xFFF59E0B))
            }
        }

        item {
            Text("Photos on device", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = FGTColors.TextPrimary)
        }

        items(smartAlbums) { album ->
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Box(
                    Modifier
                        .size(80.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(FGTColors.BgSurface)
                ) {
                    AsyncImage(
                        model = album.images.firstOrNull()?.uri,
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                }
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text(album.tagName, fontWeight = FontWeight.Bold, color = FGTColors.TextPrimary)
                    Text(
                        "Pictures/FGT/${album.folderPath.substringAfter("/")}",
                        style = MaterialTheme.typography.bodySmall,
                        color = FGTColors.TextSecondary
                    )
                }
            }
        }
    }
}

@Composable
fun LibraryUtilityItem(label: String, icon: ImageVector, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Surface(
            shape = RoundedCornerShape(16.dp),
            color = color.copy(alpha = 0.1f),
            modifier = Modifier.size(56.dp)
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(icon, contentDescription = label, tint = color, modifier = Modifier.size(28.dp))
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Text(label, style = MaterialTheme.typography.labelSmall, color = FGTColors.TextPrimary)
    }
}
