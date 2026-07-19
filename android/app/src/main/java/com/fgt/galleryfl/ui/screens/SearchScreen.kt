package com.fgt.galleryfl.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryAlbum
import com.fgt.galleryfl.data.local.GalleryImage
import com.fgt.galleryfl.ui.theme.FGTColors

/**
 * Liquid Glass "Search" destination. Semantic query over the on-device taxonomy:
 * tapping a suggested smart-tag filters to that auto-grouped album; free-text
 * falls back to the full timeline. (True per-photo predicted-tag matching is
 * wired once the active model weights are synced — see ExploreScreen scanning.)
 */
@Composable
fun SearchScreen(
    photos: List<GalleryImage>,
    smartAlbums: List<GalleryAlbum> = emptyList(),
    onImageClick: (GalleryImage) -> Unit
) {
    var query by remember { mutableStateOf("") }

    val suggestions = smartAlbums.map { it.tagName }.distinct()
    val matchedAlbum = smartAlbums.firstOrNull { it.tagName.equals(query.trim(), ignoreCase = true) }
    val displayed = if (matchedAlbum != null) matchedAlbum.images else photos

    Column(modifier = Modifier.fillMaxSize()) {
        Text(
            "Search",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = FGTColors.TextPrimary,
            modifier = Modifier.padding(start = 16.dp, top = 24.dp, bottom = 8.dp)
        )

        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            placeholder = { Text("Search tags or names") },
            leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null, tint = FGTColors.TextSecondary) },
            shape = RoundedCornerShape(24.dp),
            colors = OutlinedTextFieldDefaults.colors(
                unfocusedContainerColor = FGTColors.BgGlass,
                focusedContainerColor = FGTColors.BgGlass,
                unfocusedBorderColor = Color.Transparent,
                focusedBorderColor = FGTColors.AccentPrimary
            ),
            singleLine = true
        )

        if (suggestions.isNotEmpty()) {
            LazyRow(
                contentPadding = PaddingValues(horizontal = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(bottom = 8.dp)
            ) {
                items(suggestions) { tag ->
                    val selected = tag.equals(query.trim(), ignoreCase = true)
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = if (selected) FGTColors.AccentPrimary else FGTColors.AccentPrimary.copy(alpha = 0.12f),
                        modifier = Modifier.clickable { query = if (selected) "" else tag }
                    ) {
                        Text(
                            tag.replace("_", " "),
                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                            color = if (selected) FGTColors.OnPrimary else FGTColors.AccentPrimary,
                            fontWeight = FontWeight.Medium,
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            }
        }

        if (displayed.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No matches", color = FGTColors.TextSecondary)
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(3),
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(1.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                items(displayed, key = { it.id }) { photo ->
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
