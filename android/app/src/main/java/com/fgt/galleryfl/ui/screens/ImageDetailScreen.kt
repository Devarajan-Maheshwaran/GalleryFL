package com.fgt.galleryfl.ui.screens

import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import coil3.compose.AsyncImage
import com.fgt.galleryfl.data.local.GalleryImage
import com.fgt.galleryfl.ui.theme.FGTColors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageDetailScreen(
    image: GalleryImage,
    onBack: () -> Unit,
    onDelete: (GalleryImage) -> Unit
) {
    val context = LocalContext.current
    var showInfo by remember { mutableStateOf(false) }

    Scaffold(
        containerColor = Color.Black,
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
                .padding(padding),
            contentAlignment = Alignment.Center
        ) {
            AsyncImage(
                model = image.uri,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )
            
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
