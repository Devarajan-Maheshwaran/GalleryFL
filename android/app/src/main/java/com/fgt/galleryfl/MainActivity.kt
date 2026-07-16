package com.fgt.galleryfl

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.fgt.galleryfl.ui.theme.FGTColors
import com.fgt.galleryfl.ui.components.NeuSurface

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                MainScreen()
            }
        }
    }
}

@Composable
fun MainScreen() {
    var statusText by remember { mutableStateOf("Ready to connect") }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(FGTColors.BgBase)
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Federated Gallery Tags",
                style = MaterialTheme.typography.headlineMedium,
                color = FGTColors.TextPrimary
            )
            
            Spacer(modifier = Modifier.height(32.dp))
            
            NeuSurface {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "FL Status",
                        style = MaterialTheme.typography.titleMedium,
                        color = FGTColors.TextSecondary
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.bodyMedium,
                        color = FGTColors.AccentPrimary
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(32.dp))
            
            Button(
                onClick = { statusText = "Connecting to Server..." },
                colors = ButtonDefaults.buttonColors(containerColor = FGTColors.AccentPrimary)
            ) {
                Text("Join Training Round")
            }
        }
    }
}
