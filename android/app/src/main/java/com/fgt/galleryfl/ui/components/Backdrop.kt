package com.fgt.galleryfl.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

/**
 * Layered neutral backdrop that sits behind every screen.
 *
 * Glass only reads as glass when there is something beneath it to refract.
 * This draws a soft white -> neutral-grey vertical wash plus a few low-alpha
 * monochrome "orbs" so floating glass panels (nav, bars, cards, sheets) pick
 * up gentle depth instead of looking like flat boxes. Strictly black/white;
 * no hue.
 */
@Composable
fun AppBackdrop(modifier: Modifier = Modifier) {
    Box(modifier = modifier) {
        // Base vertical wash: pure white down to a soft neutral grey.
        Box(
            Modifier.fillMaxSize().background(
                Brush.verticalGradient(
                    0.0f to Color(0xFFFFFFFF),
                    1.0f to Color(0xFFF1F1F4)
                )
            )
        )
        // Soft monochrome light orb, upper-left.
        Box(
            Modifier.fillMaxSize().background(
                Brush.radialGradient(
                    colors = listOf(
                        Color.White.copy(alpha = 0.75f),
                        Color.White.copy(alpha = 0.0f)
                    ),
                    center = Offset(0.12f, 0.08f),
                    radius = 0.65f
                )
            )
        )
        // Deeper neutral orb, lower-right.
        Box(
            Modifier.fillMaxSize().background(
                Brush.radialGradient(
                    colors = listOf(
                        Color(0xFFE6E6EA).copy(alpha = 0.85f),
                        Color(0xFFE6E6EA).copy(alpha = 0.0f)
                    ),
                    center = Offset(0.92f, 0.96f),
                    radius = 0.8f
                )
            )
        )
        // Subtle counter-orb, upper-right, to break up symmetry.
        Box(
            Modifier.fillMaxSize().background(
                Brush.radialGradient(
                    colors = listOf(
                        Color(0xFFEDEDF1).copy(alpha = 0.7f),
                        Color(0xFFEDEDF1).copy(alpha = 0.0f)
                    ),
                    center = Offset(0.85f, 0.16f),
                    radius = 0.5f
                )
            )
        )
    }
}
