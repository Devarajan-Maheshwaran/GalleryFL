package com.fgt.galleryfl.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Shared GalleryFL palette — "Blush / Web3 soft".
 *
 * This is the SAME color logic used by the web coordinator console
 * (server/dashboard/styles.css) so the two surfaces read as one product:
 *   - near-white / light-pink surfaces
 *   - rose primary + lavender secondary ("fun but serious")
 *   - ink-plum text
 *   - soft, low-opacity shadows (see styles.css --shadow-*)
 *
 * Constant names are kept stable so every screen keeps compiling; only the
 * values changed to the light system.
 */
object FGTColors {
    val BgBase = Color(0xFFFDF6FB)        // near-white with a pink wash
    val BgSurface = Color(0xFFFFFFFF)     // pure card white
    val BgGlass = Color(0x26EF4D9B)       // translucent rose (light glass)

    val AccentPrimary = Color(0xFFEF4D9B) // rose
    val AccentSecondary = Color(0xFF8B7FE8) // lavender

    val TextPrimary = Color(0xFF2A1B2E)   // ink plum
    val TextSecondary = Color(0x9E2A1B2E) // ~62% ink plum

    val GlassOverlay = Color(0x33000000)
    val Error = Color(0xFFF0445E)
}
