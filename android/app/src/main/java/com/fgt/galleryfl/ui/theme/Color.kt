package com.fgt.galleryfl.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Shared GalleryFL palette — "Monochrome / Liquid Glass".
 *
 * A neutral black & white system (no rose/lavender). Surfaces are near-white
 * with layered translucency; the single accent is a neutral graphite so the
 * frosted material reads as glass, not colour. This mirrors the web coordinator
 * console (server/dashboard/styles.css) so the two surfaces read as one product.
 *
 * Constant names are kept stable so every screen keeps compiling; only the
 * values changed to the monochrome system.
 */
object FGTColors {
    val BgBase = Color(0xFFF7F7F8)        // near-white
    val BgSurface = Color(0xFFFFFFFF)     // pure card white
    val BgSurface2 = Color(0xFFF1F1F3)    // light neutral surface (cards/rows)
    val OnPrimary = Color(0xFFFFFFFF)     // text on dark/neutral
    val BgGlass = Color(0x26FFFFFF)       // translucent white (light glass)

    // Monochrome accent — neutral graphite, not a hue.
    val AccentPrimary = Color(0xFF0A0A0A)   // near-black
    val AccentSecondary = Color(0xFF3A3A3C) // graphite

    val TextPrimary = Color(0xFF0A0A0A)   // ink
    val TextSecondary = Color(0x9E0A0A0A) // ~62% ink

    val GlassOverlay = Color(0x1F000000)
    val Error = Color(0xFFD6392B)

    /* Dark-mode matrix (Liquid Glass over true-black OLED). Accent stays neutral. */
    val BgBaseDark = Color(0xFF0A0A0B)
    val BgSurfaceDark = Color(0xFF161618)
    val BgGlassDark = Color(0x33FFFFFF)
    val TextPrimaryDark = Color(0xFFFFFFFF)
    val TextSecondaryDark = Color(0x9EFFFFFF)
    val GlassOverlayDark = Color(0x55000000)
}
