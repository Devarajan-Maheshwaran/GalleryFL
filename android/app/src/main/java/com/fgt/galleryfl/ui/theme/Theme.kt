package com.fgt.galleryfl.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat

/**
 * Liquid Glass edge-to-edge setup: the app draws underneath the status and
 * navigation bars (translucent), and bar-icon color tracks the theme.
 */
fun enableEdgeToEdge(activity: Activity, darkTheme: Boolean) {
    val window = activity.window
    WindowCompat.setDecorFitsSystemWindows(window, false)
    window.statusBarColor = Color.Transparent.toArgb()
    window.navigationBarColor = Color.Transparent.toArgb()
    val controller = WindowCompat.getInsetsController(window, window.decorView)
    controller.isAppearanceLightStatusBars = !darkTheme
    controller.isAppearanceLightNavigationBars = !darkTheme
}

private val LightColorScheme = lightColorScheme(
    primary = FGTColors.AccentPrimary,
    secondary = FGTColors.AccentSecondary,
    tertiary = FGTColors.AccentSecondary,
    background = FGTColors.BgBase,
    surface = FGTColors.BgSurface,
    surfaceVariant = FGTColors.BgSurface2,
    onPrimary = FGTColors.OnPrimary,
    onBackground = FGTColors.TextPrimary,
    onSurface = FGTColors.TextPrimary,
    onSurfaceVariant = FGTColors.TextSecondary,
    error = FGTColors.Error
)

private val DarkColorScheme = darkColorScheme(
    primary = FGTColors.AccentPrimary,
    secondary = FGTColors.AccentSecondary,
    tertiary = FGTColors.AccentSecondary,
    background = FGTColors.BgBaseDark,
    surface = FGTColors.BgSurfaceDark,
    surfaceVariant = FGTColors.BgSurfaceDark,
    onPrimary = FGTColors.OnPrimary,
    onBackground = FGTColors.TextPrimaryDark,
    onSurface = FGTColors.TextPrimaryDark,
    onSurfaceVariant = FGTColors.TextSecondaryDark,
    error = FGTColors.Error
)

@Composable
fun GalleryFLTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val activity = view.context as? Activity
            if (activity != null) enableEdgeToEdge(activity, darkTheme)
        }
    }
    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
