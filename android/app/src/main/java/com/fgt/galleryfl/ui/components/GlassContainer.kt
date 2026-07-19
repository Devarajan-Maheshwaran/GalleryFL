package com.fgt.galleryfl.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.matchParentSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.fgt.galleryfl.ui.theme.FGTColors

/**
 * Frosted "Liquid Glass" surface. A translucent rounded panel that lets the
 * colorful app background refract through it. On API 31+ [Modifier.blur] is a
 * real RenderEffect backdrop blur; on older APIs the translucent fill alone
 * provides the glass read (the blur call is a no-op there but harmless).
 */
@Composable
fun GlassContainer(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 24.dp,
    blurRadius: Dp = 24.dp,
    backgroundColor: Color = FGTColors.BgGlass,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(cornerRadius))
            .background(backgroundColor)
    ) {
        // Faint blurred backdrop layer behind the content for extra frost.
        Box(
            Modifier
                .matchParentSize()
                .blur(blurRadius)
                .background(backgroundColor)
        )
        content()
    }
}

/** Convenience: a thin glass scrim used behind modals / sheets. */
@Composable
fun GlassScrim(
    modifier: Modifier = Modifier,
    tint: Color = FGTColors.GlassOverlay,
    content: @Composable BoxScope.() -> Unit = {}
) {
    Box(
        modifier
            .fillMaxSize()
            .background(tint)
    ) { content() }
}
