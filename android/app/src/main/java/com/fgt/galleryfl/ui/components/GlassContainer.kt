package com.fgt.galleryfl.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.fgt.galleryfl.ui.theme.FGTColors

/**
 * Frosted "Liquid Glass" surface — a real glass slab, not a translucent fill:
 *  - translucent body so the layered backdrop refracts through it,
 *  - a 1px hairline border (light catching the top-left edge),
 *  - an inner vertical frost tint (top light, bottom faint dark) so the pane
 *    reads as thick glass,
 *  - a top-edge specular sheen (light hitting the top of the slab),
 *  - a hairline bottom-right dark edge for depth,
 *  - a soft ambient + spot depth shadow.
 *
 * True backdrop blur needs API 31+ RenderEffect; below that we rely on the
 * translucency + edge treatment + textured backdrop, the standard Compose
 * glass pattern.
 */
@Composable
fun GlassContainer(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 24.dp,
    backgroundColor: Color = FGTColors.BgGlass,
    borderColor: Color = Color.White.copy(alpha = 0.6f),
    elevation: Dp = 20.dp,
    sheen: Boolean = true,
    contentAlignment: Alignment = Alignment.TopStart,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .shadow(
                elevation = elevation,
                shape = RoundedCornerShape(cornerRadius),
                ambientColor = Color.Black.copy(alpha = 0.16f),
                spotColor = Color.Black.copy(alpha = 0.12f)
            )
            .clip(RoundedCornerShape(cornerRadius))
            .background(backgroundColor)
            .border(1.dp, borderColor, RoundedCornerShape(cornerRadius)),
        contentAlignment = contentAlignment
    ) {
        // Inner frost tint: top light, bottom faint dark -> thick glass.
        Box(
            Modifier.matchParentSize().background(
                Brush.verticalGradient(
                    0f to Color.White.copy(alpha = 0.10f),
                    1f to Color.Black.copy(alpha = 0.04f)
                )
            )
        )
        if (sheen) {
            // Top-edge specular sheen.
            Box(
                Modifier.matchParentSize().background(
                    Brush.verticalGradient(
                        0f to Color.White.copy(alpha = 0.38f),
                        0.42f to Color.White.copy(alpha = 0.0f)
                    )
                )
            )
        }
        // Hairline bottom-right dark edge for depth.
        Box(
            Modifier.matchParentSize().background(
                Brush.verticalGradient(
                    0.7f to Color.Transparent,
                    1f to Color.Black.copy(alpha = 0.06f)
                )
            )
        )
        content()
    }
}

/** Convenience glass card: a [GlassContainer] with [ColumnScope] content. */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 24.dp,
    onClick: (() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit
) {
    GlassContainer(
        modifier = modifier.then(
            if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier
        ),
        cornerRadius = cornerRadius
    ) {
        Column(content = content)
    }
}

/** Convenience: a thin glass scrim used behind modals / sheets. */
@Composable
fun GlassScrim(
    modifier: Modifier = Modifier,
    tint: Color = FGTColors.GlassOverlay,
    content: @Composable BoxScope.() -> Unit = {}
) {
    Box(modifier.fillMaxSize().background(tint)) { content() }
}
