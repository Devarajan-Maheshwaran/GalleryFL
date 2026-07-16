package com.fgt.galleryfl.data.ml

import kotlin.math.sqrt

class GradientClipper(private val maxNorm: Float = 1.0f) {

    fun clip(weights: List<FloatArray>): List<FloatArray> {
        val totalNorm = computeNorm(weights)
        if (totalNorm <= maxNorm) return weights

        val scale = maxNorm / totalNorm
        return weights.map { layer ->
            FloatArray(layer.size) { i -> layer[i] * scale }
        }
    }

    private fun computeNorm(weights: List<FloatArray>): Float {
        var sumSq = 0.0f
        for (layer in weights) {
            for (v in layer) {
                sumSq += v * v
            }
        }
        return sqrt(sumSq)
    }
}
