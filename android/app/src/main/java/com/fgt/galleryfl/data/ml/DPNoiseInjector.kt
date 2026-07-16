package com.fgt.galleryfl.data.ml

import kotlin.math.sqrt
import kotlin.math.ln
import kotlin.math.cos
import kotlin.math.PI
import kotlin.random.Random

class DPNoiseInjector(
    private val epsilon: Float = 1.0f,
    private val delta: Float = 1e-5f,
    private val sensitivity: Float = 1.0f
) {
    private val sigma: Float = (sensitivity * sqrt(2.0f * ln(1.25f / delta))) / epsilon

    fun addNoise(weights: List<FloatArray>): List<FloatArray> {
        return weights.map { layer ->
            FloatArray(layer.size) { i ->
                layer[i] + gaussianSample()
            }
        }
    }

    private fun gaussianSample(): Float {
        val u1 = Random.nextFloat().coerceAtLeast(1e-10f)
        val u2 = Random.nextFloat()
        return (sigma * sqrt(-2.0f * ln(u1)) * cos(2.0f * PI.toFloat() * u2))
    }
}
