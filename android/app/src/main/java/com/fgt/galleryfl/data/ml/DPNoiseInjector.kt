package com.fgt.galleryfl.data.ml

import java.security.SecureRandom
import kotlin.math.ln
import kotlin.math.sqrt

/** Gaussian mechanism for an average of fixed-bound per-example gradients. */
class DPNoiseInjector(
    epsilon: Float,
    delta: Float,
    sensitivity: Float,
    numSamples: Int,
    private val random: SecureRandom = SecureRandom()
) {
    init {
        require(epsilon > 0f) { "epsilon must be positive when DP is enabled" }
        require(delta > 0f && delta < 1f) { "delta must be in (0, 1)" }
        require(sensitivity > 0f) { "sensitivity must be positive" }
        require(numSamples > 0) { "numSamples must be positive" }
    }

    /** Standard deviation applied to every coordinate of the averaged gradient. */
    val sigma: Float = (
        sensitivity * sqrt(2.0f * ln(1.25f / delta)) / epsilon / numSamples
    )

    fun addNoise(gradients: List<FloatArray>): List<FloatArray> {
        return gradients.map { layer ->
            FloatArray(layer.size) { index ->
                layer[index] + random.nextGaussian().toFloat() * sigma
            }
        }
    }
}
