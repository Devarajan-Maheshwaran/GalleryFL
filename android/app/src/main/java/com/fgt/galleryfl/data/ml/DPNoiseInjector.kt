package com.fgt.galleryfl.data.ml

import kotlin.math.sqrt
import kotlin.math.ln
import kotlin.math.cos
import kotlin.math.PI
import kotlin.random.Random

/**
 * Calibrated Gaussian noise for (epsilon, delta)-DP applied to the AVERAGE of
 * per-example clipped gradients.
 *
 * The DP-SGD sensitivity of the average of n gradients, each clipped to a norm
 * of `sensitivity` (= max_grad_norm C), is C / n. Using the Gaussian
 * mechanism with the standard (epsilon, delta) bound:
 *
 *     sigma = C * sqrt(2 * ln(1.25 / delta)) / epsilon / n
 *
 * This matches the server's verified calibration (see server/verify_fl_loop.py)
 * and replaces the old approach that clipped the full delta and added a fixed
 * C-sized noise to it (which dominated the signal and diverged the model).
 *
 * The noise is added to the averaged clipped gradient, NOT to the submitted
 * weights/delta.
 */
class DPNoiseInjector(
    private val epsilon: Float = 1.0f,
    private val delta: Float = 1e-5f,
    private val sensitivity: Float = 1.0f,
    private val numSamples: Int = 1
) {
    private val sigma: Float = if (numSamples > 0 && epsilon > 0f) {
        (sensitivity * sqrt(2.0f * ln(1.25f / delta))) / epsilon / numSamples
    } else {
        0f
    }

    fun addNoise(gradients: List<FloatArray>): List<FloatArray> {
        if (sigma == 0f) return gradients
        return gradients.map { layer ->
            FloatArray(layer.size) { i -> layer[i] + gaussianSample() }
        }
    }

    private fun gaussianSample(): Float {
        val u1 = Random.nextFloat().coerceAtLeast(1e-10f)
        val u2 = Random.nextFloat()
        return (sigma * sqrt(-2.0f * ln(u1)) * cos(2.0f * PI.toFloat() * u2))
    }
}
