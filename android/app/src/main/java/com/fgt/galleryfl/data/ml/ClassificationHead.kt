package com.fgt.galleryfl.data.ml

import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import kotlin.math.exp
import kotlin.math.max

/** Seven-parent softmax head used for inference and categorical local training. */
class ClassificationHead(val numClasses: Int = TaxonomyConfig.NUM_CLASSES) {

    val inputDim = 1024

    init {
        require(numClasses == TaxonomyConfig.NUM_CLASSES) {
            "GalleryFL head requires ${TaxonomyConfig.NUM_CLASSES} parent classes, got $numClasses"
        }
    }

    var w1: Array<FloatArray> = Array(inputDim) { FloatArray(256) }
    var b1: FloatArray = FloatArray(256)
    var w2: Array<FloatArray> = Array(256) { FloatArray(numClasses) }
    var b2: FloatArray = FloatArray(numClasses)

    private var lastInput: FloatArray = FloatArray(inputDim)
    private var z1: FloatArray = FloatArray(256)
    private var a1: FloatArray = FloatArray(256)
    private var a2: FloatArray = FloatArray(numClasses)

    fun forward(features: FloatArray, biasOffsets: FloatArray? = null): FloatArray {
        require(features.size == inputDim) { "Expected $inputDim features, got ${features.size}" }
        if (biasOffsets != null) {
            require(biasOffsets.size == numClasses) {
                "Expected $numClasses bias offsets, got ${biasOffsets.size}"
            }
        }
        lastInput = features.copyOf()

        for (j in 0 until 256) {
            var sum = b1[j]
            for (i in 0 until inputDim) sum += features[i] * w1[i][j]
            z1[j] = sum
            a1[j] = max(0f, sum)
        }

        val logits = FloatArray(numClasses)
        var maxLogit = Float.NEGATIVE_INFINITY
        for (j in 0 until numClasses) {
            var sum = b2[j] + (biasOffsets?.get(j) ?: 0f)
            for (i in 0 until 256) sum += a1[i] * w2[i][j]
            logits[j] = sum
            if (sum > maxLogit) maxLogit = sum
        }

        var denominator = 0.0
        for (j in 0 until numClasses) {
            val value = exp((logits[j] - maxLogit).toDouble()).toFloat()
            a2[j] = value
            denominator += value
        }
        val output = FloatArray(numClasses)
        for (j in 0 until numClasses) {
            output[j] = (a2[j] / denominator).toFloat()
            a2[j] = output[j]
        }
        return output
    }

    data class Gradients(
        val dw1: Array<FloatArray>,
        val db1: FloatArray,
        val dw2: Array<FloatArray>,
        val db2: FloatArray
    )

    data class OutputGradients(
        val dw2: Array<FloatArray>,
        val db2: FloatArray
    )

    /**
     * Output-layer-only gradient used by FL. The centralized hidden layer stays
     * frozen, reducing client compute and DP noise from ~264k to 1,799 trained
     * parameters while preserving the four-tensor wire format.
     */
    fun backwardOutputOnly(targets: FloatArray, sampleWeight: Float = 1f): OutputGradients {
        require(targets.size == numClasses) { "Expected $numClasses targets" }
        require(sampleWeight in 0f..1f) { "sampleWeight must be in [0,1]" }
        val dz2 = FloatArray(numClasses) { index -> (a2[index] - targets[index]) * sampleWeight }
        val dw2 = Array(256) { FloatArray(numClasses) }
        for (j in 0 until numClasses) {
            for (i in 0 until 256) dw2[i][j] = a1[i] * dz2[j]
        }
        return OutputGradients(dw2, dz2)
    }

    /** Gradient of softmax categorical cross-entropy for a one-hot target. */
    fun backward(targets: FloatArray): Gradients {
        require(targets.size == numClasses) { "Expected $numClasses targets" }
        val dz2 = FloatArray(numClasses) { index -> a2[index] - targets[index] }

        val dw2 = Array(256) { FloatArray(numClasses) }
        val db2 = dz2.copyOf()
        for (j in 0 until numClasses) {
            for (i in 0 until 256) dw2[i][j] = a1[i] * dz2[j]
        }

        val da1 = FloatArray(256)
        for (i in 0 until 256) {
            var sum = 0f
            for (j in 0 until numClasses) sum += w2[i][j] * dz2[j]
            da1[i] = sum
        }

        val dz1 = FloatArray(256) { index -> if (z1[index] > 0f) da1[index] else 0f }
        val dw1 = Array(inputDim) { FloatArray(256) }
        val db1 = dz1.copyOf()
        for (j in 0 until 256) {
            for (i in 0 until inputDim) dw1[i][j] = lastInput[i] * dz1[j]
        }
        return Gradients(dw1, db1, dw2, db2)
    }

    fun getWeightsFlat(): List<FloatArray> {
        val w1Flat = FloatArray(inputDim * 256)
        var index = 0
        for (row in w1) for (value in row) w1Flat[index++] = value
        val w2Flat = FloatArray(256 * numClasses)
        index = 0
        for (row in w2) for (value in row) w2Flat[index++] = value
        return listOf(w1Flat, b1.copyOf(), w2Flat, b2.copyOf())
    }

    fun setWeightsFlat(weights: List<FloatArray>) {
        require(weights.size == 4) { "Expected four head tensors" }
        require(weights[0].size == inputDim * 256) { "Invalid w1 length" }
        require(weights[1].size == 256) { "Invalid b1 length" }
        require(weights[2].size == 256 * numClasses) { "Invalid w2 length" }
        require(weights[3].size == numClasses) { "Invalid b2 length" }

        var index = 0
        for (i in 0 until inputDim) {
            for (j in 0 until 256) w1[i][j] = weights[0][index++]
        }
        b1 = weights[1].copyOf()
        index = 0
        for (i in 0 until 256) {
            for (j in 0 until numClasses) w2[i][j] = weights[2][index++]
        }
        b2 = weights[3].copyOf()
    }
}
