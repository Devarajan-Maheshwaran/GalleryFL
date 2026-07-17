package com.fgt.galleryfl.data.ml

import kotlin.math.exp
import kotlin.math.max

class ClassificationHead(val numClasses: Int = 20) {

    val inputDim = 1024

    var w1: Array<FloatArray> = Array(inputDim) { FloatArray(256) }
    var b1: FloatArray = FloatArray(256)
    var w2: Array<FloatArray> = Array(256) { FloatArray(numClasses) }
    var b2: FloatArray = FloatArray(numClasses)

    private var lastInput: FloatArray = FloatArray(inputDim)
    private var z1: FloatArray = FloatArray(256)
    private var a1: FloatArray = FloatArray(256)
    private var z2: FloatArray = FloatArray(numClasses)
    private var a2: FloatArray = FloatArray(numClasses)

    fun forward(features: FloatArray): FloatArray {
        lastInput = features.copyOf()

        for (j in 0 until 256) {
            var sum = b1[j]
            for (i in 0 until inputDim) {
                sum += features[i] * w1[i][j]
            }
            z1[j] = sum
            a1[j] = max(0f, sum)
        }

        val output = FloatArray(numClasses)
        for (j in 0 until numClasses) {
            var sum = b2[j]
            for (i in 0 until 256) {
                sum += a1[i] * w2[i][j]
            }
            z2[j] = sum
            output[j] = 1f / (1f + exp(-sum))
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

    fun backward(targets: FloatArray): Gradients {
        val dz2 = FloatArray(numClasses)
        for (i in 0 until numClasses) {
            dz2[i] = a2[i] - targets[i]
        }

        val dw2 = Array(256) { FloatArray(numClasses) }
        val db2 = FloatArray(numClasses)
        for (j in 0 until numClasses) {
            db2[j] = dz2[j]
            for (i in 0 until 256) {
                dw2[i][j] = a1[i] * dz2[j]
            }
        }

        val da1 = FloatArray(256)
        for (i in 0 until 256) {
            var sum = 0f
            for (j in 0 until numClasses) {
                sum += w2[i][j] * dz2[j]
            }
            da1[i] = sum
        }

        val dz1 = FloatArray(256)
        for (i in 0 until 256) {
            dz1[i] = if (z1[i] > 0f) da1[i] else 0f
        }

        val dw1 = Array(inputDim) { FloatArray(256) }
        val db1 = FloatArray(256)
        for (j in 0 until 256) {
            db1[j] = dz1[j]
            for (i in 0 until inputDim) {
                dw1[i][j] = lastInput[i] * dz1[j]
            }
        }

        return Gradients(dw1, db1, dw2, db2)
    }

    fun getWeightsFlat(): List<FloatArray> {
        val w1Flat = FloatArray(inputDim * 256)
        var idx = 0
        for (i in 0 until inputDim) {
            for (j in 0 until 256) {
                w1Flat[idx++] = w1[i][j]
            }
        }

        val w2Flat = FloatArray(256 * numClasses)
        idx = 0
        for (i in 0 until 256) {
            for (j in 0 until numClasses) {
                w2Flat[idx++] = w2[i][j]
            }
        }

        return listOf(w1Flat, b1.copyOf(), w2Flat, b2.copyOf())
    }

    fun setWeightsFlat(weights: List<FloatArray>) {
        if (weights.size != 4) return

        var idx = 0
        for (i in 0 until inputDim) {
            for (j in 0 until 256) {
                w1[i][j] = weights[0][idx++]
            }
        }

        b1 = weights[1].copyOf()

        idx = 0
        for (i in 0 until 256) {
            for (j in 0 until numClasses) {
                w2[i][j] = weights[2][idx++]
            }
        }

        b2 = weights[3].copyOf()
    }
}
