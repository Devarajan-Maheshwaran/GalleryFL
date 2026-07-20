package com.fgt.galleryfl.data.ml

import kotlin.math.ln
import kotlin.math.sqrt


data class TrainingResult(
    val updatedWeights: List<FloatArray>,
    val numSamples: Int,
    val localLoss: Float,
    val localAccuracy: Float
)

/**
 * Output-layer federated fine-tuning with FedProx and optional local DP.
 *
 * The first dense layer is frozen. This keeps the learned centralized feature
 * projection stable and makes example-level DP practical on a phone: only
 * w2/b2 (1,799 parameters) receive gradients/noise.
 *
 * When DP is enabled, every per-example output gradient (after its public
 * sample-quality weight) is clipped to the fixed public [maxGradNorm]. The
 * round epsilon/delta budget is divided across [epochs] with conservative basic
 * composition. No private adaptive quantile is used to choose the clip bound.
 */
class LocalTrainer(
    private val head: ClassificationHead,
    private val mu: Float = 0.01f,
) {
    fun train(
        featuresList: List<FloatArray>,
        targetsList: List<FloatArray>,
        sampleWeights: List<Float> = List(featuresList.size) { 1f },
        globalWeights: List<FloatArray>,
        epochs: Int = 1,
        lr: Float = 0.05f,
        dpEpsilon: Float = 0f,
        dpDelta: Float = 1e-5f,
        maxGradNorm: Float = 1.0f,
    ): TrainingResult {
        val n = featuresList.size
        require(n > 0) { "Local training requires at least one usable sample" }
        require(targetsList.size == n && sampleWeights.size == n) {
            "Features, targets, and sample weights must have equal size"
        }
        require(sampleWeights.all { it in 0f..1f }) { "Sample weights must be in [0,1]" }
        require(epochs > 0 && lr > 0f && maxGradNorm > 0f)
        head.setWeightsFlat(globalWeights)

        val globalW2 = globalWeights[2].copyOf()
        val globalB2 = globalWeights[3].copyOf()
        val useDP = dpEpsilon > 0f
        val epsilonPerStep = if (useDP) dpEpsilon / epochs else 0f
        val deltaPerStep = if (useDP) dpDelta / epochs else 0f

        repeat(epochs) {
            val accW2 = FloatArray(256 * head.numClasses)
            val accB2 = FloatArray(head.numClasses)

            for (index in 0 until n) {
                head.forward(featuresList[index])
                val gradient = head.backwardOutputOnly(targetsList[index], sampleWeights[index])
                val norm = outputGradNorm(gradient)
                val scale = if (norm > maxGradNorm) maxGradNorm / norm else 1f
                accumulate(accW2, gradient.dw2, scale)
                accumulate(accB2, gradient.db2, scale)
            }

            val inverseN = 1f / n
            for (index in accW2.indices) accW2[index] *= inverseN
            for (index in accB2.indices) accB2[index] *= inverseN

            val averaged = listOf(accW2, accB2)
            val noised = if (useDP) {
                DPNoiseInjector(
                    epsilon = epsilonPerStep,
                    delta = deltaPerStep,
                    sensitivity = maxGradNorm,
                    numSamples = n,
                ).addNoise(averaged)
            } else {
                averaged
            }
            val gradientW2 = noised[0]
            val gradientB2 = noised[1]

            var flatIndex = 0
            for (row in 0 until 256) {
                for (column in 0 until head.numClasses) {
                    val value = head.w2[row][column]
                    head.w2[row][column] = value - lr * (
                        gradientW2[flatIndex] + mu * (value - globalW2[flatIndex])
                    )
                    flatIndex++
                }
            }
            for (column in 0 until head.numClasses) {
                val value = head.b2[column]
                head.b2[column] = value - lr * (
                    gradientB2[column] + mu * (value - globalB2[column])
                )
            }
        }

        var weightedLoss = 0f
        var weightedCorrect = 0f
        var totalWeight = 0f
        for (index in 0 until n) {
            val probabilities = head.forward(featuresList[index])
            val targetClass = targetsList[index].indices.maxByOrNull { targetsList[index][it] } ?: 0
            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val weight = sampleWeights[index]
            weightedLoss -= weight * ln(probabilities[targetClass].coerceIn(1e-7f, 1f))
            if (predictedClass == targetClass) weightedCorrect += weight
            totalWeight += weight
        }
        val denominator = totalWeight.coerceAtLeast(1e-7f)
        return TrainingResult(
            updatedWeights = head.getWeightsFlat(),
            numSamples = n,
            localLoss = weightedLoss / denominator,
            localAccuracy = weightedCorrect / denominator,
        )
    }

    private fun outputGradNorm(gradient: ClassificationHead.OutputGradients): Float {
        var sum = 0.0
        for (row in gradient.dw2) for (value in row) sum += value * value
        for (value in gradient.db2) sum += value * value
        return sqrt(sum).toFloat().coerceAtLeast(1e-12f)
    }

    private fun accumulate(target: FloatArray, source: FloatArray, scale: Float) {
        for (index in target.indices) target[index] += source[index] * scale
    }

    private fun accumulate(target: FloatArray, source: Array<FloatArray>, scale: Float) {
        var index = 0
        for (row in source) for (value in row) target[index++] += value * scale
    }
}
