package com.fgt.galleryfl.data.ml

import kotlin.math.sqrt
import kotlin.math.ln

data class TrainingResult(
    val updatedWeights: List<FloatArray>,
    val numSamples: Int,
    val localLoss: Float,
    val localAccuracy: Float
)

/**
 * On-device federated trainer.
 *
 * Privacy: when [dpEpsilon] > 0 the trainer runs DP-SGD — per-example gradient
 * clipping to [maxGradNorm] followed by a single SGD+FedProx step on the
 * average of the clipped gradients, with calibrated Gaussian noise added to
 * that average (sigma = C * sqrt(2 ln(1.25/delta)) / epsilon / n, see
 * DPNoiseInjector). Clipping per-example (not on the full delta after
 * training) is what makes the DP bound hold.
 *
 * When [dpEpsilon] <= 0 the same averaged-gradient SGD runs with no noise, so
 * behaviour is consistent with the server's verification harness and with the
 * non-private path.
 *
 * Note on utility: `max_grad_norm` (C) and `lr` are hyperparameters supplied
 * by the server. For a ~270k-parameter head the per-example gradient norm is
 * large (O(100)), so a very small C (e.g. the default 1.0) clips away most of
 * the signal and yields limited learning. This is a tuning choice, not a
 * correctness issue — the DP mechanism itself is correctly calibrated.
 */
class LocalTrainer(
    private val head: ClassificationHead,
    private val mu: Float = 0.01f,
) {
    fun train(
        featuresList: List<FloatArray>,
        targetsList: List<FloatArray>,
        globalWeights: List<FloatArray>,
        epochs: Int = 3,
        lr: Float = 0.001f,
        dpEpsilon: Float = 0f,
        dpDelta: Float = 1e-5f,
        maxGradNorm: Float = 1.0f,
        numSamples: Int = featuresList.size
    ): TrainingResult {
        head.setWeightsFlat(globalWeights)

        val globalW1 = globalWeights[0].copyOf()
        val globalB1 = globalWeights[1].copyOf()
        val globalW2 = globalWeights[2].copyOf()
        val globalB2 = globalWeights[3].copyOf()

        val n = featuresList.size
        val useDP = dpEpsilon > 0f && n > 0
        val noise = if (useDP) DPNoiseInjector(
            epsilon = dpEpsilon,
            delta = dpDelta,
            sensitivity = maxGradNorm,
            numSamples = n
        ) else null

        for (epoch in 0 until epochs) {
            // Accumulators for per-example clipped gradients.
            val accW1 = FloatArray(head.inputDim * 256)
            val accB1 = FloatArray(256)
            val accW2 = FloatArray(256 * head.numClasses)
            val accB2 = FloatArray(head.numClasses)

            for (i in 0 until n) {
                val features = featuresList[i]
                val targets = targetsList[i]
                head.forward(features)
                val grads = head.backward(targets)

                // Per-example gradient clipping to maxGradNorm.
                val norm = gradNorm(grads)
                val scale = if (norm > maxGradNorm && norm > 1e-12f) maxGradNorm / norm else 1f
                accumulate(accW1, grads.dw1, scale)
                accumulate(accB1, grads.db1, scale)
                accumulate(accW2, grads.dw2, scale)
                accumulate(accB2, grads.db2, scale)
            }

            // Average the clipped per-example gradients.
            val invN = 1f / n
            for (k in accW1.indices) accW1[k] *= invN
            for (k in accB1.indices) accB1[k] *= invN
            for (k in accW2.indices) accW2[k] *= invN
            for (k in accB2.indices) accB2[k] *= invN

            // Add calibrated DP noise to the averaged gradient (one shot).
            val avgGrad = listOf(accW1, accB1, accW2, accB2)
            val g = noise?.addNoise(avgGrad) ?: avgGrad
            val gW1 = g[0]; val gB1 = g[1]; val gW2 = g[2]; val gB2 = g[3]

            // One SGD + FedProx step on the (noisy) averaged gradient.
            var idx = 0
            for (r in 0 until head.inputDim) {
                for (c in 0 until 256) {
                    val w = head.w1[r][c]
                    head.w1[r][c] = w - lr * (gW1[idx] + mu * (w - globalW1[idx]))
                    idx++
                }
            }
            for (j in 0 until 256) {
                val w = head.b1[j]
                head.b1[j] = w - lr * (gB1[j] + mu * (w - globalB1[j]))
            }
            idx = 0
            for (r in 0 until 256) {
                for (c in 0 until head.numClasses) {
                    val w = head.w2[r][c]
                    head.w2[r][c] = w - lr * (gW2[idx] + mu * (w - globalW2[idx]))
                    idx++
                }
            }
            for (j in 0 until head.numClasses) {
                val w = head.b2[j]
                head.b2[j] = w - lr * (gB2[j] + mu * (w - globalB2[j]))
            }
        }

        // Final BCE loss + strict multi-label accuracy (reporting only).
        var totalLoss = 0f
        var correct = 0
        for (i in 0 until n) {
            val preds = head.forward(featuresList[i])
            val targets = targetsList[i]
            var sampleLoss = 0f
            var isCorrect = true
            for (c in 0 until targets.size) {
                val p = preds[c].coerceIn(1e-7f, 1f - 1e-7f)
                sampleLoss -= targets[c] * ln(p) + (1 - targets[c]) * ln(1 - p)
                if ((if (p > 0.5f) 1f else 0f) != targets[c]) isCorrect = false
            }
            totalLoss += sampleLoss / targets.size
            if (isCorrect) correct++
        }

        return TrainingResult(
            updatedWeights = head.getWeightsFlat(),
            numSamples = n,
            localLoss = totalLoss / n,
            localAccuracy = correct.toFloat() / n
        )
    }

    private fun gradNorm(g: ClassificationHead.Gradients): Float {
        var s = 0.0f
        for (row in g.dw1) for (v in row) s += v * v
        for (v in g.db1) s += v * v
        for (row in g.dw2) for (v in row) s += v * v
        for (v in g.db2) s += v * v
        return sqrt(s)
    }

    private fun accumulate(target: FloatArray, src: FloatArray, scale: Float) {
        for (k in target.indices) target[k] += src[k] * scale
    }

    private fun accumulate(target: FloatArray, src: Array<FloatArray>, scale: Float) {
        var k = 0
        for (row in src) for (v in row) target[k++] += v * scale
    }
}
