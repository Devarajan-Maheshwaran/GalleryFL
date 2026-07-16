package com.fgt.galleryfl.data.ml

import kotlin.math.ln
import kotlin.math.sqrt

data class TrainingResult(
    val updatedWeights: List<FloatArray>,
    val numSamples: Int,
    val localLoss: Float,
    val localAccuracy: Float
)

class LocalTrainer(
    private val head: ClassificationHead
) {
    // FedProx proximal term
    private val mu = 0.01f
    private val lr = 0.05f

    fun train(
        featuresList: List<FloatArray>,
        targetsList: List<FloatArray>,
        globalWeights: List<FloatArray>,
        epochs: Int = 3
    ): TrainingResult {
        // Set global weights initially
        head.setWeightsFlat(globalWeights)
        
        // Save global weights for FedProx
        val globalW1 = globalWeights[0].copyOf()
        val globalB1 = globalWeights[1].copyOf()
        val globalW2 = globalWeights[2].copyOf()
        val globalB2 = globalWeights[3].copyOf()

        val numSamples = featuresList.size
        var totalLoss = 0f
        var correct = 0

        for (epoch in 0 until epochs) {
            totalLoss = 0f
            correct = 0
            
            for (i in 0 until numSamples) {
                val features = featuresList[i]
                val targets = targetsList[i]

                val preds = head.forward(features)
                val grads = head.backward(targets)

                // Compute metrics
                var sampleLoss = 0f
                var isCorrect = true
                for (c in 0 until targets.size) {
                    val p = preds[c].coerceIn(1e-7f, 1f - 1e-7f)
                    sampleLoss -= targets[c] * ln(p) + (1 - targets[c]) * ln(1 - p)
                    val predBinary = if (p > 0.5f) 1f else 0f
                    if (predBinary != targets[c]) {
                        isCorrect = false
                    }
                }
                totalLoss += sampleLoss / targets.size
                if (isCorrect) correct++

                // SGD Update + FedProx
                // w = w - lr * (grad + mu * (w - w_global))
                
                // Update W1
                var idx = 0
                for (r in 0 until 960) {
                    for (c in 0 until 256) {
                        val w = head.w1[r][c]
                        val prox = mu * (w - globalW1[idx])
                        head.w1[r][c] = w - lr * (grads.dw1[r][c] + prox)
                        idx++
                    }
                }
                
                // Update b1
                for (j in 0 until 256) {
                    val w = head.b1[j]
                    val prox = mu * (w - globalB1[j])
                    head.b1[j] = w - lr * (grads.db1[j] + prox)
                }

                // Update W2
                idx = 0
                for (r in 0 until 256) {
                    for (c in 0 until 20) {
                        val w = head.w2[r][c]
                        val prox = mu * (w - globalW2[idx])
                        head.w2[r][c] = w - lr * (grads.dw2[r][c] + prox)
                        idx++
                    }
                }
                
                // Update b2
                for (j in 0 until 20) {
                    val w = head.b2[j]
                    val prox = mu * (w - globalB2[j])
                    head.b2[j] = w - lr * (grads.db2[j] + prox)
                }
            }
        }

        val accuracy = correct.toFloat() / numSamples
        val avgLoss = totalLoss / numSamples

        return TrainingResult(
            updatedWeights = head.getWeightsFlat(),
            numSamples = numSamples,
            localLoss = avgLoss,
            localAccuracy = accuracy
        )
    }
}
