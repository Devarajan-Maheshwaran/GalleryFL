package com.fgt.galleryfl.data.ml

import kotlin.math.ln

data class TrainingResult(
    val updatedWeights: List<FloatArray>,
    val numSamples: Int,
    val localLoss: Float,
    val localAccuracy: Float
)

class LocalTrainer(
    private val head: ClassificationHead,
    private val clipper: GradientClipper = GradientClipper(maxNorm = 1.0f),
    private val noiseInjector: DPNoiseInjector = DPNoiseInjector()
) {
    private val mu = 0.01f
    private val lr = 0.05f

    fun train(
        featuresList: List<FloatArray>,
        targetsList: List<FloatArray>,
        globalWeights: List<FloatArray>,
        epochs: Int = 3
    ): TrainingResult {
        head.setWeightsFlat(globalWeights)

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

                var sampleLoss = 0f
                var isCorrect = true
                for (c in 0 until targets.size) {
                    val p = preds[c].coerceIn(1e-7f, 1f - 1e-7f)
                    sampleLoss -= targets[c] * ln(p) + (1 - targets[c]) * ln(1 - p)
                    if ((if (p > 0.5f) 1f else 0f) != targets[c]) isCorrect = false
                }
                totalLoss += sampleLoss / targets.size
                if (isCorrect) correct++

                var idx = 0
                for (r in 0 until head.inputDim) {
                    for (c in 0 until 256) {
                        val w = head.w1[r][c]
                        val prox = mu * (w - globalW1[idx])
                        head.w1[r][c] = w - lr * (grads.dw1[r][c] + prox)
                        idx++
                    }
                }

                for (j in 0 until 256) {
                    val w = head.b1[j]
                    head.b1[j] = w - lr * (grads.db1[j] + mu * (w - globalB1[j]))
                }

                idx = 0
                for (r in 0 until 256) {
                    for (c in 0 until head.numClasses) {
                        val w = head.w2[r][c]
                        val prox = mu * (w - globalW2[idx])
                        head.w2[r][c] = w - lr * (grads.dw2[r][c] + prox)
                        idx++
                    }
                }

                for (j in 0 until head.numClasses) {
                    val w = head.b2[j]
                    head.b2[j] = w - lr * (grads.db2[j] + mu * (w - globalB2[j]))
                }
            }
        }

        val accuracy = correct.toFloat() / numSamples
        val avgLoss = totalLoss / numSamples

        var trainedWeights = head.getWeightsFlat()
        trainedWeights = clipper.clip(trainedWeights)
        trainedWeights = noiseInjector.addNoise(trainedWeights)

        return TrainingResult(
            updatedWeights = trainedWeights,
            numSamples = numSamples,
            localLoss = avgLoss,
            localAccuracy = accuracy
        )
    }
}
