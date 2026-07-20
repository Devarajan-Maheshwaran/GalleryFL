package com.fgt.galleryfl.data.ml

import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig


data class PseudoLabel(
    val targets: FloatArray,
    val classIndex: Int,
    val confidence: Float,
)

/** High-confidence top-1 pseudo labels; uncertain images are not trained on. */
class PseudoLabelGenerator(
    private val thresholdResolver: ThresholdResolver,
    private val classificationHead: ClassificationHead,
    private val minimumConfidence: Float,
) {
    suspend fun generatePseudoLabel(
        features: FloatArray,
        biasOffsets: FloatArray = FloatArray(TaxonomyConfig.NUM_CLASSES),
    ): PseudoLabel? {
        val probabilities = classificationHead.forward(features, biasOffsets)
        val bestClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: return null
        val classGate = thresholdResolver.getThresholdForClass(bestClass)
        val requiredConfidence = maxOf(minimumConfidence, classGate)
        if (probabilities[bestClass] < requiredConfidence) return null
        val targets = FloatArray(TaxonomyConfig.NUM_CLASSES)
        targets[bestClass] = 1f
        return PseudoLabel(targets, bestClass, probabilities[bestClass])
    }
}
