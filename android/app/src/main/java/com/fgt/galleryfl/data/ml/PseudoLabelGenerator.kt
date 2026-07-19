package com.fgt.galleryfl.data.ml

import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig

/** Creates one top-1 pseudo-label, matching centralized single-label training. */
class PseudoLabelGenerator(
    private val thresholdResolver: ThresholdResolver,
    private val classificationHead: ClassificationHead
) {
    suspend fun generatePseudoLabels(
        features: FloatArray,
        biasOffsets: FloatArray = FloatArray(TaxonomyConfig.NUM_CLASSES)
    ): FloatArray {
        // ThresholdResolver remains part of the constructor/API because its
        // confidence gates are used by gallery organization. Pseudo-training
        // itself must always receive a valid one-hot categorical target.
        @Suppress("UNUSED_VARIABLE")
        val resolver = thresholdResolver
        val probabilities = classificationHead.forward(features, biasOffsets)
        val bestClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        return FloatArray(TaxonomyConfig.NUM_CLASSES).also { it[bestClass] = 1f }
    }
}
