package com.fgt.galleryfl.data.ml

import java.io.File
import androidx.exifinterface.media.ExifInterface

class PseudoLabelGenerator(
    private val thresholdResolver: ThresholdResolver,
    private val classificationHead: ClassificationHead
) {
    suspend fun generatePseudoLabels(
        features: FloatArray,
        biasOffsets: FloatArray = FloatArray(com.fgt.galleryfl.data.taxonomy.TaxonomyConfig.NUM_CLASSES)
    ): FloatArray {
        val numClasses = com.fgt.galleryfl.data.taxonomy.TaxonomyConfig.NUM_CLASSES
        val labels = FloatArray(numClasses)
        
        try {
            val thresholds = thresholdResolver.getThresholdsForAllClasses(numClasses)
            val preds = classificationHead.forward(features, biasOffsets)
            
            for (i in 0 until numClasses) {
                labels[i] = if (preds[i] >= thresholds[i]) 1.0f else 0.0f
            }
        } catch (e: Exception) {
            // Return all zeros on error
        }
        
        return labels
    }
}
