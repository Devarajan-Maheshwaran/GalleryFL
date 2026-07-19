package com.fgt.galleryfl.data.ml

import com.fgt.galleryfl.data.local.FeedbackStore
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig

class ThresholdResolver(private val feedbackStore: FeedbackStore) {

    suspend fun getThresholdForClass(classIndex: Int, fallbackThreshold: Float = 0.5f): Float {
        // 2. taxonomy default threshold
        val defaultThreshold = if (classIndex >= 0 && classIndex < TaxonomyConfig.modelTags.size) {
            TaxonomyConfig.modelTags[classIndex].defaultThreshold
        } else {
            // 3. safe fallback threshold
            fallbackThreshold
        }

        // 1. local threshold override
        return feedbackStore.getThresholdOverride(classIndex, defaultThreshold)
    }

    suspend fun getThresholdsForAllClasses(numClasses: Int): FloatArray {
        val thresholds = FloatArray(numClasses)
        for (i in 0 until numClasses) {
            thresholds[i] = getThresholdForClass(i)
        }
        return thresholds
    }
}
