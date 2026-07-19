package com.fgt.galleryfl.data.local

import com.fgt.galleryfl.data.ml.ClassificationHead
import com.fgt.galleryfl.data.ml.ThresholdResolver
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig

data class OrganizePreviewSummary(
    val folderPath: String,
    val imageCount: Int,
    val averageConfidence: Float,
    val thresholdUsed: Float,
    val localPersonalizationAffected: Boolean,
    val assignedImages: List<String> = emptyList()
)

data class ImagePrediction(
    val imageId: String,
    val features: FloatArray
)

class OrganizeGalleryUseCase(
    private val classificationHead: ClassificationHead,
    private val thresholdResolver: ThresholdResolver,
    private val feedbackStore: FeedbackStore
) {
    suspend fun previewFolderCreation(images: List<ImagePrediction>): List<OrganizePreviewSummary> {
        val numClasses = TaxonomyConfig.NUM_CLASSES
        val biasOffsets = feedbackStore.getBiasOffsets(numClasses)
        val thresholds = thresholdResolver.getThresholdsForAllClasses(numClasses)
        
        // Maps classIndex -> list of confidences for images that matched
        val classMatches = mutableMapOf<Int, MutableList<Float>>()
        val classImageIds = mutableMapOf<Int, MutableList<String>>()
        val personalizationAffected = mutableMapOf<Int, Boolean>()

        for (image in images) {
            val logitsWithoutBias = classificationHead.forward(image.features, null)
            val logitsWithBias = classificationHead.forward(image.features, biasOffsets)

            for (i in 0 until numClasses) {
                val score = logitsWithBias[i]
                val scoreWithoutBias = logitsWithoutBias[i]
                val threshold = thresholds[i]

                val isMatch = score >= threshold
                val isMatchWithoutBias = scoreWithoutBias >= threshold
                
                if (isMatch) {
                    classMatches.getOrPut(i) { mutableListOf() }.add(score)
                    classImageIds.getOrPut(i) { mutableListOf() }.add(image.imageId)
                }

                if (isMatch != isMatchWithoutBias) {
                    personalizationAffected[i] = true
                }
            }
        }

        val summaries = mutableListOf<OrganizePreviewSummary>()

        for ((classIndex, confidences) in classMatches) {
            val policy = TaxonomyConfig.getPolicyForClassIndex(classIndex) ?: continue
            val count = confidences.size
            
            if (count >= policy.minImagesToCreateFolder) {
                val avgConfidence = confidences.average().toFloat()
                val path = "${policy.category.name}/${policy.tag.name}"
                val affected = personalizationAffected[classIndex] ?: false
                
                summaries.add(
                    OrganizePreviewSummary(
                        folderPath = path,
                        imageCount = count,
                        averageConfidence = avgConfidence,
                        thresholdUsed = thresholds[classIndex],
                        localPersonalizationAffected = affected,
                        assignedImages = classImageIds[classIndex] ?: emptyList()
                    )
                )
            }
        }

        return summaries
    }
}
