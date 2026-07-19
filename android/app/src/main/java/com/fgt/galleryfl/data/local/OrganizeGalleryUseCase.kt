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

data class ImagePrediction(val imageId: String, val features: FloatArray)

/** Assign each image to at most one parent folder using softmax argmax. */
class OrganizeGalleryUseCase(
    private val classificationHead: ClassificationHead,
    private val thresholdResolver: ThresholdResolver,
    private val feedbackStore: FeedbackStore
) {
    suspend fun previewFolderCreation(images: List<ImagePrediction>): List<OrganizePreviewSummary> {
        val numClasses = TaxonomyConfig.NUM_CLASSES
        val biasOffsets = feedbackStore.getBiasOffsets(numClasses)
        val thresholds = thresholdResolver.getThresholdsForAllClasses(numClasses)
        val classMatches = mutableMapOf<Int, MutableList<Float>>()
        val classImageIds = mutableMapOf<Int, MutableList<String>>()
        val personalizationAffected = mutableMapOf<Int, Boolean>()

        for (image in images) {
            val withoutBias = classificationHead.forward(image.features, null)
            val withBias = classificationHead.forward(image.features, biasOffsets)
            val bestClass = withBias.indices.maxByOrNull { withBias[it] } ?: continue
            if (withBias[bestClass] >= thresholds[bestClass]) {
                classMatches.getOrPut(bestClass) { mutableListOf() }.add(withBias[bestClass])
                classImageIds.getOrPut(bestClass) { mutableListOf() }.add(image.imageId)
            }
            val bestWithoutBias = withoutBias.indices.maxByOrNull { withoutBias[it] } ?: bestClass
            if (bestWithoutBias != bestClass) personalizationAffected[bestClass] = true
        }

        return classMatches.mapNotNull { (classIndex, confidences) ->
            val policy = TaxonomyConfig.getPolicyForClassIndex(classIndex) ?: return@mapNotNull null
            if (confidences.size < policy.minImagesToCreateFolder) return@mapNotNull null
            OrganizePreviewSummary(
                folderPath = "${policy.category.name}/${policy.tag.name}",
                imageCount = confidences.size,
                averageConfidence = confidences.average().toFloat(),
                thresholdUsed = thresholds[classIndex],
                localPersonalizationAffected = personalizationAffected[classIndex] ?: false,
                assignedImages = classImageIds[classIndex] ?: emptyList()
            )
        }
    }
}
