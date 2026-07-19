package com.fgt.galleryfl

import com.fgt.galleryfl.data.local.ClassStats
import com.fgt.galleryfl.data.local.FeedbackStore
import com.fgt.galleryfl.data.local.ImagePrediction
import com.fgt.galleryfl.data.local.OrganizeGalleryUseCase
import com.fgt.galleryfl.data.ml.ClassificationHead
import com.fgt.galleryfl.data.ml.ThresholdResolver
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class FakeFeedbackStore : FeedbackStore {
    private val offsets = FloatArray(TaxonomyConfig.NUM_CLASSES)
    override suspend fun recordPredicted(classIndex: Int) {}
    override suspend fun recordConfirmed(classIndex: Int) {}
    override suspend fun recordRejected(classIndex: Int) {}
    override suspend fun recordDemand(classIndex: Int) {}
    override suspend fun getPerClassStats(classIndex: Int) = ClassStats(0, 0, 0)
    override suspend fun getThresholdOverride(classIndex: Int, defaultThreshold: Float) = defaultThreshold
    override suspend fun getBiasOffsets(numClasses: Int) = offsets.copyOf()
}

class OrganizeGalleryUseCaseTest {

    @Test
    fun createsExactlyOneParentFolderForTop1Class() = runBlocking {
        val store = FakeFeedbackStore()
        val head = ClassificationHead()
        val natureIndex = TaxonomyConfig.modelTags.indexOfFirst { it.id == "nature" }
        head.b2.fill(-10f)
        head.b2[natureIndex] = 10f
        val useCase = OrganizeGalleryUseCase(head, ThresholdResolver(store), store)
        val images = (1..3).map { ImagePrediction(it.toString(), FloatArray(1024)) }

        val summaries = useCase.previewFolderCreation(images)
        assertEquals(1, summaries.size)
        assertEquals("Nature/Nature", summaries.single().folderPath)
        assertEquals(3, summaries.single().imageCount)
        assertEquals(0f, summaries.single().thresholdUsed, 1e-6f)
    }

    @Test
    fun doesNotCreateFolderBelowMinimumImageCount() = runBlocking {
        val store = FakeFeedbackStore()
        val head = ClassificationHead()
        head.b2[0] = 10f
        val useCase = OrganizeGalleryUseCase(head, ThresholdResolver(store), store)
        val images = (1..2).map { ImagePrediction(it.toString(), FloatArray(1024)) }
        assertEquals(0, useCase.previewFolderCreation(images).size)
    }
}
