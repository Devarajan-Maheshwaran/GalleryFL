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
import org.junit.Assert.assertTrue
import org.junit.Test

class FakeFeedbackStore : FeedbackStore {
    val stats = mutableMapOf<Int, ClassStats>()
    val overrides = mutableMapOf<Int, Float>()
    val biasOffsets = FloatArray(TaxonomyConfig.NUM_CLASSES) { 0f }

    override suspend fun recordPredicted(classIndex: Int) {}
    override suspend fun recordConfirmed(classIndex: Int) {}
    override suspend fun recordRejected(classIndex: Int) {}
    
    override suspend fun getPerClassStats(classIndex: Int): ClassStats {
        return stats[classIndex] ?: ClassStats(0, 0, 0)
    }

    override suspend fun getThresholdOverride(classIndex: Int, defaultThreshold: Float): Float {
        return overrides[classIndex] ?: defaultThreshold
    }

    override suspend fun getBiasOffsets(numClasses: Int): FloatArray {
        return biasOffsets
    }
}

class OrganizeGalleryUseCaseTest {

    @Test
    fun testDemandDrivenFolderCreation() = runBlocking {
        val fakeStore = FakeFeedbackStore()
        val classificationHead = ClassificationHead(numClasses = TaxonomyConfig.NUM_CLASSES)
        val resolver = ThresholdResolver(fakeStore)
        val useCase = OrganizeGalleryUseCase(classificationHead, resolver, fakeStore)

        // Find index for 'beach' (Places -> beach)
        val beachIndex = TaxonomyConfig.leafTags.indexOfFirst { it.id == "beach" }
        // Find index for 'mountain' (Places -> mountain)
        val mountainIndex = TaxonomyConfig.leafTags.indexOfFirst { it.id == "mountain" }
        
        // Ensure they exist
        assertTrue(beachIndex >= 0)
        assertTrue(mountainIndex >= 0)
        
        // Mock classification head logic: 
        // A real classification head forward pass takes features (1024) and outputs (33).
        // Since weights are 0, it outputs 0.5 for all classes normally. 
        // We will mock the prediction by replacing weights so a specific feature creates a specific score.
        // Actually, an easier way is to just set b2 biases extremely low (-10) for everything except our target class (+10).
        
        for (i in 0 until TaxonomyConfig.NUM_CLASSES) {
            classificationHead.b2[i] = -10f // ~0.000045 sigmoid
        }
        classificationHead.b2[beachIndex] = 10f // ~0.9999 sigmoid

        // 3 beach images (minImagesToCreateFolder is 3)
        val images = listOf(
            ImagePrediction("1", FloatArray(1024) { 0f }),
            ImagePrediction("2", FloatArray(1024) { 0f }),
            ImagePrediction("3", FloatArray(1024) { 0f })
        )

        val summaries = useCase.previewFolderCreation(images)
        
        // Should create exactly one folder: Places/Beach
        assertEquals(1, summaries.size)
        assertEquals("Places/Beach", summaries[0].folderPath)
        assertEquals(3, summaries[0].imageCount)
        assertEquals(0.5f, summaries[0].thresholdUsed, 0.01f) // Default threshold
        
        // Test that 2 mountain images do not create a folder
        classificationHead.b2[mountainIndex] = 10f
        val imagesWithMountain = images + listOf(
            ImagePrediction("4", FloatArray(1024) { 0f }),
            ImagePrediction("5", FloatArray(1024) { 0f })
        )
        // Since features are same, all 5 images will now be both Beach and Mountain.
        // But only 2 "Mountain" specific ones if we used distinct features. 
        // Wait, for this test all 5 images would score high for both beach and mountain.
        // Let's reset b2[beachIndex] = -10f and just evaluate 2 images.
        
        classificationHead.b2[beachIndex] = -10f
        val twoImages = listOf(
            ImagePrediction("4", FloatArray(1024) { 0f }),
            ImagePrediction("5", FloatArray(1024) { 0f })
        )
        
        val summaries2 = useCase.previewFolderCreation(twoImages)
        // Should create NO folders because 2 < 3
        assertEquals(0, summaries2.size)
    }
}
