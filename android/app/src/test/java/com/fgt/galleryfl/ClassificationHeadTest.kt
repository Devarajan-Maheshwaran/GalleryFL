package com.fgt.galleryfl

import com.fgt.galleryfl.data.ml.ClassificationHead
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClassificationHeadTest {

    @Test
    fun testSevenParentSchemaDimensions() {
        val head = ClassificationHead()
        assertEquals(7, TaxonomyConfig.NUM_CLASSES)
        assertEquals(1024, head.inputDim)
        assertEquals(1024, head.w1.size)
        assertEquals(256, head.w1[0].size)
        assertEquals(256, head.b1.size)
        assertEquals(256, head.w2.size)
        assertEquals(7, head.w2[0].size)
        assertEquals(7, head.b2.size)

        val flat = head.getWeightsFlat()
        assertEquals(listOf(1024 * 256, 256, 256 * 7, 7), flat.map { it.size })
    }

    @Test
    fun testSoftmaxAndBiasOffsets() {
        val head = ClassificationHead()
        val features = FloatArray(1024)
        val uniform = head.forward(features)
        assertEquals(1f, uniform.sum(), 1e-5f)
        uniform.forEach { assertEquals(1f / 7f, it, 1e-5f) }

        val offsets = FloatArray(7)
        offsets[0] = 1f
        val personalized = head.forward(features, offsets)
        assertEquals(1f, personalized.sum(), 1e-5f)
        assertTrue(personalized[0] > personalized[1])
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsLegacy34ClassHead() {
        ClassificationHead(numClasses = 34)
    }
}
