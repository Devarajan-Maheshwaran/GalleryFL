package com.fgt.galleryfl

import com.fgt.galleryfl.data.ml.ClassificationHead
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClassificationHeadTest {

    @Test
    fun testSchemaDimensions() {
        val head = ClassificationHead(numClasses = 34)
        
        // Exact schema sizes specified
        assertEquals("inputDim should be 1024", 1024, head.inputDim)
        
        // w1: (1024, 256)
        assertEquals(1024, head.w1.size)
        assertEquals(256, head.w1[0].size)
        
        // b1: (256,)
        assertEquals(256, head.b1.size)
        
        // w2: (256, NUM_CLASSES)
        assertEquals(256, head.w2.size)
        assertEquals(34, head.w2[0].size)
        
        // b2: (NUM_CLASSES,)
        assertEquals(34, head.b2.size)
    }

    @Test
    fun testSerializationOrder() {
        val head = ClassificationHead(numClasses = 34)
        val flatWeights = head.getWeightsFlat()
        
        // Ensure exactly 4 arrays are returned
        assertEquals(4, flatWeights.size)
        
        // Check exact flattened sizes
        assertEquals(1024 * 256, flatWeights[0].size)
        assertEquals(256, flatWeights[1].size)
        assertEquals(256 * 34, flatWeights[2].size)
        assertEquals(34, flatWeights[3].size)
    }

    @Test
    fun testBiasOffsets() {
        val head = ClassificationHead(numClasses = 34)
        
        val features = FloatArray(1024) { 0f }
        
        // All weights are initialized to 0, so without bias, logits are 0 -> sigmoid(0) = 0.5
        val outNoBias = head.forward(features, null)
        assertEquals(0.5f, outNoBias[0], 1e-5f)
        
        // With bias offset of 1.0 on the first class, sigmoid(1.0) ≈ 0.731
        val biasOffsets = FloatArray(34) { 0f }
        biasOffsets[0] = 1.0f
        val outWithBias = head.forward(features, biasOffsets)
        
        val expectedSigmoid = (1f / (1f + Math.exp(-1.0))).toFloat()
        assertEquals(expectedSigmoid, outWithBias[0], 1e-5f)
        assertEquals(0.5f, outWithBias[1], 1e-5f) // other classes unaffected
    }
}
