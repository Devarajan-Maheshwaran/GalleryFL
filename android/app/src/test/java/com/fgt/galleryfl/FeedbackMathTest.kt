package com.fgt.galleryfl

import com.fgt.galleryfl.data.local.FeedbackMath
import org.junit.Assert.assertEquals
import org.junit.Test

class FeedbackMathTest {

    @Test
    fun testThresholdAdaptation() {
        val defaultThreshold = 0.5f

        // Not enough events (4 < 5) -> no change
        var threshold = FeedbackMath.calculateThreshold(confirmed = 2, rejected = 2, defaultThreshold)
        assertEquals(0.5f, threshold, 1e-5f)

        // Lots of rejections -> threshold goes UP (harder to predict)
        threshold = FeedbackMath.calculateThreshold(confirmed = 0, rejected = 10, defaultThreshold)
        // netRejections = 10, adjustment = 10 * 0.01 = +0.10
        assertEquals(0.6f, threshold, 1e-5f)

        // Lots of confirmations -> threshold goes DOWN (easier to predict)
        threshold = FeedbackMath.calculateThreshold(confirmed = 10, rejected = 0, defaultThreshold)
        // netRejections = -10, adjustment = -10 * 0.01 = -0.10
        assertEquals(0.4f, threshold, 1e-5f)

        // Bounded to MAX_THRESHOLD_ADJUSTMENT (0.10)
        threshold = FeedbackMath.calculateThreshold(confirmed = 0, rejected = 100, defaultThreshold)
        assertEquals(0.6f, threshold, 1e-5f)
        
        threshold = FeedbackMath.calculateThreshold(confirmed = 100, rejected = 0, defaultThreshold)
        assertEquals(0.4f, threshold, 1e-5f)
    }

    @Test
    fun testBoundedBiasGeneration() {
        // Not enough events (4 < 5) -> no bias
        var bias = FeedbackMath.calculateBias(confirmed = 2, rejected = 2)
        assertEquals(0f, bias, 1e-5f)

        // More confirmations -> positive bias (easier to cross threshold)
        bias = FeedbackMath.calculateBias(confirmed = 10, rejected = 0)
        // netConfirmations = 10, bias = 10 * 0.1 = 1.0
        assertEquals(1.0f, bias, 1e-5f)

        // More rejections -> negative bias (harder to cross threshold)
        bias = FeedbackMath.calculateBias(confirmed = 0, rejected = 10)
        // netConfirmations = -10, bias = -10 * 0.1 = -1.0
        assertEquals(-1.0f, bias, 1e-5f)

        // Bounded to MAX_BIAS_OFFSET (1.0)
        bias = FeedbackMath.calculateBias(confirmed = 100, rejected = 0)
        assertEquals(1.0f, bias, 1e-5f)
        
        bias = FeedbackMath.calculateBias(confirmed = 0, rejected = 100)
        assertEquals(-1.0f, bias, 1e-5f)
    }
}
