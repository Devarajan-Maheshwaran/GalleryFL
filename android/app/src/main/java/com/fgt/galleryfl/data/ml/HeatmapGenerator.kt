package com.fgt.galleryfl.data.ml

import kotlin.math.max

class HeatmapGenerator(private val classificationHead: ClassificationHead) {

    /**
     * Generates a 7x7 heatmap for a specific class index based on the spatial feature map.
     * 
     * @param spatialMap The 7x7x1024 spatial feature map extracted from the base model
     * @param targetClassIndex The class index to generate the heatmap for
     * @return A 7x7 2D FloatArray representing the heatmap, normalized between 0 and 1
     */
    fun generateHeatmap(spatialMap: FloatArray, targetClassIndex: Int): Array<FloatArray> {
        val w1 = classificationHead.w1 // Shape: 1024 x 256
        val w2 = classificationHead.w2 // Shape: 256 x 7 parent classes
        
        // 1. Compute the backward linear projection from class score to the 1024-dim features
        // We want the gradient of the class score with respect to the 1024-dim input of ClassificationHead
        // Output class score k = sum_j (w2[j][k] * relu(sum_i (w1[i][j] * x[i])))
        // Assuming linear activation for the back-projection (approximate GradCAM)
        // v_i = sum_j (w1[i][j] * w2[j][k])
        
        val classWeights = FloatArray(1024)
        for (i in 0 until 1024) {
            var sum = 0f
            for (j in 0 until 256) {
                val weight1 = w1[i][j]
                val weight2 = w2[j][targetClassIndex]
                sum += weight1 * weight2
            }
            classWeights[i] = sum
        }
        
        // 2. Compute the weighted sum across the 7x7 spatial locations
        val heatmap = Array(7) { FloatArray(7) }
        var maxVal = Float.MIN_VALUE
        var minVal = Float.MAX_VALUE
        
        for (y in 0 until 7) {
            for (x in 0 until 7) {
                var score = 0f
                for (c in 0 until 1024) {
                    val featureVal = spatialMap[(y * 7 + x) * 1024 + c]
                    score += featureVal * classWeights[c]
                }
                // Apply ReLU to keep only positive contributions
                score = max(0f, score)
                heatmap[y][x] = score
                
                if (score > maxVal) maxVal = score
                if (score < minVal) minVal = score
            }
        }
        
        // 3. Normalize the heatmap
        val range = maxVal - minVal
        if (range > 0) {
            for (y in 0 until 7) {
                for (x in 0 until 7) {
                    heatmap[y][x] = (heatmap[y][x] - minVal) / range
                }
            }
        }
        
        return heatmap
    }
}
