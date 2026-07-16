package com.fgt.galleryfl.data.network

import android.util.Base64
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Deflater
import java.util.zip.Inflater

object WeightSerializer {
    /**
     * Serializes weights into Little-Endian bytes, zlib compressed, base64 encoded.
     */
    fun serialize(weights: List<FloatArray>): String {
        var totalBytes = 0
        weights.forEach { totalBytes += 4 + (it.size * 4) }

        val combinedBuffer = ByteBuffer.allocate(totalBytes)
        combinedBuffer.order(ByteOrder.LITTLE_ENDIAN)

        for (layer in weights) {
            val layerSizeBytes = layer.size * 4
            combinedBuffer.putInt(layerSizeBytes)
            for (value in layer) {
                combinedBuffer.putFloat(value)
            }
        }

        val combinedArray = combinedBuffer.array()

        // Compress using zlib
        val deflater = Deflater()
        deflater.setInput(combinedArray)
        deflater.finish()

        val compressedBuffer = ByteArray(combinedArray.size + 1024)
        val compressedSize = deflater.deflate(compressedBuffer)
        deflater.end()

        val finalCompressed = compressedBuffer.copyOfRange(0, compressedSize)
        return Base64.encodeToString(finalCompressed, Base64.NO_WRAP)
    }

    /**
     * Decodes base64 -> zlib decompress -> little-endian unpack to float32
     */
    fun deserialize(encoded: String, expectedSizes: List<Int>): List<FloatArray> {
        val compressed = Base64.decode(encoded, Base64.NO_WRAP)
        
        val inflater = Inflater()
        inflater.setInput(compressed)
        
        // Allocate a large buffer for decompressed data
        val maxDecompressedSize = expectedSizes.sum() * 4 + (expectedSizes.size * 4)
        val decompressed = ByteArray(maxDecompressedSize * 2) 
        val decompressedSize = inflater.inflate(decompressed)
        inflater.end()
        
        val byteBuffer = ByteBuffer.wrap(decompressed, 0, decompressedSize)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        
        val result = mutableListOf<FloatArray>()
        for (expectedSize in expectedSizes) {
            val sizeInBytes = byteBuffer.getInt()
            val numFloats = sizeInBytes / 4
            val floatArray = FloatArray(numFloats)
            for (i in 0 until numFloats) {
                floatArray[i] = byteBuffer.getFloat()
            }
            result.add(floatArray)
        }
        
        return result
    }
}
