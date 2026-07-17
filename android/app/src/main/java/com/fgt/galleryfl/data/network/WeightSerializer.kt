package com.fgt.galleryfl.data.network

import android.util.Base64
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Deflater
import java.util.zip.Inflater

object WeightSerializer {

    val LAYER_SCHEMA: List<Pair<String, Int>> = listOf(
        "w1" to (1024 * 256),
        "b1" to 256,
        "w2" to (256 * 20),
        "b2" to 20
    )

    fun serialize(weights: List<FloatArray>): String {
        var totalBytes = 0
        for ((_, size) in LAYER_SCHEMA) {
            totalBytes += 4 + (size * 4)
        }

        val combinedBuffer = ByteBuffer.allocate(totalBytes)
        combinedBuffer.order(ByteOrder.LITTLE_ENDIAN)

        for ((idx, entry) in LAYER_SCHEMA.withIndex()) {
            val layer = weights[idx]
            val layerSizeBytes = layer.size * 4
            combinedBuffer.putInt(layerSizeBytes)
            for (value in layer) {
                combinedBuffer.putFloat(value)
            }
        }

        val combinedArray = combinedBuffer.array()

        val deflater = Deflater()
        deflater.setInput(combinedArray)
        deflater.finish()

        val compressedBuffer = ByteArray(combinedArray.size + 1024)
        val compressedSize = deflater.deflate(compressedBuffer)
        deflater.end()

        val finalCompressed = compressedBuffer.copyOfRange(0, compressedSize)
        return Base64.encodeToString(finalCompressed, Base64.NO_WRAP)
    }

    fun deserialize(encoded: String): List<FloatArray> {
        val compressed = Base64.decode(encoded, Base64.NO_WRAP)

        val inflater = Inflater()
        inflater.setInput(compressed)

        var maxDecompressed = 0
        for ((_, size) in LAYER_SCHEMA) {
            maxDecompressed += 4 + (size * 4)
        }

        val decompressed = ByteArray(maxDecompressed + 1024)
        val decompressedSize = inflater.inflate(decompressed)
        inflater.end()

        val byteBuffer = ByteBuffer.wrap(decompressed, 0, decompressedSize)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)

        val result = mutableListOf<FloatArray>()
        for ((_, expectedSize) in LAYER_SCHEMA) {
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
