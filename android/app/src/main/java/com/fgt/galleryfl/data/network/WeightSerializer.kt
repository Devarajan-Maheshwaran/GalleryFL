package com.fgt.galleryfl.data.network

import android.util.Base64
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Deflater
import java.util.zip.Inflater

object WeightSerializer {

    fun serialize(weights: List<FloatArray>): String {
        var totalBytes = 0
        for (layer in weights) {
            totalBytes += 4 + (layer.size * 4)
        }

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

        val maxDecompressed = 1024 * 1024 * 5 // Deliberate upper bound for the head-only contract.
        val decompressed = ByteArray(maxDecompressed)
        val decompressedSize = inflater.inflate(decompressed)
        inflater.end()
        require(decompressedSize > 0) { "Weight payload did not decompress" }

        val byteBuffer = ByteBuffer.wrap(decompressed, 0, decompressedSize)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)

        val result = mutableListOf<FloatArray>()
        // FGT v1 has exactly four schema-ordered head tensors: w1, b1, w2, b2.
        for (i in 0 until 4) {
            require(byteBuffer.remaining() >= 4) { "Missing length prefix for tensor $i" }
            val sizeInBytes = byteBuffer.getInt()
            require(sizeInBytes >= 0 && sizeInBytes % 4 == 0) { "Invalid tensor length for tensor $i" }
            require(byteBuffer.remaining() >= sizeInBytes) { "Truncated tensor payload for tensor $i" }
            val numFloats = sizeInBytes / 4
            val floatArray = FloatArray(numFloats)
            for (j in 0 until numFloats) {
                floatArray[j] = byteBuffer.getFloat()
            }
            result.add(floatArray)
        }
        require(byteBuffer.remaining() == 0) { "Unexpected trailing bytes in weight payload" }
        return result
    }
}
