package com.fgt.galleryfl.data.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class ExtractedFeatures(
    val projection: FloatArray,
    val spatialMap: FloatArray
)

/** Runs the same frozen two-output TFLite backbone used by the server model contract. */
class FeatureExtractor(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private val imageSize = 224
    private val channels = 3
    private val bytesPerChannel = 4
    private val featureSize = 1024
    private val spatialSize = 7 * 7 * 1024

    private val inputBuffer = ByteBuffer.allocateDirect(
        imageSize * imageSize * channels * bytesPerChannel
    ).apply { order(ByteOrder.nativeOrder()) }
    private val outputBufferSpatial = ByteBuffer.allocateDirect(
        spatialSize * bytesPerChannel
    ).apply { order(ByteOrder.nativeOrder()) }
    private val outputBufferProjection = ByteBuffer.allocateDirect(
        featureSize * bytesPerChannel
    ).apply { order(ByteOrder.nativeOrder()) }

    private var projectionOutputIndex = -1
    private var spatialOutputIndex = -1

    init {
        val options = Interpreter.Options()
        try {
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)
        } catch (_: Exception) {
            options.setNumThreads(4)
        }

        val modelBuffer = loadModelFile("models/base_model.tflite")
        val loaded = Interpreter(modelBuffer, options)
        interpreter = loaded
        for (index in 0 until loaded.outputTensorCount) {
            val shape = loaded.getOutputTensor(index).shape()
            when {
                shape.contentEquals(intArrayOf(1, featureSize)) -> projectionOutputIndex = index
                shape.contentEquals(intArrayOf(1, 7, 7, featureSize)) -> spatialOutputIndex = index
            }
        }
        check(projectionOutputIndex >= 0) {
            "GalleryFL backbone does not expose a [1,1024] projection output"
        }
        android.util.Log.d(
            "FeatureExtractor",
            "Canonical model loaded: projection=$projectionOutputIndex spatial=$spatialOutputIndex"
        )
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val descriptor = context.assets.openFd(fileName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    @Synchronized
    fun extractFeatures(bitmap: Bitmap): ExtractedFeatures {
        val resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        inputBuffer.rewind()
        val pixels = IntArray(imageSize * imageSize)
        resized.getPixels(pixels, 0, resized.width, 0, 0, resized.width, resized.height)
        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 127.5f) - 1f)
            inputBuffer.putFloat(((pixel shr 8 and 0xFF) / 127.5f) - 1f)
            inputBuffer.putFloat(((pixel and 0xFF) / 127.5f) - 1f)
        }

        val outputs = mutableMapOf<Int, Any>()
        outputBufferProjection.rewind()
        outputs[projectionOutputIndex] = outputBufferProjection
        if (spatialOutputIndex >= 0) {
            outputBufferSpatial.rewind()
            outputs[spatialOutputIndex] = outputBufferSpatial
        }
        interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        outputBufferProjection.rewind()
        val projection = FloatArray(featureSize)
        outputBufferProjection.asFloatBuffer().get(projection)
        val spatial = if (spatialOutputIndex >= 0) {
            outputBufferSpatial.rewind()
            FloatArray(spatialSize).also { outputBufferSpatial.asFloatBuffer().get(it) }
        } else {
            FloatArray(spatialSize)
        }
        return ExtractedFeatures(projection, spatial)
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
