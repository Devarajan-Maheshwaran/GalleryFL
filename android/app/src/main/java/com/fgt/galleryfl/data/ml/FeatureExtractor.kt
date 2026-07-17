package com.fgt.galleryfl.data.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.FileInputStream

class FeatureExtractor(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private val IMAGE_SIZE = 224
    private val CHANNELS = 3
    private val BYTES_PER_CHANNEL = 4
    private val FEATURE_SIZE = 960

    private val inputBuffer = ByteBuffer.allocateDirect(1 * IMAGE_SIZE * IMAGE_SIZE * CHANNELS * BYTES_PER_CHANNEL).apply {
        order(ByteOrder.nativeOrder())
    }

    private val outputBuffer = ByteBuffer.allocateDirect(1 * FEATURE_SIZE * BYTES_PER_CHANNEL).apply {
        order(ByteOrder.nativeOrder())
    }

    init {
        val options = Interpreter.Options()
        try {
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)
        } catch (e: Exception) {
            options.setNumThreads(4)
        }

        val modelBuffer = loadModelFile("models/base_model.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun extractFeatures(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)

        inputBuffer.rewind()
        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)

        for (pixel in intValues) {
            inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 127.5f) - 1f)
            inputBuffer.putFloat(((pixel shr 8 and 0xFF) / 127.5f) - 1f)
            inputBuffer.putFloat(((pixel and 0xFF) / 127.5f) - 1f)
        }

        outputBuffer.rewind()
        interpreter?.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val features = FloatArray(FEATURE_SIZE)
        outputBuffer.asFloatBuffer().get(features)
        return features
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
