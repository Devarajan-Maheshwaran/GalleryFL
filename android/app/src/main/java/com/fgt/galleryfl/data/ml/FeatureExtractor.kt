package com.fgt.galleryfl.data.ml

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.google.ai.edge.litert.Interpreter
import com.google.ai.edge.litert.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.FileInputStream

class FeatureExtractor(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Model expects 224x224x3 float input (usually)
    // Wait, we need to check if MobileNetV3 expects float or uint8, we exported it as float.
    private val IMAGE_SIZE = 224
    private val CHANNELS = 3
    private val BYTES_PER_CHANNEL = 4 // Float32
    
    // Feature vector size for MobileNetV3Large is 960
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
            // Fallback to CPU if GPU delegate fails
            options.setNumThreads(4)
        }
        
        val modelBuffer = loadModelFile("base_model.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun extractFeatures(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        
        inputBuffer.rewind()
        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)

        var pixel = 0
        for (i in 0 until IMAGE_SIZE) {
            for (j in 0 until IMAGE_SIZE) {
                val valPixel = intValues[pixel++]
                // Normalize to [-1, 1] standard mobilenet
                inputBuffer.putFloat(((valPixel shr 16 and 0xFF) / 127.5f) - 1f)
                inputBuffer.putFloat(((valPixel shr 8 and 0xFF) / 127.5f) - 1f)
                inputBuffer.putFloat(((valPixel and 0xFF) / 127.5f) - 1f)
            }
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
