package com.fgt.galleryfl.data.local

import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Persists the synced classification head ([activeWeights]) and its model
 * version to app-private storage so the model stays loaded across app exits.
 * Without this the app forgot the model on every relaunch and forced a re-sync
 * before Scan & Group could run.
 *
 * Layout (little-endian): int32 layerCount, then per layer int32 floatCount
 * followed by floatCount * float32 bytes. Matches the four FGT head tensors
 * (w1, b1, w2, b2) in [WeightSerializer] order.
 */
object ModelStateStore {
    private const val PREFS = "fgt_prefs"
    private const val KEY_VERSION = "model_version"
    private const val FILE_NAME = "fgt_head.bin"

    fun saveWeights(context: Context, weights: List<FloatArray>) {
        val file = File(context.filesDir, FILE_NAME)
        file.outputStream().use { out ->
            val header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(weights.size)
            out.write(header.array())
            for (layer in weights) {
                val buf = ByteBuffer.allocate(4 + layer.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                buf.putInt(layer.size)
                for (v in layer) buf.putFloat(v)
                out.write(buf.array())
            }
        }
    }

    fun loadWeights(context: Context): List<FloatArray>? {
        val file = File(context.filesDir, FILE_NAME)
        if (!file.exists()) return null
        return try {
            file.inputStream().use { inp ->
                val hb = ByteArray(4)
                if (inp.read(hb) < 4) return null
                val n = ByteBuffer.wrap(hb).order(ByteOrder.LITTLE_ENDIAN).int
                val result = mutableListOf<FloatArray>()
                repeat(n) {
                    val sb = ByteArray(4)
                    if (inp.read(sb) < 4) return null
                    val size = ByteBuffer.wrap(sb).order(ByteOrder.LITTLE_ENDIAN).int
                    val fb = ByteArray(size * 4)
                    if (inp.read(fb) < fb.size) return null
                    val fbBuf = ByteBuffer.wrap(fb).order(ByteOrder.LITTLE_ENDIAN)
                    val arr = FloatArray(size)
                    for (j in 0 until size) arr[j] = fbBuf.getFloat()
                    result.add(arr)
                }
                if (result.size == 4) result else null
            }
        } catch (_: Exception) {
            null
        }
    }

    fun clearWeights(context: Context) {
        File(context.filesDir, FILE_NAME).delete()
    }

    fun saveModelVersion(context: Context, version: Int) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putInt(KEY_VERSION, version).apply()
    }

    fun loadModelVersion(context: Context): Int {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getInt(KEY_VERSION, 0)
    }
}
