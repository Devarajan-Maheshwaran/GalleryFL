package com.fgt.galleryfl.data.local

import android.content.Context
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

data class TagResult(
    val category: String,
    val subcategory: String,
    val confidence: Float
)

class ExifTagWriter(private val context: Context) {

    suspend fun writeTags(uri: Uri, tags: List<TagResult>) = withContext(Dispatchers.IO) {
        val pfd = context.contentResolver.openFileDescriptor(uri, "rw")
        pfd?.use {
            val exif = ExifInterface(it.fileDescriptor)
            
            val jsonArray = JSONArray()
            for (tag in tags) {
                val obj = JSONObject().apply {
                    put("category", tag.category)
                    put("subcategory", tag.subcategory)
                    put("confidence", tag.confidence.toDouble())
                }
                jsonArray.put(obj)
            }
            
            exif.setAttribute(ExifInterface.TAG_USER_COMMENT, jsonArray.toString())
            exif.saveAttributes()
        }
    }
}
