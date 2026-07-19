package com.fgt.galleryfl.data.local

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.FileInputStream
import java.io.FileOutputStream

class AlbumCreator(private val context: Context) {

    suspend fun createTagAlbum(tagName: String, imageUris: List<Uri>): AlbumResult = withContext(Dispatchers.IO) {
        val newUris = mutableListOf<Uri>()
        val failedUris = mutableListOf<Uri>()

        for (uri in imageUris) {
            val originalName = getFileName(uri) ?: "image_${System.currentTimeMillis()}.jpg"
            
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "FGT_$originalName")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/FGT/$tagName")
                }
            }

            val contentResolver = context.contentResolver
            val newUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            
            if (newUri != null) {
                try {
                    contentResolver.openFileDescriptor(uri, "r")?.use { pfdIn ->
                        contentResolver.openFileDescriptor(newUri, "w")?.use { pfdOut ->
                            val input = FileInputStream(pfdIn.fileDescriptor)
                            val output = FileOutputStream(pfdOut.fileDescriptor)
                            input.copyTo(output)
                        }
                    }
                    newUris.add(newUri)
                } catch (e: Exception) {
                    failedUris.add(uri)
                    contentResolver.delete(newUri, null, null)
                }
            } else {
                failedUris.add(uri)
            }
        }

        AlbumResult(
            albumPath = "Pictures/FGT/$tagName",
            copiedCount = newUris.size,
            failedUris = failedUris,
            newUris = newUris
        )
    }

    private fun getFileName(uri: Uri): String? {
        var name: String? = null
        val projection = arrayOf(MediaStore.MediaColumns.DISPLAY_NAME)
        context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                name = cursor.getString(0)
            }
        }
        return name
    }
}

data class AlbumResult(
    val albumPath: String,
    val copiedCount: Int,
    val failedUris: List<Uri>,
    val newUris: List<Uri>
)
