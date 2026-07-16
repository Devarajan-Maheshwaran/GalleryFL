package com.fgt.galleryfl.data.local

import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class GalleryImage(
    val id: Long,
    val uri: Uri,
    val displayName: String,
    val dateAdded: Long
)

class GalleryRepository(private val context: Context) {

    suspend fun fetchRecentImages(limit: Int = 100): List<GalleryImage> = withContext(Dispatchers.IO) {
        val images = mutableListOf<GalleryImage>()
        
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATE_ADDED
        )
        
        val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"
        
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            sortOrder
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            val dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
            
            while (cursor.moveToNext() && images.size < limit) {
                val id = cursor.getLong(idColumn)
                val name = cursor.getString(nameColumn)
                val date = cursor.getLong(dateColumn)
                
                val uri = Uri.withAppendedPath(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    id.toString()
                )
                
                images.add(GalleryImage(id, uri, name, date))
            }
        }
        images
    }
}
