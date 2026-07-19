package com.fgt.galleryfl.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Per-image organisational state backing the For You utilities:
 * Favorites, Archive and Trash. A single row per image URI is upserted as the
 * user toggles these states; the main gallery grid hides trashed images.
 */
@Entity(tableName = "media_state")
data class MediaStateEntity(
    @PrimaryKey val uri: String,
    val favorite: Boolean = false,
    val archived: Boolean = false,
    val trashed: Boolean = false,
    val updatedAt: Long = 0L
)
