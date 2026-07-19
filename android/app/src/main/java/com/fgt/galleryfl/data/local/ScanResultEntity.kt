package com.fgt.galleryfl.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Persisted record of a Scan & Group run so the user's albums survive app
 * exit. One row per (album, image) membership; the album is rebuilt on launch
 * by grouping on [classIndex] and ordering on [positionInAlbum].
 */
@Entity(tableName = "scan_result")
data class ScanResultEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val classIndex: Int,
    val tagName: String,
    val folderPath: String,
    val imageUri: String,
    val confidence: Float,
    val positionInAlbum: Int,
    val avgConfidence: Float,
    val thresholdUsed: Float
)
