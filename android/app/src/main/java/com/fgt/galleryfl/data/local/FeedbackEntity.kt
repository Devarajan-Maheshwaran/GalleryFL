package com.fgt.galleryfl.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "feedback")
data class FeedbackEntity(
    @PrimaryKey val imageId: Long,
    val classIndex: Int,
    val isConfirmed: Boolean
)
