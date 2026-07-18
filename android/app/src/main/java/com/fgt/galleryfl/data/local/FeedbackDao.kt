package com.fgt.galleryfl.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface FeedbackDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertFeedback(feedback: FeedbackEntity)

    @Query("SELECT * FROM feedback WHERE classIndex = :classIndex")
    suspend fun getFeedbackForClass(classIndex: Int): List<FeedbackEntity>
    
    @Query("SELECT * FROM feedback WHERE imageId = :imageId")
    suspend fun getFeedbackForImage(imageId: Long): FeedbackEntity?
    
    @Query("SELECT * FROM feedback")
    suspend fun getAllFeedback(): List<FeedbackEntity>
}
