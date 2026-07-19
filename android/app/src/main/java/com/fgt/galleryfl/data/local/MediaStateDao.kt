package com.fgt.galleryfl.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface MediaStateDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(state: MediaStateEntity)

    @Query("SELECT * FROM media_state WHERE favorite = 1")
    suspend fun getFavorites(): List<MediaStateEntity>

    @Query("SELECT * FROM media_state WHERE archived = 1")
    suspend fun getArchived(): List<MediaStateEntity>

    @Query("SELECT * FROM media_state WHERE trashed = 1")
    suspend fun getTrashed(): List<MediaStateEntity>

    @Query("SELECT * FROM media_state WHERE uri = :uri")
    suspend fun getForUri(uri: String): MediaStateEntity?
}
