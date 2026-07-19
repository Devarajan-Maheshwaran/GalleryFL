package com.fgt.galleryfl.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface ScanResultDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(results: List<ScanResultEntity>)

    @Query("SELECT * FROM scan_result ORDER BY classIndex ASC, positionInAlbum ASC")
    suspend fun getAll(): List<ScanResultEntity>

    @Query("DELETE FROM scan_result")
    suspend fun clear()
}
