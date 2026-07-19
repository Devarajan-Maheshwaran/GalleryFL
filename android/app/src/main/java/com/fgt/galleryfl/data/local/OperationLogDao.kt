package com.fgt.galleryfl.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface OperationLogDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLog(log: OperationLogEntity): Long

    @Query("SELECT * FROM operation_log ORDER BY id DESC")
    suspend fun getAllLogs(): List<OperationLogEntity>

    @Query("SELECT * FROM operation_log ORDER BY id DESC LIMIT 1")
    suspend fun getLatestLog(): OperationLogEntity?

    @Query("DELETE FROM operation_log WHERE id = :id")
    suspend fun deleteLog(id: Long)
}
