package com.fgt.galleryfl.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Append-only log of reversible local operations (e.g. gallery organization),
 * so the user can Undo them. `detailsJson` stores the operation-specific
 * payload (which albums / copied image URIs were created) needed to revert.
 */
@Entity(tableName = "operation_log")
data class OperationLogEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val operationType: String,
    val timestamp: Long,
    val detailsJson: String
)
