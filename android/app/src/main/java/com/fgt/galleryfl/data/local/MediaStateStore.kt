package com.fgt.galleryfl.data.local

import android.content.Context

/**
 * Convenience facade over [MediaStateDao] for the Favorites / Archive / Trash
 * utilities shown in the For You tab. Persists a single row per image URI and
 * lets the UI toggle each flag independently.
 */
object MediaStateStore {

    private fun now() = System.currentTimeMillis()

    private suspend fun update(
        context: Context,
        uri: String,
        transform: (MediaStateEntity) -> MediaStateEntity
    ) {
        val dao = AppDatabase.getDatabase(context).mediaStateDao()
        val current = dao.getForUri(uri) ?: MediaStateEntity(uri = uri)
        dao.upsert(transform(current))
    }

    suspend fun setFavorite(context: Context, uri: String, value: Boolean) =
        update(context, uri) { it.copy(favorite = value, updatedAt = now()) }

    suspend fun setArchived(context: Context, uri: String, value: Boolean) =
        update(context, uri) { it.copy(archived = value, updatedAt = now()) }

    suspend fun setTrashed(context: Context, uri: String, value: Boolean) =
        update(context, uri) { it.copy(trashed = value, updatedAt = now()) }

    suspend fun getFavorites(context: Context): List<String> =
        AppDatabase.getDatabase(context).mediaStateDao().getFavorites().map { it.uri }

    suspend fun getArchived(context: Context): List<String> =
        AppDatabase.getDatabase(context).mediaStateDao().getArchived().map { it.uri }

    suspend fun getTrashed(context: Context): List<String> =
        AppDatabase.getDatabase(context).mediaStateDao().getTrashed().map { it.uri }
}
