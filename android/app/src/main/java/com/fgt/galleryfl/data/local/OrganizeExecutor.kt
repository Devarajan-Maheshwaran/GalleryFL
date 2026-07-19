package com.fgt.galleryfl.data.local

import android.content.Context
import android.net.Uri
import com.fgt.galleryfl.data.ml.ClassificationHead
import com.fgt.galleryfl.data.ml.FeatureExtractor
import com.fgt.galleryfl.data.ml.ThresholdResolver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Executes the "Organize gallery" action end-to-end and supports Undo.
 *
 * Flow:
 *   1. Extract features for on-device images.
 *   2. Ask [OrganizeGalleryUseCase] which tag folders would be created.
 *   3. For each folder, copy the matched photos into Pictures/FGT/<tag>
 *      ([AlbumCreator]) and write the predicted tag(s) into each photo's EXIF
 *      user-comment ([ExifTagWriter]).
 *   4. Persist an operation log ([AppDatabase]) so the whole batch can be
 *      reverted with [undoLast].
 *
 * All four building blocks requested for the UI wiring
 * (AlbumCreator / ExifTagWriter / OrganizeGalleryUseCase / AppDatabase) are
 * used here; MainActivity only triggers it and reflects status.
 */
class OrganizeExecutor(
    private val context: Context,
    private val featureExtractor: FeatureExtractor,
    private val feedbackStore: FeedbackStore,
    private val albumCreator: AlbumCreator,
    private val exifTagWriter: ExifTagWriter,
    private val db: AppDatabase
) {
    suspend fun organize(activeWeights: List<FloatArray>): OrganizeOutcome = withContext(Dispatchers.IO) {
        val numClasses = activeWeights[2].size / 256
        val head = ClassificationHead(numClasses)
        head.setWeightsFlat(activeWeights)
        val resolver = ThresholdResolver(feedbackStore)
        val useCase = OrganizeGalleryUseCase(head, resolver, feedbackStore)

        val repo = GalleryRepository(context)
        val images = repo.fetchRecentImages(limit = 500)
        val loaded = images.mapNotNull { img ->
            repo.loadBitmap(img.uri)?.let { bmp -> img to bmp }
        }
        val valid = loaded.map { it.first }
        val features = loaded.map { (_, bmp) -> featureExtractor.extractFeatures(bmp).projection }

        val predictions = valid.mapIndexed { i, img ->
            ImagePrediction(img.id.toString(), features[i])
        }
        val summaries = useCase.previewFolderCreation(predictions)

        val folders = mutableListOf<OpFolder>()
        val tagsByImage = mutableMapOf<Long, MutableList<TagResult>>()

        for (summary in summaries) {
            if (summary.assignedImages.isEmpty()) continue
            val category = summary.folderPath.substringBefore("/")
            val tag = summary.folderPath.substringAfter("/")
            val matched = valid.filter { it.id.toString() in summary.assignedImages }

            val album: AlbumResult = albumCreator.createTagAlbum(tag, matched.map { it.uri })
            folders.add(
                OpFolder(
                    tag = tag,
                    folderPath = summary.folderPath,
                    copiedCount = album.copiedCount,
                    newUris = album.newUris.map { it.toString() }
                )
            )
            for (img in matched) {
                tagsByImage.getOrPut(img.id) { mutableListOf() }
                    .add(TagResult(category, tag, summary.averageConfidence))
            }
        }

        // Write each photo's full set of predicted tags once (EXIF user-comment
        // is overwritten, so multiple tags must be written together).
        for ((imgId, tags) in tagsByImage) {
            val img = valid.firstOrNull { it.id == imgId } ?: continue
            exifTagWriter.writeTags(img.uri, tags)
        }

        if (folders.isNotEmpty()) {
            val log = OperationLogEntity(
                operationType = "organize",
                timestamp = System.currentTimeMillis(),
                detailsJson = OpFolder.listToJson(folders)
            )
            db.operationLogDao().insertLog(log)
        }

        OrganizeOutcome(folders = folders, totalCopied = folders.sumOf { it.copiedCount })
    }

    suspend fun undoLast(): Boolean = withContext(Dispatchers.IO) {
        val log = db.operationLogDao().getLatestLog() ?: return@withContext false
        val folders = OpFolder.listFromJson(log.detailsJson)
        for (folder in folders) {
            for (uriStr in folder.newUris) {
                try {
                    context.contentResolver.delete(Uri.parse(uriStr), null, null)
                } catch (_: Exception) {
                    // Ignore individual delete failures; best-effort revert.
                }
            }
        }
        db.operationLogDao().deleteLog(log.id)
        true
    }

    suspend fun hasUndoable(): Boolean = withContext(Dispatchers.IO) {
        db.operationLogDao().getLatestLog() != null
    }
}

data class OpFolder(
    val tag: String,
    val folderPath: String,
    val copiedCount: Int,
    val newUris: List<String>
) {
    companion object {
        fun listToJson(folders: List<OpFolder>): String {
            val arr = JSONArray()
            for (f in folders) {
                val o = JSONObject()
                o.put("tag", f.tag)
                o.put("folderPath", f.folderPath)
                o.put("copiedCount", f.copiedCount)
                val uris = JSONArray()
                f.newUris.forEach { uris.put(it) }
                o.put("newUris", uris)
                arr.put(o)
            }
            return arr.toString()
        }

        fun listFromJson(json: String): List<OpFolder> {
            val arr = JSONArray(json)
            val out = mutableListOf<OpFolder>()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val uris = mutableListOf<String>()
                val u = o.getJSONArray("newUris")
                for (j in 0 until u.length()) uris.add(u.getString(j))
                out.add(
                    OpFolder(
                        tag = o.getString("tag"),
                        folderPath = o.getString("folderPath"),
                        copiedCount = o.getInt("copiedCount"),
                        newUris = uris
                    )
                )
            }
            return out
        }
    }
}

data class OrganizeOutcome(val folders: List<OpFolder>, val totalCopied: Int)
