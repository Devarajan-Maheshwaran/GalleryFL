package com.fgt.galleryfl.data.local

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.fgt.galleryfl.data.taxonomy.TaxonomyConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlin.math.max
import kotlin.math.min

val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "fgt_feedback_store")

interface FeedbackStore {
    suspend fun recordPredicted(classIndex: Int)
    suspend fun recordConfirmed(classIndex: Int)
    suspend fun recordRejected(classIndex: Int)
    /** Record that this user actually *used* (applied / predicted with high
     * confidence) a tag. This is the local half of demand-driven Non-IID
     * personalisation: the tags a user exercises shape how eagerly the on-device
     * head fires them for that user. */
    suspend fun recordDemand(classIndex: Int)
    suspend fun getPerClassStats(classIndex: Int): ClassStats
    suspend fun getThresholdOverride(classIndex: Int, defaultThreshold: Float): Float
    suspend fun getBiasOffsets(numClasses: Int): FloatArray
}

class LocalFeedbackStore(private val context: Context) : FeedbackStore {

    // Threshold boundaries
    private val MIN_EVIDENCE_EVENTS = 5
    private val MAX_THRESHOLD_ADJUSTMENT = 0.10f
    private val ADJUSTMENT_RATE = 0.01f

    // Personalization boundaries
    private val MAX_BIAS_OFFSET = 1.0f
    private val BIAS_RATE = 0.1f

    override suspend fun recordPredicted(classIndex: Int) {
        incrementCounter("predicted_$classIndex")
    }

    override suspend fun recordConfirmed(classIndex: Int) {
        incrementCounter("confirmed_$classIndex")
    }

    override suspend fun recordRejected(classIndex: Int) {
        incrementCounter("rejected_$classIndex")
    }

    override suspend fun recordDemand(classIndex: Int) {
        incrementCounter("demand_$classIndex")
    }

    private suspend fun getDemandCount(classIndex: Int): Int {
        val prefs = context.dataStore.data.first()
        return prefs[intPreferencesKey("demand_$classIndex")] ?: 0
    }

    private suspend fun incrementCounter(keyName: String) {
        val key = intPreferencesKey(keyName)
        context.dataStore.edit { preferences ->
            val current = preferences[key] ?: 0
            preferences[key] = current + 1
        }
    }

    override suspend fun getPerClassStats(classIndex: Int): ClassStats {
        val prefs = context.dataStore.data.first()
        val predicted = prefs[intPreferencesKey("predicted_$classIndex")] ?: 0
        val confirmed = prefs[intPreferencesKey("confirmed_$classIndex")] ?: 0
        val rejected = prefs[intPreferencesKey("rejected_$classIndex")] ?: 0
        return ClassStats(predicted, confirmed, rejected)
    }

    override suspend fun getThresholdOverride(classIndex: Int, defaultThreshold: Float): Float {
        val stats = getPerClassStats(classIndex)
        return FeedbackMath.calculateThreshold(stats.confirmed, stats.rejected, defaultThreshold)
    }

    override suspend fun getBiasOffsets(numClasses: Int): FloatArray {
        val offsets = FloatArray(numClasses)
        if (!PersonalizationConfig.isLocalBiasEnabled) return offsets

        // Demand-driven Non-IID: normalise this user's local tag demand so the
        // bias is centred (0.5 = average usage). Tags a user exercises far more
        // than average get a positive offset (fire more readily for them); tags
        // they rarely use get a negative offset. This is what makes smart
        // tagging adapt per-user instead of being identical for everyone.
        val demandCounts = IntArray(numClasses) { i -> getDemandCount(i) }
        val dMin = demandCounts.minOrNull() ?: 0
        val dMax = demandCounts.maxOrNull() ?: 0
        val demandRange = (dMax - dMin).coerceAtLeast(1)

        for (i in 0 until numClasses) {
            val stats = getPerClassStats(i)
            val feedbackBias = FeedbackMath.calculateBias(stats.confirmed, stats.rejected)
            val norm = (demandCounts[i] - dMin).toFloat() / demandRange
            val demandBias = FeedbackMath.calculateDemandBias(norm)
            offsets[i] = max(-MAX_BIAS_OFFSET, min(MAX_BIAS_OFFSET, feedbackBias + demandBias))
        }
        return offsets
    }
}

object FeedbackMath {
    private const val MIN_EVIDENCE_EVENTS = 5
    private const val MAX_THRESHOLD_ADJUSTMENT = 0.10f
    private const val ADJUSTMENT_RATE = 0.01f
    private const val MAX_BIAS_OFFSET = 1.0f
    private const val BIAS_RATE = 0.1f
    private const val DEMAND_BIAS_GAIN = 0.8f

    fun calculateThreshold(confirmed: Int, rejected: Int, defaultThreshold: Float): Float {
        val totalEvents = confirmed + rejected
        if (totalEvents < MIN_EVIDENCE_EVENTS) return defaultThreshold

        val netRejections = rejected - confirmed
        var adjustment = netRejections * ADJUSTMENT_RATE
        adjustment = max(-MAX_THRESHOLD_ADJUSTMENT, min(MAX_THRESHOLD_ADJUSTMENT, adjustment))
        return max(0.1f, min(0.9f, defaultThreshold + adjustment))
    }

    fun calculateBias(confirmed: Int, rejected: Int): Float {
        val totalEvents = confirmed + rejected
        if (totalEvents < MIN_EVIDENCE_EVENTS) return 0f

        val netConfirmations = confirmed - rejected
        var bias = netConfirmations * BIAS_RATE
        return max(-MAX_BIAS_OFFSET, min(MAX_BIAS_OFFSET, bias))
    }

    /** Demand bias from a normalised usage value in [0,1] (0.5 = average).
     * Positive when a user exercises a tag more than average, negative when less. */
    fun calculateDemandBias(normalised: Float): Float {
        val centered = (normalised - 0.5f) * 2f
        return max(-MAX_BIAS_OFFSET, min(MAX_BIAS_OFFSET, centered * DEMAND_BIAS_GAIN))
    }
}

data class ClassStats(
    val predicted: Int,
    val confirmed: Int,
    val rejected: Int
)

object PersonalizationConfig {
    var isLocalBiasEnabled: Boolean = true
}
