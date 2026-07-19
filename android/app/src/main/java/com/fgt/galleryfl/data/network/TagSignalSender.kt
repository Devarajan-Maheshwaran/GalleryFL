package com.fgt.galleryfl.data.network

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Fire-and-forget client half of the server's demand-weighted taxonomy feature.
 *
 * The server accumulates the tags clients actually *use* — predicted with high
 * confidence, or applied while organizing — into a recency-weighted demand
 * signal that drives the dashboard's Trending Tags view. See server/tag_demand.py
 * and the POST /api/taxonomy/signal endpoint.
 *
 * The sender is configured once at connection time (with the live API service,
 * access token and client id) and then emission sites call [emit] without those
 * dependencies having to be threaded through the Compose tree. All failures are
 * swallowed: the demand signal is strictly best-effort and must never block or
 * crash the UI.
 */
object TagSignalSender {
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var apiService: FGTApiService? = null
    private var token: String? = null
    private var clientId: String? = null

    fun configure(apiService: FGTApiService, token: String, clientId: String) {
        this.apiService = apiService
        this.token = token
        this.clientId = clientId
    }

    fun emit(tags: List<String>) {
        val svc = apiService ?: return
        val tk = token ?: return
        if (tags.isEmpty()) return
        val cid = clientId
        scope.launch {
            try {
                svc.postTagSignal(tk, TagSignalRequest(cid, tags))
            } catch (_: Exception) {
                // Best-effort demand signal; ignore any network/serialization error.
            }
        }
    }
}
