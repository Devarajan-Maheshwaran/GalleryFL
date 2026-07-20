package com.fgt.galleryfl.data.network

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class FLRoundConfig(
    val learningRate: Float,
    val localEpochs: Int,
    val mu: Float,
    val dpEpsilon: Float,
    val dpDelta: Float,
    val maxGradNorm: Float,
    val pseudoLabelThreshold: Float,
    val pseudoLabelWeight: Float,
    val minLocalSamples: Int,
)

class FGTWebSocketClient(private val client: OkHttpClient) {
    private var webSocket: WebSocket? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var reconnectJob: Job? = null
    private var heartbeatJob: Job? = null
    private var connection: Connection? = null
    private var manualDisconnect = true
    private var retryAttempt = 0

    private data class Connection(val serverUrl: String, val clientId: String, val token: String)

    var onUpdateRequested: ((Int, FLRoundConfig) -> Unit)? = null
    var onRoundCompleted: ((Int) -> Unit)? = null
    var onTrainingComplete: (() -> Unit)? = null
    var onConnectionChanged: ((Boolean, String?) -> Unit)? = null

    fun connect(serverUrl: String, clientId: String, token: String) {
        manualDisconnect = false
        retryAttempt = 0
        connection = Connection(serverUrl, clientId, token)
        reconnectJob?.cancel()
        webSocket?.close(1000, "Reconnecting")
        openConnection()
    }

    private fun openConnection() {
        val current = connection ?: return
        val wsUrl = current.serverUrl
            .replace("http://", "ws://")
            .replace("https://", "wss://")

        val request = Request.Builder()
            .url("$wsUrl/ws/feed?client_id=${current.clientId}")
            .header("X-FGT-Token", current.token)
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                retryAttempt = 0
                onConnectionChanged?.invoke(true, null)
                heartbeatJob?.cancel()
                heartbeatJob = scope.launch {
                    while (isActive && !manualDisconnect) {
                        delay(15_000)
                        webSocket.send("{\"type\":\"heartbeat\"}")
                    }
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                val json = JSONObject(text)
                val data = json.optJSONObject("data")
                when (json.optString("type")) {
                    "update_requested" -> {
                        val round = data?.getInt("round") ?: return
                        val config = data.optJSONObject("config")
                        val roundConfig = FLRoundConfig(
                            learningRate = config?.optDouble("lr", 0.05)?.toFloat() ?: 0.05f,
                            localEpochs = config?.optInt("local_epochs", 1) ?: 1,
                            mu = config?.optDouble("mu", 0.01)?.toFloat() ?: 0.01f,
                            dpEpsilon = config?.optDouble("dp_epsilon", 0.0)?.toFloat() ?: 0.0f,
                            dpDelta = config?.optDouble("dp_delta", 1e-5)?.toFloat() ?: 1e-5f,
                            maxGradNorm = config?.optDouble("max_grad_norm", 1.0)?.toFloat() ?: 1.0f,
                            pseudoLabelThreshold = config?.optDouble("pseudo_label_threshold", 0.8)?.toFloat() ?: 0.8f,
                            pseudoLabelWeight = config?.optDouble("pseudo_label_weight", 0.25)?.toFloat() ?: 0.25f,
                            minLocalSamples = config?.optInt("min_local_samples", 8) ?: 8,
                        )
                        onUpdateRequested?.invoke(round, roundConfig)
                    }
                    "round_completed" -> {
                        val round = data?.getInt("round") ?: return
                        onRoundCompleted?.invoke(round)
                    }
                    "training_complete" -> {
                        onTrainingComplete?.invoke()
                    }
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                handleDisconnect("Connection closed ($code): $reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                handleDisconnect(response?.let { "WebSocket rejected (${it.code})" } ?: (t.message ?: "Connection failed"))
            }
        })
    }

    private fun handleDisconnect(reason: String) {
        heartbeatJob?.cancel()
        onConnectionChanged?.invoke(false, reason)
        if (manualDisconnect || reconnectJob?.isActive == true) return
        val delayMillis = minOf(30_000L, 1_000L shl retryAttempt.coerceAtMost(5))
        retryAttempt++
        reconnectJob = scope.launch {
            delay(delayMillis)
            if (!manualDisconnect) openConnection()
        }
    }

    fun disconnect() {
        manualDisconnect = true
        reconnectJob?.cancel()
        heartbeatJob?.cancel()
        webSocket?.close(1000, null)
        webSocket = null
        onConnectionChanged?.invoke(false, null)
    }
}
