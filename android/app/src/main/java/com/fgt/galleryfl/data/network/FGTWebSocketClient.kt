package com.fgt.galleryfl.data.network

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject

class FGTWebSocketClient(private val client: OkHttpClient) {
    private var webSocket: WebSocket? = null
    var onUpdateRequested: ((Int, Float, Int) -> Unit)? = null
    var onRoundCompleted: ((Int) -> Unit)? = null
    var onTrainingComplete: (() -> Unit)? = null

    fun connect(serverUrl: String, clientId: String) {
        val wsUrl = serverUrl
            .replace("http://", "ws://")
            .replace("https://", "wss://")

        val request = Request.Builder()
            .url("$wsUrl/ws/feed?client_id=$clientId")
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                val json = JSONObject(text)
                val data = json.optJSONObject("data")
                when (json.optString("type")) {
                    "update_requested" -> {
                        val round = data?.getInt("round") ?: return
                        val config = data.optJSONObject("config")
                        val lr = config?.optDouble("lr", 0.05)?.toFloat() ?: 0.05f
                        val epochs = config?.optInt("local_epochs", 3) ?: 3
                        onUpdateRequested?.invoke(round, lr, epochs)
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

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                android.util.Log.e("FGTWebSocket", "Connection failed: ${t.message}")
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, null)
        webSocket = null
    }
}
