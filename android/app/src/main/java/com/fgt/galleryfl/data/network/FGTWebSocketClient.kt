package com.fgt.galleryfl.data.network

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject

class FGTWebSocketClient(private val client: OkHttpClient) {
    private var webSocket: WebSocket? = null
    var onUpdateRequested: ((Int) -> Unit)? = null
    var onRoundCompleted: ((Int) -> Unit)? = null

    fun connect(serverUrl: String, clientId: String) {
        val request = Request.Builder()
            .url("$serverUrl/ws/feed?client_id=$clientId")
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                println("WebSocket Connected")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                val json = JSONObject(text)
                when (json.optString("type")) {
                    "update_requested" -> {
                        val round = json.getJSONObject("data").getInt("round")
                        onUpdateRequested?.invoke(round)
                    }
                    "round_completed" -> {
                        val round = json.getJSONObject("data").getInt("round")
                        onRoundCompleted?.invoke(round)
                    }
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                println("WebSocket Error: ${t.message}")
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "App closed")
    }
}
