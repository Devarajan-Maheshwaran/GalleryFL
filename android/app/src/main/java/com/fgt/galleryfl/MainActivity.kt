package com.fgt.galleryfl

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.fgt.galleryfl.ui.theme.FGTColors
import com.fgt.galleryfl.ui.components.NeuSurface
import com.fgt.galleryfl.data.network.FGTWebSocketClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.util.UUID

class MainActivity : ComponentActivity() {
    private val httpClient = OkHttpClient()
    private val wsClient = FGTWebSocketClient(httpClient)
    private val defaultServerUrl = "http://10.0.2.2:8080"
    private val clientId = UUID.randomUUID().toString()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                MainScreen(wsClient, httpClient, defaultServerUrl, clientId)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        wsClient.disconnect()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    wsClient: FGTWebSocketClient,
    httpClient: OkHttpClient,
    defaultServerUrl: String,
    clientId: String
) {
    var statusText by remember { mutableStateOf("Ready to connect") }
    var isConnected by remember { mutableStateOf(false) }
    var currentServerUrl by remember { mutableStateOf(defaultServerUrl) }
    var accessCode by remember { mutableStateOf("dev-token-change-me") }
    var currentRound by remember { mutableIntStateOf(0) }
    var isTraining by remember { mutableStateOf(false) }

    val scope = rememberCoroutineScope()

    DisposableEffect(Unit) {
        wsClient.onUpdateRequested = { round ->
            currentRound = round
            isTraining = true
            statusText = "Round $round: Training locally..."
            scope.launch(Dispatchers.IO) {
                try {
                    val apiService = com.fgt.galleryfl.data.network.RetrofitClient.getApiService(currentServerUrl, httpClient)
                    val response = apiService.getCurrentModel()
                    val encodedWeights = response.string()
                    val globalWeights = com.fgt.galleryfl.data.network.WeightSerializer.deserialize(encodedWeights)
                    val numClasses = globalWeights[2].size / 256

                    val featuresList = List(50) { FloatArray(1024) { kotlin.random.Random.nextFloat() * 2 - 1 } }
                    val targetsList = List(50) {
                        FloatArray(numClasses) { if (kotlin.random.Random.nextFloat() > 0.9f) 1f else 0f }
                    }

                    val trainer = com.fgt.galleryfl.data.ml.LocalTrainer(
                        com.fgt.galleryfl.data.ml.ClassificationHead(numClasses)
                    )
                    val result = trainer.train(featuresList, targetsList, globalWeights, epochs = 3)

                    val serializedUpdate = com.fgt.galleryfl.data.network.WeightSerializer.serialize(result.updatedWeights)
                    val updateRequest = com.fgt.galleryfl.data.network.ClientUpdateRequest(
                        client_id = clientId,
                        weights = serializedUpdate,
                        num_samples = result.numSamples,
                        local_loss = result.localLoss,
                        local_accuracy = result.localAccuracy
                    )
                    apiService.submitUpdate(accessCode, updateRequest)

                    withContext(Dispatchers.Main) {
                        statusText = "Round $round: Update submitted"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        statusText = "Error: ${e.message}"
                        isTraining = false
                    }
                }
            }
        }
        wsClient.onRoundCompleted = { round ->
            isTraining = false
            statusText = "Round $round complete"
        }
        wsClient.onTrainingComplete = {
            isTraining = false
            statusText = "Training session complete"
        }
        onDispose {
            wsClient.disconnect()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(FGTColors.BgBase)
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Federated Gallery Tags",
                style = MaterialTheme.typography.headlineMedium,
                color = FGTColors.TextPrimary
            )

            Spacer(modifier = Modifier.height(32.dp))

            OutlinedTextField(
                value = currentServerUrl,
                onValueChange = { currentServerUrl = it },
                label = { Text("Server URL", color = FGTColors.TextSecondary) },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = FGTColors.AccentPrimary,
                    unfocusedBorderColor = FGTColors.TextSecondary,
                    focusedTextColor = FGTColors.TextPrimary,
                    unfocusedTextColor = FGTColors.TextPrimary
                ),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(12.dp))

            OutlinedTextField(
                value = accessCode,
                onValueChange = { accessCode = it },
                label = { Text("Access Code", color = FGTColors.TextSecondary) },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = FGTColors.AccentPrimary,
                    unfocusedBorderColor = FGTColors.TextSecondary,
                    focusedTextColor = FGTColors.TextPrimary,
                    unfocusedTextColor = FGTColors.TextPrimary
                ),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            NeuSurface {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = if (currentRound > 0) "Round $currentRound" else "Status",
                        style = MaterialTheme.typography.titleMedium,
                        color = FGTColors.TextSecondary
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (isTraining) FGTColors.AccentGold else FGTColors.AccentPrimary
                    )
                    if (isTraining) {
                        Spacer(modifier = Modifier.height(8.dp))
                        LinearProgressIndicator(
                            color = FGTColors.AccentPrimary,
                            trackColor = FGTColors.BgSurface,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = {
                    scope.launch(Dispatchers.IO) {
                        try {
                            val apiService = com.fgt.galleryfl.data.network.RetrofitClient.getApiService(currentServerUrl, httpClient)
                            val regResp = apiService.register(
                                accessCode,
                                com.fgt.galleryfl.data.network.RegisterRequest(
                                    device_model = android.os.Build.MODEL,
                                    nickname = "Android-${clientId.take(6)}"
                                )
                            )
                            withContext(Dispatchers.Main) {
                                wsClient.connect(currentServerUrl, regResp.client_id)
                                isConnected = true
                                statusText = "Registered. Waiting for training..."
                            }
                        } catch (e: Exception) {
                            withContext(Dispatchers.Main) {
                                statusText = "Connection failed: ${e.message}"
                            }
                        }
                    }
                },
                enabled = !isConnected,
                colors = ButtonDefaults.buttonColors(
                    containerColor = FGTColors.AccentPrimary,
                    disabledContainerColor = FGTColors.BgSurface
                )
            ) {
                Text(
                    text = if (isConnected) "Connected" else "Join Training",
                    color = if (isConnected) FGTColors.TextSecondary else FGTColors.BgBase
                )
            }
        }
    }
}
