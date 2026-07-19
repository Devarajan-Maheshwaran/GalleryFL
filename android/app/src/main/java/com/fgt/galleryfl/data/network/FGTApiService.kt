package com.fgt.galleryfl.data.network

import okhttp3.ResponseBody
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.POST

interface FGTApiService {
    @POST("api/register")
    suspend fun register(
        @Header("X-FGT-Token") token: String,
        @Body request: RegisterRequest
    ): RegisterResponse

    @GET("api/model/current")
    suspend fun getCurrentModel(
        @retrofit2.http.Query("client_version") clientVersion: Int = 0
    ): retrofit2.Response<ResponseBody>

    @POST("api/training/submit-update")
    suspend fun submitUpdate(
        @Header("X-FGT-Token") token: String,
        @Body update: ClientUpdateRequest
    ): UpdateResponse

    @GET("api/training/status")
    suspend fun getTrainingStatus(): TrainingStatusResponse

    @GET("api/model/status")
    suspend fun getModelStatus(): ModelStatusResponse

    @POST("api/taxonomy/signal")
    suspend fun postTagSignal(
        @Header("X-FGT-Token") token: String,
        @Body request: TagSignalRequest
    ): TagSignalResponse
}

data class ModelStatusResponse(
    val loaded: Boolean,
    val model_version: Int,
    val head_present: Boolean,
    val min_clients: Int
)

data class RegisterRequest(
    val device_model: String,
    val nickname: String,
    val client_id: String? = null
)

data class RegisterResponse(
    val client_id: String,
    val status: String,
    val model_version: Int
)

data class ClientUpdateRequest(
    val client_id: String,
    val weights: String,
    val num_samples: Int,
    val local_loss: Float,
    val local_accuracy: Float,
    val round: Int,
    val base_model_version: Int
)

data class UpdateResponse(
    val status: String
)

data class TrainingStatusResponse(
    val is_training: Boolean,
    val current_round: Int,
    val max_rounds: Int,
    val model_version: Int
)

data class TagSignalRequest(
    val client_id: String? = null,
    val tags: List<String>
)

data class TagSignalResponse(
    val ok: Boolean,
    val received: Int,
    val recorded: Int,
    val client_id: String? = null
)
