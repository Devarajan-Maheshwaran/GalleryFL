package com.fgt.galleryfl.data.network

import okhttp3.ResponseBody
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path

interface FGTApiService {
    @POST("api/register")
    suspend fun register(@Body request: RegisterRequest): RegisterResponse

    @GET("api/model/current")
    suspend fun getCurrentModel(): ResponseBody

    @POST("api/training/submit-update")
    suspend fun submitUpdate(@Body update: ClientUpdateRequest): UpdateResponse

    @GET("api/training/status")
    suspend fun getTrainingStatus(): TrainingStatusResponse
}

data class RegisterRequest(
    val device_model: String,
    val nickname: String
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
    val local_accuracy: Float
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
