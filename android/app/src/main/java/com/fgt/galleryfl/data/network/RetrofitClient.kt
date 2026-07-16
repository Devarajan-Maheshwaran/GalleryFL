package com.fgt.galleryfl.data.network

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory

object RetrofitClient {
    private var retrofit: Retrofit? = null

    fun getClient(baseUrl: String, okHttpClient: OkHttpClient): Retrofit {
        if (retrofit == null || retrofit?.baseUrl().toString() != baseUrl.removeSuffix("/") + "/") {
            retrofit = Retrofit.Builder()
                .baseUrl(baseUrl.removeSuffix("/") + "/")
                .client(okHttpClient)
                .addConverterFactory(MoshiConverterFactory.create())
                .build()
        }
        return retrofit!!
    }

    fun getApiService(baseUrl: String, okHttpClient: OkHttpClient): FGTApiService {
        return getClient(baseUrl, okHttpClient).create(FGTApiService::class.java)
    }
}
