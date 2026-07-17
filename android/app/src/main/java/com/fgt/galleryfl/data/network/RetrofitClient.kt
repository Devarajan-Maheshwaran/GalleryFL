package com.fgt.galleryfl.data.network

import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory

object RetrofitClient {
    private var retrofit: Retrofit? = null

    private val moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build()

    fun getClient(baseUrl: String, okHttpClient: OkHttpClient): Retrofit {
        if (retrofit == null || retrofit?.baseUrl().toString() != baseUrl.removeSuffix("/") + "/") {
            retrofit = Retrofit.Builder()
                .baseUrl(baseUrl.removeSuffix("/") + "/")
                .client(okHttpClient)
                .addConverterFactory(MoshiConverterFactory.create(moshi))
                .build()
        }
        return retrofit!!
    }

    fun getApiService(baseUrl: String, okHttpClient: OkHttpClient): FGTApiService {
        return getClient(baseUrl, okHttpClient).create(FGTApiService::class.java)
    }
}
