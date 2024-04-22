package com.example.audio_classification

import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
    val classifications: List<Category> = listOf(),
    val setting: Setting = Setting(),
    val errorMessage: String? = null,
)

@Immutable
data class Setting(
    val inferenceTime: Long = 0L,
    val model: AudioClassificationHelper.TFLiteModel = AudioClassificationHelper.DEFAULT_MODEL,
    val delegate: AudioClassificationHelper.Delegate = AudioClassificationHelper.DEFAULT_DELEGATE,
    val overlap: Float = AudioClassificationHelper.DEFAULT_OVERLAP,
    val resultCount: Int = AudioClassificationHelper.DEFAULT_RESULT_COUNT,
    val threshold: Float = AudioClassificationHelper.DEFAULT_PROBABILITY_THRESHOLD,
    val threadCount: Int = AudioClassificationHelper.DEFAULT_THREAD_COUNT
)
