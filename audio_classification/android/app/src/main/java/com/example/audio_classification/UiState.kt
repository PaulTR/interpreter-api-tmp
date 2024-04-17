package com.example.audio_classification

import androidx.compose.runtime.Immutable
import com.example.audio_classification.AudioClassificationHelper.Companion.DEFAULT_NUM_OF_RESULTS
import com.example.audio_classification.AudioClassificationHelper.Companion.DEFAULT_OVERLAP_VALUE
import com.example.audio_classification.AudioClassificationHelper.Companion.DISPLAY_THRESHOLD
import org.tensorflow.lite.support.label.Category

@Immutable
data class UiState(
    val classifications: List<Category> = listOf(), val setting: Setting = Setting()
)

@Immutable
data class Setting(
    val inferenceTime: Long = 0L,
    val model: AudioClassificationHelper.TFLiteModel = AudioClassificationHelper.TFLiteModel.YAMNET,
    val delegate: AudioClassificationHelper.Delegate = AudioClassificationHelper.Delegate.CPU,
    val overlap: Float = DEFAULT_OVERLAP_VALUE,
    val maxResults: Int = DEFAULT_NUM_OF_RESULTS,
    val threshold: Float = DISPLAY_THRESHOLD,
    val threadCount: Int = 2
)

