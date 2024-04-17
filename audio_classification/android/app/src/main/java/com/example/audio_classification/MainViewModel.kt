package com.example.audio_classification

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val audioClassificationHelper: AudioClassificationHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val audioClassificationHelper = AudioClassificationHelper(context)
                return MainViewModel(audioClassificationHelper) as T
            }
        }
    }

    private var job: Job? = null

    private val settingFlow = MutableStateFlow<Setting?>(null)
        .apply {
            viewModelScope.launch {
                collect {
                    if (it == null) {
                        return@collect
                    }
                    stopClassifier()
                    audioClassificationHelper.currentModel = it.model
                    audioClassificationHelper.currentDelegate = it.delegate
                    audioClassificationHelper.overlap = it.overlap
                    audioClassificationHelper.numOfResults = it.maxResults
                    audioClassificationHelper.classificationThreshold = it.threshold
                    audioClassificationHelper.numThreads = it.threadCount
                    if (job?.isActive == false) {
                        job?.cancel()
                    }
                    job = launch {
                        audioClassificationHelper.initClassifier()
                    }
                }
            }
        }
    private val probabilitiesFlow = audioClassificationHelper.probabilities.distinctUntilChanged()
        .map {
            it.fold(
                onSuccess = { pair ->
                    pair
                },
                onFailure = {
                    Pair(emptyList(), 0L)
                }
            )
        }

    val uiState: StateFlow<UiState> =
        combine(settingFlow.filterNotNull(), probabilitiesFlow) { setting, probabilities ->
            UiState(
                classifications = probabilities.first,
                setting = setting.copy(inferenceTime = probabilities.second),
            )
        }.stateIn(
            viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState()
        )

    fun startAudioClassification() {
        settingFlow.update { Setting() }
    }

    fun setTFLiteModel(model: AudioClassificationHelper.TFLiteModel) {
        viewModelScope.launch {
            settingFlow.update { it?.copy(model = model) }
        }
    }

    fun setDelegate(delegate: AudioClassificationHelper.Delegate) {
        viewModelScope.launch {
            settingFlow.update { it?.copy(delegate = delegate) }
        }
    }

    fun setOverlap(value: Float) {
        viewModelScope.launch {
            settingFlow.update { it?.copy(overlap = value) }
        }
    }

    fun setMaxResults(max: Int) {
        viewModelScope.launch {
            settingFlow.update { it?.copy(maxResults = max) }
        }
    }

    fun setThreshold(threshold: Float) {
        viewModelScope.launch {
            settingFlow.update { it?.copy(threshold = threshold) }
        }
    }

    fun setThreadCount(count: Int) {
        viewModelScope.launch {
            settingFlow.update { it?.copy(threadCount = count) }
        }
    }

    fun stopClassifier() {
        audioClassificationHelper.stopAudioClassification()
    }
}
