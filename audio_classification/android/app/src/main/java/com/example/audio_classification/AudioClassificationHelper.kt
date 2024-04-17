/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.audio_classification

import android.content.Context
import android.media.AudioRecord
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.core.BaseOptions

class AudioClassificationHelper(
    private val context: Context,
    var currentModel: TFLiteModel = DEFAULT_MODEL,
    var classificationThreshold: Float = DISPLAY_THRESHOLD,
    var overlap: Float = DEFAULT_OVERLAP_VALUE,
    var numOfResults: Int = DEFAULT_NUM_OF_RESULTS,
    var currentDelegate: Delegate = DEFAULT_DELEGATE,
    var numThreads: Int = 2
) {
    private var classifier: AudioClassifier? = null
    private var tensorAudio: TensorAudio? = null
    private var recorder: AudioRecord? = null

    /** As the result of sound classification, this value emits map of probabilities */
    val probabilities: SharedFlow<Result<Pair<List<Category>, Long>>>
        get() = _probabilities
    private val _probabilities = MutableSharedFlow<Result<Pair<List<Category>, Long>>>()

    suspend fun initClassifier() {
        // Set general detection options, e.g. number of used threads

        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU.
        // Possible to also use a GPU delegate, but this requires that the classifier be created
        // on the same thread that is using the classifier, which is outside of the scope of this
        // sample's design.
        when (currentDelegate) {
            Delegate.CPU -> {
                // Default
            }

            Delegate.NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        // Configures a set of parameters for the classifier and what results will be returned.
        val options = AudioClassifier.AudioClassifierOptions.builder()
            .setScoreThreshold(classificationThreshold)
            .setMaxResults(numOfResults)
            .setBaseOptions(baseOptionsBuilder.build())
            .build()

        try {
            // Create the classifier and required supporting objects
            classifier =
                AudioClassifier.createFromFileAndOptions(context, currentModel.fileName, options)
            tensorAudio = classifier?.createInputTensorAudio()
            recorder = classifier?.createAudioRecord()
            startAudioClassification()
        } catch (e: Exception) {
            _probabilities.emit(Result.failure(e))
            Log.e("AudioClassification", "TFLite failed to load with error: " + e.message)
        }
    }

    private var job: Job? = null

    private suspend fun startAudioClassification() {
        if (classifier == null) {
            return
        }
        withContext(Dispatchers.IO) {
            if (recorder?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                return@withContext
            }
            recorder?.startRecording()

            // Each model will expect a specific audio recording length. This formula calculates that
            // length using the input buffer size and tensor format sample rate.
            // For example, YAMNET expects 0.975 second length recordings.
            // This needs to be in milliseconds to avoid the required Long value dropping decimals.
            job = launch {
                classifyAudio()
            }

            val lengthInMilliSeconds = ((classifier!!.requiredInputBufferSize * 1.0f) /
                    classifier!!.requiredTensorAudioFormat.sampleRate) * 1000

            while (isActive) {
                delay((lengthInMilliSeconds * (1 - overlap)).toLong())
            }

        }
    }

    private suspend fun classifyAudio() {
        coroutineScope {
            while (isActive) {
                tensorAudio?.load(recorder)
                var inferenceTime = SystemClock.uptimeMillis()
                if (classifier?.isClosed == false) {
                    val output = classifier?.classify(tensorAudio)
                    inferenceTime = SystemClock.uptimeMillis() - inferenceTime
                    if (!output.isNullOrEmpty()) {
                        _probabilities.emit(
                            Result.success(
                                Pair(
                                    output[0].categories,
                                    inferenceTime
                                )
                            )
                        )
                    }
                }
                delay(300)
            }
        }
    }

    fun stopAudioClassification() {
        recorder?.stop()
        classifier?.close()
        job?.cancel()
    }

    companion object {
        val DEFAULT_DELEGATE = Delegate.CPU
        const val DISPLAY_THRESHOLD = 0.3f
        const val DEFAULT_NUM_OF_RESULTS = 2
        const val DEFAULT_OVERLAP_VALUE = 0.5f
        val DEFAULT_MODEL = TFLiteModel.YAMNET
    }

    enum class Delegate {
        CPU,
        NNAPI
    }

    enum class TFLiteModel(val fileName: String) {
        YAMNET("yamnet.tflite"),
        SpeechCommand("speech.tflite")
    }
}