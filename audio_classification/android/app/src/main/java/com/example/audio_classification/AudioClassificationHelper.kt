/*
 * Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.sin

/**
 * Performs classification on sound.
 *
 * <p>The API supports models which accept sound input via {@code AudioRecord} and one classification output tensor.
 * The output of the recognition is emitted as LiveData of Map.
 *
 */
class AudioClassificationHelper(private val context: Context, val options: Options = Options()) {
    class Options(
        /** The required audio sample rate in Hz.  */
        val sampleRate: Int = 44_100,
        /** How many milliseconds to sleep between successive audio sample pulls.  */
        val audioPullPeriod: Long = 50L,
        /** Number of warm up runs to do after loading the TFLite model.  */
        val warmupRuns: Int = 3,
        /** Number of points in average to reduce noise. */
        val pointsInAverage: Int = 10,
        /** Overlap factor of recognition period */
        var overlapFactor: Float = DEFAULT_OVERLAP,
        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        var probabilityThreshold: Float = DEFAULT_PROBABILITY_THRESHOLD,
        /** The enum contains the .tflite file name, relative to the assets/ directory */
        var currentModel: TFLiteModel = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations*/
        var delegate: Delegate = DEFAULT_DELEGATE,
        /** Number of output classes of the TFLite model.  */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Number of threads to be used for ops that support multi-threading.
         * threadCount>= -1. Setting numThreads to 0 has the effect of disabling multithreading,
         * which is equivalent to setting numThreads to 1. If unspecified, or set to the value -1,
         * the number of threads used will be implementation-defined and platform-dependent.
         * */
        var threadCount: Int = DEFAULT_THREAD_COUNT
    )

    /** As the result of sound classification, this value emits map of probabilities */
    val probabilities: SharedFlow<Pair<List<Category>, Long>>
        get() = _probabilities
    private val _probabilities = MutableSharedFlow<Pair<List<Category>, Long>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    /** Overlap factor of recognition period */
    private var overlapFactor: Float
        get() = options.overlapFactor
        set(value) {
            options.overlapFactor = value.also {
                recognitionPeriod = (1000L * (1 - value)).toLong()
            }
        }

    private var audioRecord: AudioRecord? = null

    /** How many milliseconds between consecutive model inference calls.  */
    private var recognitionPeriod = (1000L * (1 - overlapFactor)).toLong()

    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null

    /** Audio length (in # of PCM samples) required by the TFLite model.  */
    private var modelInputLength = 0

    /** Number of output classes of the TFLite model.  */
    private var modelNumClasses = 0

    /** Used to hold the real-time probabilities predicted by the model for the output classes.  */
    private lateinit var predictionProbs: FloatArray

    /** Latest prediction latency in milliseconds.  */
    private var latestPredictionLatencyMs = 0f

    private var recordingOffset = 0
    private lateinit var recordingBuffer: ShortArray

    /** Buffer that holds audio PCM sample that are fed to the TFLite model for inference.  */
    private lateinit var inputBuffer: FloatBuffer

    private var job: Job? = null

    /** Stop, cancel or reset all necessary variable*/
    fun stop() {
        audioRecord?.stop()
        audioRecord = null
        job?.cancel()
        interpreter?.resetVariableTensors()
        interpreter?.close()
        interpreter = null
    }

    suspend fun setupInterpreter() {
        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, options.currentModel.fileName)
            Log.i(TAG, "Done creating TFLite buffer from ${options.currentModel}")
            Interpreter(tfliteBuffer, Interpreter.Options().apply {
                numThreads = options.threadCount
                //TODO: NNAPI crash on YAMNET model
                useNNAPI = options.delegate == Delegate.NNAPI
            })
        } catch (e: IOException) {
            _error.emit(Throwable(message = "Failed to load TFLite model - ${e.message}"))
            return
        } catch (e: IllegalArgumentException) {
            _error.emit(Throwable("Failed to create Interpreter - ${e.message}"))
            return
        }

        // Inspect input and output specs.
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        /**
         * YAMNET input: float32[15600]
         * Speech Command input: float32[1,44032]
         */
        modelInputLength = inputShape[if (options.currentModel == TFLiteModel.YAMNET) 0 else 1]

        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return
        modelNumClasses = outputShape[1]
        // Fill the array with NaNs initially.
        predictionProbs = FloatArray(modelNumClasses) { Float.NaN }

        inputBuffer = FloatBuffer.allocate(modelInputLength)

        generateDummyAudioInput(inputBuffer)
        for (n in 0 until options.warmupRuns) {
            val t0 = SystemClock.elapsedRealtimeNanos()

            // Create input and output buffers.
            val outputBuffer = FloatBuffer.allocate(modelNumClasses)
            inputBuffer.rewind()
            outputBuffer.rewind()
            interpreter?.run(inputBuffer, outputBuffer)

            Log.i(
                TAG, "Switches: Done calling interpreter.run(): %s (%.6f ms)".format(
                    outputBuffer.array().contentToString(),
                    (SystemClock.elapsedRealtimeNanos() - t0) / NANOS_IN_MILLIS
                )
            )
        }

    }

    private fun generateDummyAudioInput(inputBuffer: FloatBuffer) {
        val twoPiTimesFreq = 2 * Math.PI.toFloat() * 1000f
        for (i in 0 until modelInputLength) {
            val x = i.toFloat() / (modelInputLength - 1)
            inputBuffer.put(i, sin(twoPiTimesFreq * x.toDouble()).toFloat())
        }
    }

    private suspend fun startRecognition(labels: List<String>) {
        coroutineScope {
            if (modelInputLength <= 0) {
                Log.e(TAG, "Switches: Cannot start recognition because model is unavailable.")
                return@coroutineScope
            }
            val outputBuffer = FloatBuffer.allocate(modelNumClasses)
            while (isActive) {
                var samplesAreAllZero = true

                var j = (recordingOffset - modelInputLength) % modelInputLength
                if (j < 0) {
                    j += modelInputLength
                }

                for (i in 0 until modelInputLength) {
                    if (!isActive) {
                        return@coroutineScope
                    }

                    val s = if (i >= options.pointsInAverage && j >= options.pointsInAverage) {
                        ((j - options.pointsInAverage + 1)..j).map { recordingBuffer[it % modelInputLength] }
                            .average()
                    } else {
                        recordingBuffer[j % modelInputLength]
                    }
                    j += 1

                    if (samplesAreAllZero && s.toInt() != 0) {
                        samplesAreAllZero = false
                    }
                    inputBuffer.put(i, s.toFloat())
                }
                if (samplesAreAllZero) {
                    Log.w(TAG, "No audio input: All audio samples are zero!")
                    continue
                }
                val t0 = SystemClock.elapsedRealtimeNanos()
                inputBuffer.rewind()
                outputBuffer.rewind()
                interpreter?.run(inputBuffer, outputBuffer)
                latestPredictionLatencyMs =
                    ((SystemClock.elapsedRealtimeNanos() - t0) / NANOS_IN_MILLIS).toFloat()
                outputBuffer.rewind()
                outputBuffer.get(predictionProbs) // Copy data to predictionProbs.

                val probList = predictionProbs.map {
                    /** Scores in range 0..1.0 for each of the output classes. */
                    if (it < options.probabilityThreshold) 0f else it
                }

                val categories = labels.zip(probList).map {
                    Category(label = it.first, score = it.second)
                }.sortedByDescending { it.score }.take(options.resultCount)

                _probabilities.emit(
                    Pair(
                        categories, latestPredictionLatencyMs.toLong()
                    )
                )

                delay(recognitionPeriod)
            }
        }
    }

    /*
     * Starts sound classification, which triggers running on IO Thread
     */
    @SuppressLint("MissingPermission")
    suspend fun startRecord() {
        withContext(Dispatchers.IO) {
            val labels = loadLabels(currentModel = options.currentModel)
            var bufferSize = AudioRecord.getMinBufferSize(
                options.sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
                bufferSize = options.sampleRate * 2
                Log.w(TAG, "bufferSize has error or bad value")
            }
            Log.i(TAG, "bufferSize = $bufferSize")
            audioRecord = AudioRecord(
                // including MIC, UNPROCESSED, and CAMCORDER.
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                options.sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                return@withContext
            }
            Log.i(TAG, "Successfully initialized AudioRecord")
            val bufferSamples = bufferSize / 2
            val audioBuffer = ShortArray(bufferSamples)
            val recordingBufferSamples =
                ceil(modelInputLength.toFloat() / bufferSamples.toDouble()).toInt() * bufferSamples
            Log.i(TAG, "recordingBufferSamples = $recordingBufferSamples")
            recordingOffset = 0
            recordingBuffer = ShortArray(recordingBufferSamples)
            audioRecord?.startRecording()
            Log.i(TAG, "Successfully started AudioRecord recording")

            // Start recognition (model inference) coroutine.
            job = launch {
                startRecognition(labels = labels)
            }

            while (isActive) {
                delay(options.audioPullPeriod)

                when (audioRecord?.read(audioBuffer, 0, audioBuffer.size)) {
                    AudioRecord.ERROR_INVALID_OPERATION -> {
                        Log.w(TAG, "AudioRecord.ERROR_INVALID_OPERATION")
                    }

                    AudioRecord.ERROR_BAD_VALUE -> {
                        Log.w(TAG, "AudioRecord.ERROR_BAD_VALUE")
                    }

                    AudioRecord.ERROR_DEAD_OBJECT -> {
                        Log.w(TAG, "AudioRecord.ERROR_DEAD_OBJECT")
                    }

                    AudioRecord.ERROR -> {
                        Log.w(TAG, "AudioRecord.ERROR")
                    }

                    bufferSamples -> {
                        // We apply locks here to avoid two separate threads (the recording and
                        // recognition threads) reading and writing from the recordingBuffer at the same
                        // time, which can cause the recognition thread to read garbled audio snippets.
                        audioBuffer.copyInto(
                            recordingBuffer, recordingOffset, 0, bufferSamples
                        )
                        recordingOffset = (recordingOffset + bufferSamples) % recordingBufferSamples
                    }

                }
            }
        }
    }

    companion object {
        private const val TAG = "SoundClassifier"

        /** Number of nanoseconds in a millisecond  */
        private const val NANOS_IN_MILLIS = 1_000_000.toDouble()
        val DEFAULT_MODEL = TFLiteModel.YAMNET
        val DEFAULT_DELEGATE = Delegate.CPU
        const val DEFAULT_THREAD_COUNT = 2
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_OVERLAP = 0.8f
        const val DEFAULT_PROBABILITY_THRESHOLD = 0.3f
    }

    enum class Delegate {
        CPU, NNAPI
    }

    enum class TFLiteModel(val fileName: String, val labelFile: String) {
        /** Yamnet labels: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv*/
        YAMNET(
            "yamnet.tflite",
            "yamnet_label.txt",
        ),

        /** Speech command labels: https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition */
        SpeechCommand(
            "speech.tflite",
            "speech_label.txt",
        )
    }

    /** Retrieve labels from "labels.txt" file */
    private fun loadLabels(currentModel: TFLiteModel): List<String> {
        return try {
            val reader =
                BufferedReader(InputStreamReader(context.assets.open(currentModel.labelFile)))
            val wordList = mutableListOf<String>()
            reader.useLines { lines ->
                wordList.addAll(lines.toList())
            }
            wordList
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read model ${currentModel.labelFile}: ${e.message}")
            emptyList()
        }
    }
}

data class Category(val label: String, val score: Float)
