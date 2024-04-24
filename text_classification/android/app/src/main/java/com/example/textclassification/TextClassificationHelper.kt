package com.example.textclassification

import android.content.Context
import android.util.Log
import android.widget.MultiAutoCompleteTextView.Tokenizer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.IntBuffer

class TextClassificationHelper(private val context: Context) {
    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null

    fun setupInterpreter() {
        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, "mobile_bert.tflite")
            Log.i(TAG, "Done creating TFLite buffer from mobile_bert.tflite")
            Interpreter(tfliteBuffer, Interpreter.Options().apply {})
        } catch (e: IOException) {
            return
        } catch (e: IllegalArgumentException) {
            return
        }
    }

    fun run() {
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return

        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

        Log.d(TAG, "run: ${inputShape[1]} -- ${outputShape[1]}")

        val inputBuffer = IntBuffer.allocate(inputShape[1])
        val outputBuffer = FloatBuffer.allocate(outputShape[1])

        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter?.run(inputBuffer, outputBuffer)
        Log.d(TAG, "run: ${outputBuffer.array().contentToString()}")
    }

    companion object {
        private const val TAG = "TextClassifier"
    }

}