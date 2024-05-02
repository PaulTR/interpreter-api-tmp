package com.google.edgeai.examples.image_segmentation

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val imageSegmentationHelper: ImageSegmentationHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageSegmentationHelper = ImageSegmentationHelper(context)
                return MainViewModel(imageSegmentationHelper) as T
            }
        }
    }

    init {
        viewModelScope.launch {
            imageSegmentationHelper.initClassifier()
        }
    }

    private var segmentJob: Job? = null

    private val segmentationUiShareFlow = MutableStateFlow<Pair<ImageInfo?, Long>>(
        Pair(null, 0L)
    ).also { flow ->
        viewModelScope.launch {
            imageSegmentationHelper.segmentation.filter { it.segmentation.masks.isNotEmpty() }.map {
                val segmentation = it.segmentation
                val maskTensor = segmentation.masks[0]
                val maskArray = maskTensor.buffer.array()
                val pixels = IntArray(maskArray.size)

                val colorLabels = segmentation.coloredLabels.mapIndexed { index, coloredLabel ->
                    ColorLabel(
                        index, coloredLabel.getlabel(), coloredLabel.argb
                    )
                }
                // Set color for pixels
                for (i in maskArray.indices) {
                    val colorLabel = colorLabels[maskArray[i].toInt()]
                    val color = colorLabel.getColor()
                    pixels[i] = color
                }
                // Get image info
                val width = maskTensor.width
                val height = maskTensor.height
                val imageInfo = ImageInfo(
                    pixels = pixels, width = width, height = height
                )

                val inferenceTime = it.inferenceTime
                Pair(imageInfo, inferenceTime)
            }.collect {
                flow.emit(it)
            }
        }
    }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            imageSegmentationHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        segmentationUiShareFlow,
        errorMessage,
    ) { segmentationUiPair, error ->
        UiState(
            imageInfo = segmentationUiPair.first,
            inferenceTime = segmentationUiPair.second,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())


    /** Start segment an image.
     *  @param imageProxy contain `imageBitMap` and imageInfo as `image rotation degrees`.
     *
     */
    fun segment(imageProxy: ImageProxy) {
        segmentJob = viewModelScope.launch {
            imageSegmentationHelper.segment(
                imageProxy.toBitmap(),
                imageProxy.imageInfo.rotationDegrees,
            )
            imageProxy.close()
        }
    }

    /** Start segment an image.
     *  @param bitmap Tries to make a new bitmap based on the dimensions of this bitmap,
     *  setting the new bitmap's config to Bitmap.Config.ARGB_8888
     *  @param rotationDegrees to correct the rotationDegrees during segmentation
     */
    fun segment(bitmap: Bitmap, rotationDegrees: Int) {
        viewModelScope.launch {
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            imageSegmentationHelper.segment(argbBitmap, rotationDegrees)
        }
    }

    /** Stop current segmentation */
    fun stopSegment() {
        segmentJob?.cancel()
        viewModelScope.launch {
            segmentationUiShareFlow.emit(Pair(null, 0L))
        }
    }

    /** Set Delegate for ImageSegmentationHelper(CPU/NNAPI)*/
    fun setDelegate(delegate: ImageSegmentationHelper.Delegate) {
        viewModelScope.launch {
            imageSegmentationHelper.initClassifier(delegate)
        }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}