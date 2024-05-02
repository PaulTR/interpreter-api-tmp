package com.google.edgeai.examples.image_segmentation

import android.graphics.Color
import androidx.compose.runtime.Immutable

@Immutable
class UiState(
    val imageInfo: ImageInfo? = null,
    val inferenceTime: Long = 0L,
    val errorMessage: String? = null,
)

class ImageInfo(
    val pixels: IntArray,
    val width: Int,
    val height: Int,
)

@Immutable
data class ColorLabel(
    val id: Int,
    val label: String,
    val rgbColor: Int,
) {

    fun getColor(): Int {
        // Use completely transparent for the background color.
        return if (id == 0) Color.TRANSPARENT else Color.argb(
            128,
            Color.red(rgbColor),
            Color.green(rgbColor),
            Color.blue(rgbColor)
        )
    }
}