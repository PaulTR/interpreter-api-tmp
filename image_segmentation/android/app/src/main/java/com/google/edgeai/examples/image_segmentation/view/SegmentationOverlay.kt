package com.google.edgeai.examples.image_segmentation.view

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import com.google.edgeai.examples.image_segmentation.ImageInfo

@Composable
fun SegmentationOverlay(modifier: Modifier = Modifier, imageInfo: ImageInfo) {
    Canvas(
        modifier = modifier
    ) {
        val size = this.size
        val imageWidth: Float = size.width
        val imageHeight: Float = size.height

        // Create the mask bitmap with colors and the set of detected labels.
        // We only need the first mask for this sample because we are using
        // the OutputType CATEGORY_MASK, which only provides a single mask.
        val image = Bitmap.createBitmap(
            imageInfo.pixels, imageInfo.width, imageInfo.height, Bitmap.Config.ARGB_8888
        )

        val scaleBitmap =
            Bitmap.createScaledBitmap(image, imageWidth.toInt(), imageHeight.toInt(), true)
        drawImage(scaleBitmap.asImageBitmap())

    }
}