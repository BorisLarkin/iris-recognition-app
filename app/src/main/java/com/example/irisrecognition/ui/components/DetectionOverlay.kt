package com.example.irisrecognition.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import com.example.irisrecognition.detection.models.Face
import com.example.irisrecognition.detection.models.Iris
import timber.log.Timber
import kotlin.math.min

@Composable
fun DetectionOverlay(
    faces: List<Face>,
    irisPairs: List<Iris>,
    modifier: Modifier = Modifier,
    imageWidth: Int,  // Bitmap width after rotation
    imageHeight: Int, // Bitmap height after rotation
    isFrontCamera: Boolean = false
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        // Get the actual preview viewport dimensions
        val previewWidth = size.width
        val previewHeight = size.height

        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val canvasAspect = size.width / size.height

        // Calculate the scale factor between bitmap and preview
        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (canvasAspect > imageAspect) {
            // Letterbox (wide canvas)
            scale = size.height / imageHeight.toFloat()
            offsetX = (size.width - imageWidth * scale) / 2
            offsetY = 0f
        } else {
            // Pillarbox (tall canvas)
            scale = size.width / imageWidth.toFloat()
            offsetX = 0f
            offsetY = (size.height - imageHeight * scale) / 2
        }


        // Simple transform that just scales and centers
        fun transformPoint(x: Float, y: Float): Offset {
            val screenX = x * scale + offsetX
            val screenY = y * scale + offsetY
            Timber.log(previewHeight.toInt(), "Height")
            return if (!isFrontCamera) {
                // Mirror only the X coordinate for front camera
                Offset(previewWidth - screenX, previewHeight - screenY)
            } else {
                Offset(screenX, screenY)
            }
        }

        // Draw faces
        faces.forEach { face ->
            val topLeft = transformPoint(face.rect.x.toFloat(), face.rect.y.toFloat())
            val bottomRight = transformPoint(
                (face.rect.x + face.rect.width).toFloat(),
                (face.rect.y + face.rect.height).toFloat()
            )

            drawRect(
                color = Color.Green.copy(alpha = 0.3f),
                topLeft = topLeft,
                size = Size(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y),
                style = Stroke(width = 4f)
            )
        }

        // Draw irises
        irisPairs.forEach { iris ->
            iris.leftIris?.let { irisData ->
                val center = transformPoint(
                    irisData.center.x.toFloat()-30,
                    irisData.center.y.toFloat()
                )
                drawIrisCircle(center, irisData.radius * scale, Color.Red)
            }

            iris.rightIris?.let { irisData ->
                val center = transformPoint(
                    irisData.center.x.toFloat()+50,
                    irisData.center.y.toFloat()
                )
                drawIrisCircle(center, irisData.radius * scale, Color.Blue)
            }
        }
    }
}

private fun DrawScope.drawIrisCircle(
    center: Offset,
    radius: Float,
    color: Color
) {
    // Outer circle (thicker)
    drawCircle(
        color = color.copy(alpha = 0.5f),
        center = center,
        radius = radius,
        style = Stroke(width = 4f)
    )

    // Center point
    drawCircle(
        color = color,
        center = center,
        radius = 8f
    )
}

