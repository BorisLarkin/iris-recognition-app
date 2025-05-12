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
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import com.example.irisrecognition.detection.models.Face
import com.example.irisrecognition.detection.models.Iris

@Composable
fun DetectionOverlay(
    faces: List<Face>,
    irisPairs: List<Iris>,
    modifier: Modifier = Modifier,
    previewWidth: Float,
    previewHeight: Float,
    imageWidth: Int,
    imageHeight: Int,
    rotationDegrees: Int = 0,
    isFrontCamera: Boolean = false
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        // Calculate aspect ratios
        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val previewAspect = size.width / size.height

        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (previewAspect > imageAspect) {
            // Preview is wider than image (letterbox)
            scale = size.height / imageHeight.toFloat()
            offsetX = (size.width - imageWidth * scale) / 2
            offsetY = 0f
        } else {
            // Preview is taller than image (pillarbox)
            scale = size.width / imageWidth.toFloat()
            offsetX = 0f
            offsetY = (size.height - imageHeight * scale) / 2
        }

        // Function to transform coordinates based on rotation
        fun transformPoint(x: Float, y: Float): Offset {
            val rotated = when (rotationDegrees) {
                90 -> Offset(y, imageHeight - x)
                180 -> Offset(imageWidth - x, imageHeight - y)
                270 -> Offset(imageWidth - y, x)
                else -> Offset(x, y)
            }

            // Mirror for front camera
            val mirrored = if (isFrontCamera) {
                Offset(imageWidth - rotated.x, rotated.y)
            } else {
                rotated
            }

            return Offset(
                mirrored.x * scale + offsetX,
                mirrored.y * scale + offsetY
            )
        }

        faces.forEach { face ->
            val faceLeft = face.rect.x
            val faceTop = face.rect.y
            val faceRight = faceLeft + face.rect.width
            val faceBottom = faceTop + face.rect.height

            val topLeft = transformPoint(faceLeft.toFloat(), faceTop.toFloat())
            val bottomRight = transformPoint(faceRight.toFloat(), faceBottom.toFloat())

            drawRect(
                color = Color.Green.copy(alpha = 0.3f),
                topLeft = topLeft,
                size = Size(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y),
                style = Stroke(width = 4f)
            )
        }

        irisPairs.forEach { iris ->
            iris.leftIris?.let { irisData ->
                val center = transformPoint(irisData.center.x.toFloat(), irisData.center.y.toFloat())
                drawIrisCircle(
                    center = center,
                    radius = irisData.radius * scale,
                    color = Color.Red
                )
            }

            iris.rightIris?.let { irisData ->
                val center = transformPoint(irisData.center.x.toFloat(), irisData.center.y.toFloat())
                drawIrisCircle(
                    center = center,
                    radius = irisData.radius * scale,
                    color = Color.Blue
                )
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

