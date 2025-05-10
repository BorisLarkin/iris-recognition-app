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
    imageHeight: Int
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

        // Draw face rectangle
        faces.forEach { face ->
            val faceLeft = face.rect.x * scale + offsetX
            val faceTop = face.rect.y * scale + offsetY
            val faceWidth = face.rect.width * scale
            val faceHeight = face.rect.height * scale

            drawRect(
                color = Color.Green.copy(alpha = 0.3f),
                topLeft = Offset(faceLeft, faceTop),
                size = Size(faceWidth, faceHeight),
                style = Stroke(width = 4f)
            )

            // Draw eye landmarks
            face.landmarks.forEachIndexed { index, point ->
                val x = point.x * scale + offsetX
                val y = point.y * scale + offsetY
                val color = if (index == 0) Color.Red else Color.Blue

                drawCircle(
                    color = color.copy(alpha = 0.8f),
                    center = Offset(x.toFloat(), y.toFloat()),
                    radius = 12f
                )
            }
        }

        // Draw iris markers
        irisPairs.forEach { iris ->
            iris.leftIris?.let { irisData ->
                drawIrisMarker(
                    center = Offset(
                        (irisData.center.x * scale + offsetX).toFloat(),
                        (irisData.center.y * scale + offsetY).toFloat()
                    ),
                    radius = irisData.radius * scale,
                    color = Color.Red
                )
            }

            iris.rightIris?.let { irisData ->
                drawIrisMarker(
                    center = Offset(
                        (irisData.center.x * scale + offsetX).toFloat(),
                        (irisData.center.y * scale + offsetY).toFloat()
                    ),
                    radius = irisData.radius * scale,
                    color = Color.Blue
                )
            }
        }
    }
}

private fun DrawScope.drawIrisMarker(
    center: Offset,
    radius: Float,
    color: Color,
    showCrosshair: Boolean = false // Add this parameter
) {
    // Outer circle
    drawCircle(
        color = color.copy(alpha = 0.3f),
        center = center,
        radius = radius,
        style = Stroke(width = 3f)
    )

    // Inner dot
    drawCircle(
        color = color,
        center = center,
        radius = 6f
    )

    // Only draw crosshair if showCrosshair is true
    if (showCrosshair) {
        drawLine(
            color = color,
            start = Offset(center.x - radius, center.y),
            end = Offset(center.x + radius, center.y),
            strokeWidth = 2f
        )
        drawLine(
            color = color,
            start = Offset(center.x, center.y - radius),
            end = Offset(center.x, center.y + radius),
            strokeWidth = 2f
        )
    }
}
