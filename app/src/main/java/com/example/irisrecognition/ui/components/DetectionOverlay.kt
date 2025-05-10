package com.example.irisrecognition.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
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
    val density = LocalDensity.current
    val strokeWidth2dp = with(density) { 2.dp.toPx() }
    val radius4dp = with(density) { 4.dp.toPx() }
    val strokeWidth1dp = with(density) { 1.dp.toPx() }
    val radius2dp = with(density) { 2.dp.toPx() }

    Canvas(modifier = modifier.fillMaxSize()) {
        // Calculate scale factors while maintaining aspect ratio
        val scaleX = size.width / imageWidth.toFloat()
        val scaleY = size.height / imageHeight.toFloat()
        val scale = minOf(scaleX, scaleY)

        // Calculate offset to center the image
        val offsetX = (size.width - imageWidth * scale) / 2
        val offsetY = (size.height - imageHeight * scale) / 2

        faces.forEach { face ->
            // Apply scaling and offset to face rectangle
            val left = face.rect.x * scale + offsetX
            val top = face.rect.y * scale + offsetY
            val width = face.rect.width * scale
            val height = face.rect.height * scale

            drawRect(
                color = Color.Green.copy(alpha = 0.5f),
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(width, height),
                style = Stroke(width = strokeWidth2dp)
            )

            // Apply same transformation to landmarks
            face.landmarks.forEach { point ->
                val x = point.x * scale + offsetX
                val y = point.y * scale + offsetY

                drawCircle(
                    color = Color.Yellow.copy(alpha = 0.8f),
                    center = Offset(x.toFloat(), y.toFloat()),
                    radius = radius4dp
                )
            }
        }

        irisPairs.forEach { iris ->
            iris.leftIris?.let { irisData ->
                // Draw left iris
                drawIris(
                    drawScope = this,
                    center = Offset(
                        (irisData.center.x * scale + offsetX).toFloat(),
                        (irisData.center.y * scale + offsetY).toFloat()
                    ),
                    radius = (irisData.radius * scale).coerceAtLeast(5f),
                    color = Color(0xFF4CAF50), // Green
                    strokeWidth2dp = strokeWidth2dp,
                    radius2dp = radius2dp,
                    strokeWidth1dp = strokeWidth1dp
                )
            }

            iris.rightIris?.let { irisData ->
                // Draw right iris
                drawIris(
                    drawScope = this,
                    center = Offset(
                        (irisData.center.x * scale + offsetX).toFloat(),
                        (irisData.center.y * scale + offsetY).toFloat()
                    ),
                    radius = (irisData.radius * scale).coerceAtLeast(5f),
                    color = Color(0xFF2196F3), // Blue
                    strokeWidth2dp = strokeWidth2dp,
                    radius2dp = radius2dp,
                    strokeWidth1dp = strokeWidth1dp
                )
            }
        }
    }
}

private fun drawIris(
    center: Offset,
    radius: Float,
    drawScope: DrawScope,
    color: Color,
    strokeWidth2dp: Float,
    radius2dp: Float,
    strokeWidth1dp: Float
) {
    // Draw iris circle
    drawScope.drawCircle(
        color = color.copy(alpha = 0.3f),
        center = center,
        radius = radius,
        style = Stroke(width = strokeWidth2dp)
    )

    // Draw iris center point
    drawScope.drawCircle(
        color = color,
        center = center,
        radius = radius2dp * 2f
    )

    // Draw crosshair
    drawScope.drawLine(
        color = color,
        start = Offset(center.x - radius, center.y),
        end = Offset(center.x + radius, center.y),
        strokeWidth = strokeWidth1dp
    )

    drawScope.drawLine(
        color = color,
        start = Offset(center.x, center.y - radius),
        end = Offset(center.x, center.y + radius),
        strokeWidth = strokeWidth1dp
    )
}
