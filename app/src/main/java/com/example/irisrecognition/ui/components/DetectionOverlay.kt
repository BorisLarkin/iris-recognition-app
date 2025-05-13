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
    imageWidth: Int,
    imageHeight: Int,
    isFrontCamera: Boolean = false
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        // Calculate aspect ratio preserving scale
        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val canvasAspect = size.width / size.height

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

        // Unified coordinate transform (no rotation needed)
        fun transformPoint(x: Float, y: Float): Offset {
            val screenX = x * scale + offsetX
            val screenY = y * scale + offsetY
            return if (isFrontCamera) {
                Offset(size.width - screenX, screenY) // Mirror only X for front camera
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
                    irisData.center.x.toFloat(),
                    irisData.center.y.toFloat()
                ).let { original ->
                    // Manual adjustments
                    original.copy(
                        y = original.y - 20f,  // Move up by 15 pixels
                        x = original.x + 30f   // Move left by 10 pixels
                    )
                }
                drawIrisCircle(center, irisData.radius * scale, Color.Red)
            }

            iris.rightIris?.let { irisData ->
                val center = transformPoint(
                    irisData.center.x.toFloat(),
                    irisData.center.y.toFloat()
                ).let { original ->
                    // Manual adjustments
                    original.copy(
                        y = original.y - 20f,  // Move up by 15 pixels
                        x = original.x - 100f   // Move right by 10 pixels
                    )
                }
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

