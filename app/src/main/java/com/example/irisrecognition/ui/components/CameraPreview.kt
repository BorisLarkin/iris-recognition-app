package com.example.irisrecognition.ui.components

import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.filled.Face
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.ui.geometry.Size
import com.example.irisrecognition.detection.models.Face
import com.example.irisrecognition.detection.models.Iris


@Composable
fun CameraPreview(
    cameraController: LifecycleCameraController,
    faces: List<Face>,
    irisPairs: List<Iris>,
    recognizedUser: String?,
    previewSize: Size,
    imageSize: Size,
    onSwitchCamera: () -> Unit,
    onCapture: () -> Unit,
    modifier: Modifier = Modifier,
    isScanning: Boolean
) {
    Box(modifier = modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    controller = cameraController
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        DetectionOverlay(
            faces = faces,
            irisPairs = irisPairs,
            previewWidth = previewSize.width,
            previewHeight = previewSize.height,
            imageWidth = imageSize.width.toInt(),
            imageHeight = imageSize.height.toInt()
        )

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp)
        ) {
            if (recognizedUser != null) {
                RecognitionResult(user = recognizedUser)
            }



            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(
                    onClick = onSwitchCamera,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Face,
                        contentDescription = "Switch Camera"
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Flip")
                }

                Spacer(modifier = Modifier.width(16.dp))

                Button(
                    onClick = onCapture,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Scan Iris"
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Scan")
                }
            }
        }

        if (isScanning) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(100.dp)
                    .background(Color.Black.copy(alpha = 0.5f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = Color.White)
            }
        }
    }
}

@Composable
fun RecognitionResult(user: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Default.Check,
            contentDescription = "Verified",
            tint = Color.Green
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(text = "User: $user")
    }
}
