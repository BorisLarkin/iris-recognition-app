package com.example.irisrecognition.ui.components

import android.view.ViewGroup
import androidx.camera.core.CameraSelector
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.filled.Face
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.remember
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
    isScanning: Boolean,
    currentRotation : Int,
    isFrontCamera : Boolean
) {
    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { context ->
                PreviewView(context).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            },
            modifier = Modifier.fillMaxSize(),
            update = { previewView ->
                previewView.controller = cameraController
            }
        )

        DetectionOverlay(
            faces = faces,
            irisPairs = irisPairs,
            imageWidth = imageSize.width.toInt(),
            imageHeight = imageSize.height.toInt(),
            isFrontCamera = isFrontCamera
        )

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(
                    onClick = {
                        // Directly trigger the camera flip without any additional logic
                        onSwitchCamera()
                    },
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
                    .size(120.dp)
                    .background(Color.Black.copy(alpha = 0.7f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator(color = Color.White)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Scanning Iris", color = Color.White)
                }
            }
        }
    }
}
