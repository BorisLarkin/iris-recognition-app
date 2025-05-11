package com.example.irisrecognition.ui

import IrisDatabase
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.YuvImage
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.core.content.ContextCompat
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.layout.Arrangement
import com.example.irisrecognition.detection.FaceDetector
import com.example.irisrecognition.detection.IrisDetector
import com.example.irisrecognition.detection.models.Face
import com.example.irisrecognition.detection.models.Iris
import com.example.irisrecognition.ui.components.CameraPreview
import com.example.irisrecognition.ui.theme.IrisRecognitionTheme
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import timber.log.Timber
import java.util.concurrent.Executors
import androidx.compose.material3.Text
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.graphics.Rect as Rect_andr
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity() {
    private lateinit var faceDetector: FaceDetector
    private lateinit var irisDetector: IrisDetector
    private val irisDatabase = IrisDatabase()
    private val executor = Executors.newSingleThreadExecutor()
    private var isOpenCvInitialized = false
    private var isFrontCamera = false
    private var showNameInputDialog by mutableStateOf(false)
    private var tempIrisFeatures: FloatArray? = null
    private var cameraSwitchInProgress by mutableStateOf(false)
    private var showIrisResultDialog by mutableStateOf(false)
    private var irisDetectionResult by mutableStateOf<String?>(null)
    private var isProcessingFrame by mutableStateOf(false)
    private var lastProcessedBitmap by mutableStateOf<Bitmap?>(null)


    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Timber.e("Camera permission denied")
            showError("Camera permission is required")
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!hasCameraPermission()) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
        initializeOpenCv()

        setContent {
            IrisRecognitionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (isOpenCvInitialized) {
                        AppContent()
                    } else {
                        Column(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.Center,
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            CircularProgressIndicator()
                            Spacer(modifier = Modifier.height(16.dp))
                            Text("Initializing OpenCV...")
                        }
                    }
                }
            }
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    @Composable
    private fun AppContent() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val coroutineScope = rememberCoroutineScope()

        var faces by remember { mutableStateOf(emptyList<Face>()) }
        var irisPairs by remember { mutableStateOf(emptyList<Iris>()) }
        var recognizedUser by remember { mutableStateOf<String?>(null) }
        var previewSize by remember { mutableStateOf(Size(1f, 1f)) }
        var imageSize by remember { mutableStateOf(Size(1f, 1f)) }


        val cameraController = remember {
            LifecycleCameraController(context).apply {
                setEnabledUseCases(
                    CameraController.IMAGE_CAPTURE or
                            CameraController.IMAGE_ANALYSIS or
                            CameraController.VIDEO_CAPTURE
                )
                cameraSelector = if (isFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
            }
        }

        // Update preview size when camera is initialized
        LaunchedEffect(cameraController) {
            withContext(Dispatchers.Main) {
                previewSize = Size(
                    previewSize.width,
                    previewSize.height
                )
            }
            cameraController.bindToLifecycle(lifecycleOwner)
        }

        suspend fun processFrameForIrisDetection() {
            withContext(Dispatchers.IO) {
                try {
                    irisPairs.firstOrNull()?.let { iris ->
                        iris.leftIris?.let { irisData ->
                            val matchResult = irisDatabase.findBestMatch(irisData.features)
                            irisDetectionResult = if (matchResult.first != null && matchResult.second >= 0.8f) {
                                recognizedUser = matchResult.first
                                "User recognized: ${matchResult.first} (${(matchResult.second * 100).toInt()}%)"
                            } else {
                                tempIrisFeatures = irisData.features
                                showNameInputDialog = true
                                "New user detected"
                            }
                        } ?: run {
                            irisDetectionResult = "Error: Iris not detected"
                        }
                    } ?: run {
                        irisDetectionResult = "Error: No Irises detected"
                    }
                } catch (e: Exception) {
                    Timber.e(e, "Error processing frame for iris detection")
                    irisDetectionResult = "Error: ${e.localizedMessage}"
                }
            }
        }


        fun createAnalyzer(
            onResult: (List<Face>, Size, List<Iris>) -> Unit
        ): ImageAnalysis.Analyzer {
            return object : ImageAnalysis.Analyzer {
                @OptIn(ExperimentalGetImage::class)
                override fun analyze(image: ImageProxy) {
                    try {
                        if (frameCounter++ % 3 != 0) {
                            image.close()
                            return
                        }

                        val rotationDegrees = image.imageInfo.rotationDegrees
                        val rotationMatrix = Matrix().apply {
                            when (rotationDegrees) {
                                90 -> postRotate(90f)
                                180 -> postRotate(180f)
                                270 -> postRotate(270f)
                            }
                            if (isFrontCamera) {
                                postScale(-1f, 1f)
                            }
                        }

                        val bitmap = image.toBitmap(rotationMatrix)
                        val mat = Mat()
                        Utils.bitmapToMat(bitmap, mat)

                        // Detect faces first
                        faceDetector.detectFaces(mat) { detectedFaces ->
                            // Then detect irises in the full image
                            irisDetector.detectIris(bitmap) { iris ->
                                onResult(
                                    detectedFaces,
                                    Size(image.width.toFloat(), image.height.toFloat()),
                                    listOf(iris)
                                )
                            }
                        }
                    } catch (e: Exception) {
                        Timber.e(e, "Error analyzing image")
                    } finally {
                        image.close()
                    }
                }
            }
        }

        LaunchedEffect(isFrontCamera) {
            if (cameraSwitchInProgress) return@LaunchedEffect

            cameraSwitchInProgress = true
            try {
                irisPairs = emptyList() // Clear detections
                recognizedUser = null // Clear recognition
                // Simply update the camera selector
                cameraController.cameraSelector = if (isFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }

                // No need to clear/restart analyzer unless you're experiencing issues
            } catch (e: Exception) {
                Timber.e(e, "Camera switch failed")
            } finally {
                cameraSwitchInProgress = false
            }
        }

        // Keep your existing analyzer setup in a separate LaunchedEffect
        LaunchedEffect(Unit) {
            val analyzer = createAnalyzer { detectedFaces, currentImageSize, irises ->
                faces = detectedFaces
                imageSize = currentImageSize
                irisPairs = irises
            }

            cameraController.setImageAnalysisAnalyzer(executor, analyzer)

            // Add this to handle camera controller lifecycle
            awaitCancellation()
        }


        CameraPreview(
            cameraController = cameraController,
            faces = faces,
            irisPairs = irisPairs,
            recognizedUser = recognizedUser,
            previewSize = previewSize,
            imageSize = imageSize,
            onSwitchCamera = {
                isFrontCamera = !isFrontCamera
                irisPairs = emptyList() // Clear previous iris detections
                recognizedUser = null // Clear previous recognition
                cameraController.cameraSelector = if (isFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
            },
            onCapture = {
                if (faces.isEmpty()) {
                    showIrisResultDialog = true
                    irisDetectionResult = "Error: No face detected"
                } else if (isProcessingFrame) {
                    showIrisResultDialog = true
                    irisDetectionResult = "Already processing a frame"
                } else {
                    isProcessingFrame = true
                    showIrisResultDialog = true
                    irisDetectionResult = "Scanning iris..."

                    coroutineScope.launch {
                        try {
                            // Get fresh frame and process it
                            val frame = getLatestFrame(cameraController)
                            if (frame != null) {
                                lastProcessedBitmap = frame

                                // Perform fresh iris detection on this frame
                                irisDetector.detectIris(frame) { iris ->
                                    irisPairs = listOf(iris)

                                    launch {
                                        processFrameForIrisDetection()
                                    }
                                }
                            } else {
                                irisDetectionResult = "Error: Could not capture frame"
                            }
                        } catch (e: Exception) {
                            irisDetectionResult = "Error: ${e.localizedMessage}"
                        } finally {
                            isProcessingFrame = false
                        }
                    }
                }
            },
            isScanning = showIrisResultDialog && irisDetectionResult == "Scanning iris..."
        )

        if (showIrisResultDialog) {
            AlertDialog(
                onDismissRequest = { showIrisResultDialog = false },
                title = { Text("Iris Detection") },
                text = { Text(irisDetectionResult ?: "Unknown error") },
                confirmButton = {
                    Button(onClick = { showIrisResultDialog = false }) {
                        Text("OK")
                    }
                }
            )
        }

        if (showNameInputDialog) {
            var name by remember { mutableStateOf("") }

            AlertDialog(
                onDismissRequest = { showNameInputDialog = false },
                title = { Text("New User Detected") },
                text = {
                    Column {
                        Text("Please enter your name:")
                        OutlinedTextField(
                            value = name,
                            onValueChange = { name = it },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                },
                confirmButton = {
                    Button(
                        onClick = {
                            tempIrisFeatures?.let { features ->
                                if (name.isNotBlank()) {
                                    irisDatabase.addUser(features, name)
                                    recognizedUser = name
                                } else {
                                    irisDatabase.addUser(features, "Unknown User")
                                    recognizedUser = "Unknown User"
                                }
                            }
                            showNameInputDialog = false
                        }
                    ) {
                        Text("Save")
                    }
                },
                dismissButton = {
                    Button(
                        onClick = { showNameInputDialog = false }
                    ) {
                        Text("Cancel")
                    }
                }
            )
        }
    }

    private var frameCounter = 0

    // Add this extension function
    private fun ImageProxy.toBitmap(rotationMatrix: Matrix? = null): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect_andr(0, 0, yuvImage.width, yuvImage.height), 100, outputStream)
        val jpegArray = outputStream.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(jpegArray, 0, jpegArray.size)

        return if (rotationMatrix != null) {
            Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.width,
                bitmap.height,
                rotationMatrix,
                true
            )
        } else {
            bitmap
        }
    }

    // Update the getLatestFrame function
    private suspend fun getLatestFrame(cameraController: LifecycleCameraController): Bitmap? {
        return withContext(Dispatchers.IO) {
            try {
                val latch = CountDownLatch(1)
                var capturedBitmap: Bitmap? = null

                // Create a temporary analyzer to capture a single frame
                val analyzer = ImageAnalysis.Analyzer { imageProxy ->
                    try {
                        val rotationMatrix = Matrix().apply {
                            when (imageProxy.imageInfo.rotationDegrees) {
                                90 -> postRotate(90f)
                                180 -> postRotate(180f)
                                270 -> postRotate(270f)
                            }
                            if (isFrontCamera) {
                                postScale(-1f, 1f)
                            }
                        }
                        capturedBitmap = imageProxy.toBitmap(rotationMatrix)
                    } catch (e: Exception) {
                        Timber.e(e, "Error converting image to bitmap")
                    } finally {
                        imageProxy.close()
                        latch.countDown()
                    }
                }

                // Set the analyzer on the main thread
                withContext(Dispatchers.Main) {
                    cameraController.setImageAnalysisAnalyzer(executor, analyzer)
                }

                // Wait for the frame to be captured (with timeout)
                if (!latch.await(1, TimeUnit.SECONDS)) {
                    Timber.w("Timeout waiting for frame capture")
                    return@withContext null
                }

                // Clear the analyzer on the main thread
                withContext(Dispatchers.Main) {
                    cameraController.clearImageAnalysisAnalyzer()
                }

                capturedBitmap
            } catch (e: Exception) {
                Timber.e(e, "Error capturing frame")
                null
            }
        }
    }

    private fun initializeOpenCv() {
        try {
            Timber.d("Initializing OpenCV...")
            if (OpenCVLoader.initDebug()) {
                Timber.d("OpenCV initialized successfully")
                faceDetector = FaceDetector(this).also {
                    Timber.d("FaceDetector created")
                }
                irisDetector = IrisDetector(this).also {
                    Timber.d("IrisDetector created")
                }
                isOpenCvInitialized = true
            } else {
                Timber.e("OpenCV initialization failed!")
                showError("OpenCV initialization failed")
                finish()
            }
        } catch (e: Exception) {
            Timber.e(e, "Error initializing OpenCV")
            showError("Error: ${e.localizedMessage}")
            finish()
        }
    }

    private fun showError(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::faceDetector.isInitialized) {
            faceDetector.close()
        }
        executor.shutdown()
    }
}