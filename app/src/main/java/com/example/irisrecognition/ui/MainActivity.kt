package com.example.irisrecognition.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
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
import androidx.camera.view.PreviewView
import com.example.irisrecognition.detection.FaceDetector
import com.example.irisrecognition.detection.IrisDetector
import com.example.irisrecognition.detection.models.Face
import com.example.irisrecognition.detection.models.Iris
import com.example.irisrecognition.detection.models.IrisData
import com.example.irisrecognition.ui.components.CameraPreview
import com.example.irisrecognition.ui.theme.IrisRecognitionTheme
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import timber.log.Timber
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.math.pow
import androidx.compose.material3.Text
import androidx.camera.core.ImageInfo
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.geometry.Size
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Rect
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
    var showIrisResultDialog by mutableStateOf(false)
    var irisDetectionResult by mutableStateOf<String?>(null)

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeOpenCv()
        } else {
            Timber.e("Camera permission denied")
            showError("Camera permission is required")
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!hasCameraPermission()) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            initializeOpenCv()
        }

        setContent {
            IrisRecognitionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (isOpenCvInitialized) {
                        AppContent()
                    } else {
                        Text("Initializing...")
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

        var faces by remember { mutableStateOf(emptyList<Face>()) }
        var irisPairs by remember { mutableStateOf(emptyList<Iris>()) }
        var recognizedUser by remember { mutableStateOf<String?>(null) }
        var previewSize by remember { mutableStateOf(Size(1f, 1f)) }
        var imageSize by remember { mutableStateOf(Size(1f, 1f)) }

        val cameraController = remember {
            LifecycleCameraController(context).apply {
                // Add both IMAGE_ANALYSIS AND PREVIEW
                setEnabledUseCases(CameraController.IMAGE_ANALYSIS or CameraController.VIDEO_CAPTURE)
            }
        }

        // Update preview size when camera is initialized
        LaunchedEffect(cameraController) {
            withContext(Dispatchers.Main) {
                val cameraInfo = cameraController.cameraInfo
                previewSize = Size(
                    previewSize.width.toFloat(),
                    previewSize.height.toFloat()
                )
            }
            cameraController.bindToLifecycle(lifecycleOwner)
        }

        // Separate suspending function for face processing
        suspend fun processDetectedFace(
            detectedFaces: List<Face>,
            cameraController: LifecycleCameraController
        ) {
            detectedFaces.firstOrNull()?.let { face ->
                val bitmap = withContext(Dispatchers.IO) {
                    getLatestFrame(cameraController)?.let { fullBitmap ->
                        // Expand face rectangle to ensure we capture entire eye region
                        val expandedRect = Rect(
                            (face.rect.x - face.rect.width * 0.2).toInt().coerceAtLeast(0),
                            (face.rect.y - face.rect.height * 0.1).toInt().coerceAtLeast(0),
                            (face.rect.width * 1.4).toInt().coerceAtMost(fullBitmap.width),
                            (face.rect.height * 1.2).toInt().coerceAtMost(fullBitmap.height)
                        )
                        extractFaceBitmap(fullBitmap, expandedRect)
                    }
                }

                bitmap?.let { faceBitmap ->
                    // Ensure bitmap is large enough for iris detection
                    val minSize = 128
                    val resizedBitmap = if (faceBitmap.width < minSize || faceBitmap.height < minSize) {
                        Bitmap.createScaledBitmap(
                            faceBitmap,
                            faceBitmap.width.coerceAtLeast(minSize),
                            faceBitmap.height.coerceAtLeast(minSize),
                            true
                        )
                    } else {
                        faceBitmap
                    }

                    irisDetector.detectIris(resizedBitmap) { iris ->
                        irisPairs = listOf(iris)
                        if (iris.leftIris == null && iris.rightIris == null) {
                            showIrisResultDialog = true
                            irisDetectionResult = "Iris not detected - ensure eyes are visible"
                        }
                    }
                }
            } ?: run {
                irisPairs = emptyList()
            }
        }

        LaunchedEffect(isFrontCamera) {
            if (cameraSwitchInProgress) return@LaunchedEffect
            cameraSwitchInProgress = true
            try {
                cameraController.cameraSelector = if (isFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
                cameraController.bindToLifecycle(lifecycleOwner)
            } catch (e: Exception) {
                Timber.e(e, "Camera switch failed")
            } finally {
                cameraSwitchInProgress = false
            }
        }

        LaunchedEffect(Unit) {
            cameraController.setImageAnalysisAnalyzer(
                executor,
                createAnalyzer { detectedFaces, currentImageSize ->
                    faces = detectedFaces
                    imageSize = currentImageSize

                    // Launch a new coroutine for the face processing
                    launch {
                        processDetectedFace(detectedFaces, cameraController)
                    }
                }
            )
        }

        CameraPreview(
            cameraController = cameraController,
            faces = faces,
            irisPairs = irisPairs,
            recognizedUser = recognizedUser,
            previewSize = previewSize,
            imageSize = imageSize,
            onSwitchCamera = { isFrontCamera = !isFrontCamera },
            onCapture = {
                if (faces.isEmpty()) {
                    showIrisResultDialog = true
                    irisDetectionResult = "Error: No face detected"
                } else {
                    showIrisResultDialog = true
                    irisDetectionResult = "Scanning iris..."
                    irisPairs.firstOrNull()?.let { iris ->
                        iris.leftIris?.let { irisData ->
                            val bestMatch = irisDatabase.findBestMatch(irisData.features)
                            if (bestMatch != null) {
                                recognizedUser = bestMatch
                                irisDetectionResult = "User recognized: $bestMatch"
                            } else {
                                tempIrisFeatures = irisData.features
                                showNameInputDialog = true
                                irisDetectionResult = "New user detected"
                            }
                        } ?: run {
                            irisDetectionResult = "Error: Iris not detected"
                        }
                    } ?: run {
                        irisDetectionResult = "Error: No iris data available"
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
            AlertDialog(
                onDismissRequest = { showNameInputDialog = false },
                title = { Text("New User Detected") },
                text = {
                    var name by remember { mutableStateOf("") }
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
                                irisDatabase.addUser(features, "Unknown User")
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

    private fun createAnalyzer(
        onResult: (List<Face>, Size) -> Unit
    ): ImageAnalysis.Analyzer {
        return object : ImageAnalysis.Analyzer {
            @OptIn(ExperimentalGetImage::class)
            override fun analyze(image: ImageProxy) {
                try {
                    // Process only every 3rd frame to reduce load
                    if (frameCounter++ % 3 != 0) {
                        image.close()
                        return
                    }

                    val bitmap = image.toBitmap()
                    val mat = Mat()
                    Utils.bitmapToMat(bitmap, mat)

                    faceDetector.detectFaces(mat) { faces ->
                        onResult(faces, Size(image.width.toFloat(), image.height.toFloat()))
                    }
                } catch (e: Exception) {
                    Timber.e(e, "Error analyzing image")
                } finally {
                    image.close()
                }
            }
        }
    }

    private suspend fun getLatestFrame(cameraController: LifecycleCameraController): Bitmap? {
        return try {
            var capturedBitmap: Bitmap? = null
            val latch = CountDownLatch(1)

            val analyzer = ImageAnalysis.Analyzer { imageProxy ->
                try {
                    capturedBitmap = imageProxy.toBitmap()
                } finally {
                    imageProxy.close()
                    latch.countDown()
                }
            }

            withContext(Dispatchers.Main) {
                cameraController.setImageAnalysisAnalyzer(executor, analyzer)
            }

            latch.await(500, TimeUnit.MILLISECONDS)

            withContext(Dispatchers.Main) {
                cameraController.clearImageAnalysisAnalyzer()
            }

            capturedBitmap
        } catch (e: Exception) {
            Timber.e(e, "Error capturing frame")
            null
        }
    }

    private fun extractFaceBitmap(bitmap: Bitmap?, faceRect: Rect): Bitmap? {
        if (bitmap == null) return null

        val x = faceRect.x.coerceAtLeast(0)
        val y = faceRect.y.coerceAtLeast(0)
        val width = faceRect.width.coerceAtMost(bitmap.width - x)
        val height = faceRect.height.coerceAtMost(bitmap.height - y)

        return try {
            Bitmap.createBitmap(bitmap, x, y, width, height)
        } catch (e: Exception) {
            Timber.e(e, "Error extracting face bitmap")
            null
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

class IrisDatabase {
    private val database = mutableMapOf<String, FloatArray>()

    fun addUser(features: FloatArray, name: String) {
        database[name] = features
    }

    fun findBestMatch(features: FloatArray): String? {
        if (database.isEmpty()) return null

        var minDistance = Float.MAX_VALUE
        var bestMatch: String? = null

        database.forEach { (userId, dbFeatures) ->
            val distance = features.zip(dbFeatures).sumOf {
                (it.first - it.second).toDouble().pow(2)
            }.toFloat()

            if (distance < minDistance) {
                minDistance = distance
                bestMatch = userId
            }
        }

        return if (minDistance < 0.5f) bestMatch else null
    }
}