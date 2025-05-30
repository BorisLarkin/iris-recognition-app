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
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.room.util.copy
import com.example.irisrecognition.detection.models.IrisData
import com.example.irisrecognition.ui.components.RecognitionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Rect
import android.graphics.Rect as Rect_andr
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlin.math.max
import kotlin.math.min
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.runtime.LaunchedEffect

class MainActivity : ComponentActivity() {
    private lateinit var faceDetector: FaceDetector
    private lateinit var irisDetector: IrisDetector
    private val executor = Executors.newSingleThreadExecutor()
    private var isOpenCvInitialized = false
    private var isFrontCamera = false
    private var cameraSwitchInProgress by mutableStateOf(false)
    private var showIrisResultDialog by mutableStateOf(false)
    private var irisDetectionResult by mutableStateOf<String?>(null)
    private var isProcessingFrame by mutableStateOf(false)
    private var lastProcessedBitmap by mutableStateOf<Bitmap?>(null)
    private lateinit var irisDatabase: IrisDatabase


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
        irisDatabase = IrisDatabase(this)
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
        var confidence by remember { mutableStateOf<Float?>(null) }
        var currentRotation by remember{ mutableIntStateOf(0) }
        var isResultVisible by remember { mutableStateOf(false) }



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

        // Automatically hide after delay when showIrisResultDialog becomes true
        LaunchedEffect(showIrisResultDialog) {
            if (showIrisResultDialog) {
                isResultVisible = true
                delay(5000) // Show for 3 seconds before fading out
                isResultVisible = false
                showIrisResultDialog = false // Also reset the dialog flag
            }
        }

        suspend fun FaceDetector.detectFacesSuspended(mat: Mat): List<Face> = suspendCoroutine { continuation ->
            detectFaces(mat) { faces ->
                continuation.resume(faces)
            }
        }

        suspend fun processBitmapForIrisDetection(bitmap: Bitmap, currentImageSize: Size, recognize : Boolean) {
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)

            val detectedFaces = faceDetector.detectFacesSuspended(mat)
            faces = detectedFaces

            if (detectedFaces.isNotEmpty()) {
                irisDetector.detectIris(bitmap) { iris ->
                    irisPairs = listOf(iris)
                    if (recognize){
                        iris.leftIris?.let { irisData ->
                            val matchResult = irisDatabase.findBestMatch(irisData.features)
                            recognizedUser = matchResult.first
                            confidence = matchResult.second
                        } ?: run {
                            recognizedUser = null
                            confidence = null
                        }
                    }
                }
            } else {
                irisPairs = emptyList()
                recognizedUser = null
            }

            imageSize = currentImageSize
        }
        suspend fun processFrameForIrisDetection(image: ImageProxy, recognize : Boolean = false) {
            withContext(Dispatchers.IO) {
                try {
                    if (frameCounter++ % 3 != 0) {
                        image.close()
                        return@withContext
                    }

                    val bitmap = image.toBitmap(currentRotation, isFrontCamera)
                    processBitmapForIrisDetection(bitmap, Size(image.width.toFloat(), image.height.toFloat()), recognize)
                } catch (e: Exception) {
                    Timber.e(e, "Error processing frame")
                } finally {
                    image.close()
                }
            }
        }

        suspend fun processFrameForIrisDetection(bitmap: Bitmap, recognize : Boolean = false) {
            withContext(Dispatchers.IO) {
                try {
                    processBitmapForIrisDetection(bitmap, imageSize, recognize)
                } catch (e: Exception) {
                    Timber.e(e, "Error processing Bitmap frame")
                    irisDetectionResult = "Error: ${e.localizedMessage}"
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

        fun createAnalyzer(
            onFrame: suspend (ImageProxy) -> Unit
        ): ImageAnalysis.Analyzer {
            return object : ImageAnalysis.Analyzer {
                @OptIn(ExperimentalGetImage::class)
                override fun analyze(image: ImageProxy) {
                    coroutineScope.launch {
                        try {
                            onFrame(image)
                        } catch (e: Exception) {
                            Timber.e(e, "Error in analyzer")
                        }
                    }
                }
            }
        }

        fun createSingleFrameAnalyzer(
            latch: CountDownLatch,
            onFrameCaptured: (Bitmap) -> Unit
        ): ImageAnalysis.Analyzer {
            return object : ImageAnalysis.Analyzer {
                @OptIn(ExperimentalGetImage::class)
                override fun analyze(image: ImageProxy) {
                    try {
                        val bitmap = image.toBitmap(currentRotation, isFrontCamera)
                        onFrameCaptured(bitmap)
                    } catch (e: Exception) {
                        Timber.e(e, "Error in single frame analyzer")
                    } finally {
                        image.close()
                        latch.countDown()
                    }
                }
            }
        }

        // Update the getLatestFrame function
        suspend fun getLatestFrame(cameraController: LifecycleCameraController): Bitmap? {
            return withContext(Dispatchers.IO) {
                try {
                    val latch = CountDownLatch(1)
                    var capturedBitmap: Bitmap? = null

                    // Create and set single frame analyzer
                    val analyzer = createSingleFrameAnalyzer(latch) { bitmap ->
                        capturedBitmap = bitmap
                    }

                    withContext(Dispatchers.Main) {
                        cameraController.setImageAnalysisAnalyzer(executor, analyzer)
                    }

                    // Wait for frame capture
                    if (!latch.await(1, TimeUnit.SECONDS)) {
                        Timber.w("Timeout waiting for frame capture")
                        return@withContext null
                    }

                    // Restore the original analyzer
                    val mainAnalyzer = createAnalyzer { image ->
                        processFrameForIrisDetection(image)
                    }

                    withContext(Dispatchers.Main) {
                        cameraController.setImageAnalysisAnalyzer(executor, mainAnalyzer)
                    }

                    capturedBitmap
                } catch (e: Exception) {
                    Timber.e(e, "Error capturing frame")
                    null
                }
            }
        }

        // Keep your existing analyzer setup in a separate LaunchedEffect
        LaunchedEffect(Unit) {
            val mainAnalyzer = createAnalyzer { image ->
                processFrameForIrisDetection(image)
            }

            cameraController.setImageAnalysisAnalyzer(executor, mainAnalyzer)

            // Keep the effect alive
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
                                val frame = getLatestFrame(cameraController)
                                if (frame != null) {
                                    lastProcessedBitmap = frame
                                    processFrameForIrisDetection(frame, recognize = true)
                                    irisDetectionResult = recognizedUser?.let {
                                        "User recognized: $it"
                                    } ?: "Not recognized"
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
                isScanning = showIrisResultDialog && irisDetectionResult == "Scanning iris...",
                currentRotation = currentRotation,
                isFrontCamera = isFrontCamera
        )

        if (showIrisResultDialog) {
            // Wrap your RecognitionResult with AnimatedVisibility
            AnimatedVisibility(
                visible = isResultVisible,
                enter = fadeIn(animationSpec = tween(durationMillis = 300)),
                exit = fadeOut(animationSpec = tween(durationMillis = 500))
            ) {
                RecognitionResult(
                    user = recognizedUser,
                    confidence = confidence,
                    modifier = Modifier
                        .padding(16.dp)
                )
            }

        }
    }

    private var frameCounter = 0

    // Add this extension function
    private fun ImageProxy.toBitmap(rotationDegrees: Int, isFrontCamera: Boolean): Bitmap {
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

        val rotationMatrix = Matrix().apply {
            if (isFrontCamera) {
                postRotate(-90f)
                postScale(-1f, 1f)
            }
            else {
                postRotate(90f)
            }
        }

        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, rotationMatrix, true
        )
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