package com.example.irisrecognition.ui

import IrisDatabase
import android.Manifest
import android.content.Context
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
import kotlin.math.pow
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
import com.example.irisrecognition.detection.models.IrisData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Rect
import android.graphics.Rect as Rect_andr
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min

private data class FaceDetectionResult(
    val faceBitmap: Bitmap,
    val scaleX: Float,
    val scaleY: Float,
    val eyeRegion: Rect
)

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
                val cameraInfo = cameraController.cameraInfo
                previewSize = Size(
                    previewSize.width.toFloat(),
                    previewSize.height.toFloat()
                )
            }
            cameraController.bindToLifecycle(lifecycleOwner)
        }

        suspend fun processFrameForIrisDetection(frame: Bitmap) {
            withContext(Dispatchers.IO) {
                try {
                    val mat = Mat()
                    Utils.bitmapToMat(frame, mat)

                    // Get the first face (we'll use its position to focus iris detection)
                    faces.firstOrNull()?.let { face ->
                        // Convert face rect to bitmap coordinates
                        val scaleX = frame.width / imageSize.width
                        val scaleY = frame.height / imageSize.height

                        val faceRect = Rect(
                            (face.rect.x * scaleX).toInt().coerceAtLeast(0),
                            (face.rect.y * scaleY).toInt().coerceAtLeast(0),
                            (face.rect.width * scaleX).toInt().coerceAtMost(frame.width),
                            (face.rect.height * scaleY).toInt().coerceAtMost(frame.height)
                        )

                        // Focus on upper face region (eyes area) with bounds checking
                        val eyeRegionX = max(0, faceRect.x - faceRect.width / 4)
                        val eyeRegionY = max(0, faceRect.y + faceRect.height / 4)
                        val eyeRegionWidth = min(frame.width - eyeRegionX, faceRect.width * 3 / 2)
                        val eyeRegionHeight = min(frame.height - eyeRegionY, faceRect.height / 2)

                        // Verify the region is valid before cropping
                        if (eyeRegionWidth > 0 && eyeRegionHeight > 0 &&
                            eyeRegionX + eyeRegionWidth <= frame.width &&
                            eyeRegionY + eyeRegionHeight <= frame.height) {

                            // Crop and process the eye region
                            val eyeBitmap = Bitmap.createBitmap(
                                frame,
                                eyeRegionX,
                                eyeRegionY,
                                eyeRegionWidth,
                                eyeRegionHeight
                            )

                            irisDetector.detectIris(eyeBitmap) { iris ->
                                // Adjust coordinates back to full image
                                val adjustedIris = iris.copy(
                                    leftIris = iris.leftIris?.let { irisData ->
                                        IrisData(
                                            Point(
                                                irisData.center.x + eyeRegionX,
                                                irisData.center.y + eyeRegionY
                                            ),
                                            irisData.radius,
                                            irisData.features
                                        )
                                    },
                                    rightIris = iris.rightIris?.let { irisData ->
                                        IrisData(
                                            Point(
                                                irisData.center.x + eyeRegionX,
                                                irisData.center.y + eyeRegionY
                                            ),
                                            irisData.radius,
                                            irisData.features
                                        )
                                    }
                                )

                                irisPairs = listOf(adjustedIris)

                                // Handle recognition
                                adjustedIris.leftIris?.let { irisData ->
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
                            }
                        } else {
                            irisDetectionResult = "Error: Invalid eye region coordinates"
                        }
                    } ?: run {
                        irisDetectionResult = "Error: Face position lost"
                    }
                } catch (e: Exception) {
                    Timber.e(e, "Error processing frame for iris detection")
                    irisDetectionResult = "Error: ${e.localizedMessage}"
                }
            }
        }

        fun createAnalyzer(
            onResult: (List<Face>, Size) -> Unit
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
                                postScale(-1f, 1f) // Mirror for front camera
                            }
                        }

                        val bitmap = image.toBitmap(rotationMatrix)
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

        // Separate suspending function for face processing
        suspend fun processDetectedFace(
            detectedFaces: List<Face>,
            cameraController: LifecycleCameraController,
            imageSize: Size
        ) {
            detectedFaces.firstOrNull()?.let { face ->
                val detectionResult = withContext(Dispatchers.IO) {
                    getLatestFrame(cameraController)?.let { fullBitmap ->
                        // Calculate scaling factors
                        val scaleX = fullBitmap.width / imageSize.width
                        val scaleY = fullBitmap.height / imageSize.height

                        // Calculate face rectangle in bitmap coordinates
                        val faceRectInBitmap = Rect(
                            (face.rect.x * scaleX).toInt(),
                            (face.rect.y * scaleY).toInt(),
                            (face.rect.width * scaleX).toInt(),
                            (face.rect.height * scaleY).toInt()
                        )

                        // Create expanded ROI focusing on upper face region (eyes area)
                        val eyeRegion = Rect(
                            (faceRectInBitmap.x - faceRectInBitmap.width * 0.1).coerceAtLeast(0.0).toInt(),
                            (faceRectInBitmap.y + faceRectInBitmap.height * 0.1).coerceAtLeast(0.0).toInt(),
                            (faceRectInBitmap.width * 1.2).coerceAtMost((fullBitmap.width - faceRectInBitmap.x).toDouble())
                                .toInt(),
                            (faceRectInBitmap.height * 0.5).coerceAtMost((fullBitmap.height - faceRectInBitmap.y).toDouble())
                                .toInt()
                        )

                        // Create face bitmap and return it with the scaling info
                        FaceDetectionResult(
                            faceBitmap = Bitmap.createBitmap(
                                fullBitmap,
                                eyeRegion.x,
                                eyeRegion.y,
                                eyeRegion.width,
                                eyeRegion.height
                            ).also { cropped ->
                                if (isFrontCamera) {
                                    val matrix = Matrix().apply {
                                        postScale(-1f, 1f) // Mirror for front camera
                                    }
                                    Bitmap.createBitmap(
                                        cropped, 0, 0,
                                        cropped.width, cropped.height,
                                        matrix, true
                                    )?.let { rotated ->
                                        cropped.recycle()
                                        rotated
                                    } ?: cropped
                                } else {
                                    cropped
                                }
                            },
                            scaleX = scaleX,
                            scaleY = scaleY,
                            eyeRegion = eyeRegion
                        )
                    }
                }

                detectionResult?.let { result ->
                    saveToInternalStorage(bitmap = result.faceBitmap, context, "face_roi")
                    // Calculate the original face rectangle position relative to the cropped eye region
                    val faceRectInCropped = Rect(
                        (face.rect.x * result.scaleX - result.eyeRegion.x).toInt(),
                        (face.rect.y * result.scaleY - result.eyeRegion.y).toInt(),
                        (face.rect.width * result.scaleX).toInt(),
                        (face.rect.height * result.scaleY).toInt()
                    )

                    irisDetector.detectIris(result.faceBitmap) { iris ->
                        // Convert iris positions back to original image coordinates
                        val adjustedIris = iris.copy(
                            leftIris = iris.leftIris?.let { irisData ->
                                IrisData(
                                    Point(
                                        irisData.center.x + result.eyeRegion.x,
                                        irisData.center.y + result.eyeRegion.y
                                    ),
                                    irisData.radius,
                                    irisData.features
                                )
                            },
                            rightIris = iris.rightIris?.let { irisData ->
                                IrisData(
                                    Point(
                                        irisData.center.x + result.eyeRegion.x,
                                        irisData.center.y + result.eyeRegion.y
                                    ),
                                    irisData.radius,
                                    irisData.features
                                )
                            }
                        )
                        irisPairs = listOf(adjustedIris)
                    }
                }
            } ?: run { irisPairs = emptyList() }
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
            cameraController.setImageAnalysisAnalyzer(
                executor,
                createAnalyzer { detectedFaces, currentImageSize ->
                    faces = detectedFaces
                    imageSize = currentImageSize
                    launch {
                        processDetectedFace(
                            detectedFaces, cameraController,
                            imageSize = imageSize
                        )
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
                                processFrameForIrisDetection(frame)
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

    private fun adjustIrisPosition(irisData: IrisData, faceBitmap: Bitmap, imageSize: Size): IrisData {
        // Calculate scale factors
        val scaleX = imageSize.width / faceBitmap.width
        val scaleY = imageSize.height / faceBitmap.height

        return IrisData(
            center = Point(
                irisData.center.x * scaleX,
                irisData.center.y * scaleY
            ),
            radius = irisData.radius * max(scaleX, scaleY),
            features = irisData.features
        )
    }

    private fun saveToInternalStorage(bitmap: Bitmap, context: Context, title: String) {
        try {
            // Ensure directory exists
            val directory = File(context.filesDir, "iris_images")
            if (!directory.exists()) {
                directory.mkdirs()
            }

            val file = File(directory, "${title}_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { outputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                outputStream.flush()
            }
            Timber.d("Image saved to: ${file.absolutePath}")
        } catch (e: Exception) {
            Timber.e(e, "Failed to save image")
        }
    }

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