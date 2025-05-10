package com.example.irisrecognition.ui

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.YuvImage
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.annotation.RequiresApi
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
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.geometry.Size
import com.example.irisrecognition.detection.models.IrisData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Rect
import android.graphics.Rect as Rect_andr
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.max

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
                setEnabledUseCases(
                    CameraController.IMAGE_ANALYSIS or
                            CameraController.VIDEO_CAPTURE
                )
                // Add this workaround for encoder issues
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

        // Separate suspending function for face processing
        suspend fun processDetectedFace(
            detectedFaces: List<Face>,
            cameraController: LifecycleCameraController,
            imageSize: Size
        ) {
            detectedFaces.firstOrNull()?.let { face ->
                val bitmap = withContext(Dispatchers.IO) {
                    getLatestFrame(cameraController)?.let { fullBitmap ->
                        // Get accurate face position in bitmap coordinates
                        val scaleX = fullBitmap.width / imageSize.width
                        val scaleY = fullBitmap.height / imageSize.height

                        val faceRect = Rect(
                            (face.rect.x * scaleX).toInt(),
                            (face.rect.y * scaleY).toInt(),
                            (face.rect.width * scaleX).toInt(),
                            (face.rect.height * scaleY).toInt()
                        )

                        // Create expanded ROI focusing on upper face region
                        val eyeRegion = Rect(
                            (faceRect.x - faceRect.width * 0.1).coerceAtLeast(0.0).toInt(),
                            (faceRect.y + faceRect.height * 0.1).coerceAtLeast(0.0).toInt(),
                            (faceRect.width * 1.2).coerceAtMost((fullBitmap.width - faceRect.x).toDouble())
                                .toInt(),
                            (faceRect.height * 0.4).coerceAtMost((fullBitmap.height - faceRect.y).toDouble())
                                .toInt()
                        )

                        Bitmap.createBitmap(
                            fullBitmap,
                            eyeRegion.x,
                            eyeRegion.y,
                            eyeRegion.width,
                            eyeRegion.height
                        ).also { cropped ->
                            // Apply rotation correction if needed
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
                        }
                    }
                }

                bitmap?.let { faceBitmap ->
                    irisDetector.detectIrisInImage(bitmap) { iris ->
                        // Convert iris positions back to original image coordinates
                        val adjustedIris = iris.copy(
                            leftIris = iris.leftIris?.let { adjustIrisPosition(it, faceBitmap, imageSize) },
                            rightIris = iris.rightIris?.let { adjustIrisPosition(it, faceBitmap, imageSize) }
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
                            // Find best match in database
                            val bestMatch = irisDatabase.findBestMatch(irisData.features)
                            if (bestMatch != null) {
                                recognizedUser = bestMatch
                                irisDetectionResult = "User recognized: $bestMatch"
                            } else {
                                // New user flow
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

    private fun createAnalyzer(
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
    private fun saveToGallery(bitmap: Bitmap, context: Context, title: String) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            saveToGalleryApi29Plus(bitmap, context, title)
        } else {
            saveToGalleryLegacy(bitmap, context, title)
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun saveToGalleryApi29Plus(bitmap: Bitmap, context: Context, title: String) {
        try {
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "iris_$title.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }

            val resolver = context.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                ?: throw IOException("Failed to create MediaStore entry")

            resolver.openOutputStream(uri).use { outputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream ?: throw IOException("Failed to get output stream"))
            }

            contentValues.clear()
            contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, contentValues, null, null)

        } catch (e: Exception) {
            Timber.e(e, "API 29+ Gallery save failed")
            // Fallback to legacy method
            saveToGalleryLegacy(bitmap, context, title)
        }
    }

    private fun saveToGalleryLegacy(bitmap: Bitmap, context: Context, title: String) {
        try {
            val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val appDir = File(imagesDir, "IrisRecognition")
            if (!appDir.exists()) {
                appDir.mkdirs()
            }

            val file = File(appDir, "iris_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { outputStream ->
                if (bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)) {
                    // Notify gallery
                    MediaScannerConnection.scanFile(
                        context,
                        arrayOf(file.absolutePath),
                        arrayOf("image/jpeg"),
                        null
                    )
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Legacy gallery save failed")
            // Final fallback to app-specific storage
            saveToInternalStorage(bitmap, context, title)
        }
    }

    private fun saveToInternalStorage(bitmap: Bitmap, context: Context, title: String) {
        try {
            val file = File(context.filesDir, "iris_$title.jpg")
            FileOutputStream(file).use { outputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            }
            Timber.d("Image saved to internal storage: ${file.absolutePath}")
        } catch (e: Exception) {
            Timber.e(e, "Internal storage save failed")
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
        return try {
            var capturedBitmap: Bitmap? = null
            val latch = CountDownLatch(1)

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
