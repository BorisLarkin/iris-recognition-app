package com.example.irisrecognition.detection

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.ui.geometry.Rect
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import timber.log.Timber
import com.example.irisrecognition.detection.models.Face
import java.io.File
import java.io.FileOutputStream

private const val FACE_DETECTION_SCALE = 1.2f
private const val MIN_FACE_SIZE = 100

class FaceDetector(context: Context) {
    private var faceCascade: CascadeClassifier? = null

    init {
        try {
            // Load face cascade
            faceCascade = loadCascadeClassifier(context, "haarcascade_frontalface_alt.xml")

            Timber.d("FaceDetector initialized successfully")
        } catch (e: Exception) {
            Timber.e(e, "Error initializing FaceDetector")
        }
    }

    private fun loadCascadeClassifier(context: Context, filename: String): CascadeClassifier {
        // Load cascade file from assets
        val inputStream = context.assets.open(filename)
        val cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE)
        val cascadeFile = File(cascadeDir, filename)

        FileOutputStream(cascadeFile).use { os ->
            val buffer = ByteArray(4096)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                os.write(buffer, 0, bytesRead)
            }
        }

        return CascadeClassifier(cascadeFile.absolutePath).also {
            if (it.empty()) {
                throw RuntimeException("Failed to load cascade classifier: $filename")
            }
            cascadeFile.delete()
            cascadeDir.delete()
        }
    }

    fun detectFaces(image: Mat, callback: (List<Face>) -> Unit) {
        try {
            val grayImage = Mat()
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY)
            Imgproc.equalizeHist(grayImage, grayImage)

            // Increased minimum face size and adjusted parameters
            val faces = MatOfRect()
            faceCascade?.detectMultiScale(
                grayImage, faces, 1.05, 4, 0,
                Size(150.0, 150.0), // Increased minimum face size
                Size(800.0, 800.0)  // Reduced maximum face size
            )

            val faceResults = faces.toList().map { faceRect ->
                // Return absolute coordinates (no normalization)
                Face(
                    org.opencv.core.Rect(
                        faceRect.x,
                        faceRect.y,
                        faceRect.width,
                        faceRect.height
                    ),
                    emptyList()
                )
            }

            callback(faceResults)
        } catch (e: Exception) {
            Timber.e(e, "Face detection error")
            callback(emptyList())
        }
    }

    fun close() {
        faceCascade = null
    }
}