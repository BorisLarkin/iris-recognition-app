package com.example.irisrecognition.detection

import android.content.Context
import android.graphics.Bitmap
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
    private var eyeCascade: CascadeClassifier? = null

    init {
        try {
            // Load face cascade
            faceCascade = loadCascadeClassifier(context, "haarcascade_frontalface_alt.xml")

            // Load eye cascade
            eyeCascade = loadCascadeClassifier(context, "haarcascade_eye.xml")

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
        if (faceCascade == null || eyeCascade == null) {
            callback(emptyList())
            return
        }

        try {
            val grayImage = Mat()
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY)
            Imgproc.equalizeHist(grayImage, grayImage)

            val faces = MatOfRect()
            faceCascade?.detectMultiScale(
                grayImage, faces, FACE_DETECTION_SCALE.toDouble(), 3, 0,
                Size(MIN_FACE_SIZE.toDouble(), MIN_FACE_SIZE.toDouble()),
                Size(1000.0, 1000.0)
            )

            val faceResults = faces.toList().map { faceRect ->
                // Expand face rectangle by 20% for better eye detection
                val expandedFaceRect = Rect(
                    (faceRect.x - faceRect.width * 0.2).toInt().coerceAtLeast(0),
                    (faceRect.y - faceRect.height * 0.2).toInt().coerceAtLeast(0),
                    (faceRect.width * 1.4).toInt().coerceAtMost(grayImage.cols()),
                    (faceRect.height * 1.4).toInt().coerceAtMost(grayImage.rows())
                )

                val faceROI = grayImage.submat(expandedFaceRect)
                val eyes = MatOfRect()
                eyeCascade?.detectMultiScale(
                    faceROI, eyes, 1.1, 2, 0,
                    Size(30.0, 30.0), Size()
                )

                val eyeLandmarks = eyes.toList().map { eyeRect ->
                    Point(
                        expandedFaceRect.x + eyeRect.x + eyeRect.width * 0.5,
                        expandedFaceRect.y + eyeRect.y + eyeRect.height * 0.5
                    )
                }

                Face(expandedFaceRect, eyeLandmarks)
            }

            callback(faceResults)
        } catch (e: Exception) {
            Timber.e(e, "Error detecting faces")
            callback(emptyList())
        }
    }

    fun close() {
        faceCascade = null
        eyeCascade = null
    }
}