package com.example.irisrecognition.detection

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector as MpFaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import timber.log.Timber
import com.example.irisrecognition.detection.models.Face
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint
import com.google.mediapipe.tasks.core.BaseOptions
import org.opencv.android.Utils

class FaceDetector(context: Context) {
    private val faceDetector: MpFaceDetector
    private var currentCallback: ((List<Face>) -> Unit)? = null

    init {
        // Проверка существования файла модели
        try {
            context.assets.open("blaze_face_short_range.tflite").close()
        } catch (e: Exception) {
            Timber.e(e, "Model file not found")
            throw RuntimeException("Model file not found in assets", e)
        }

        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("blaze_face_short_range.tflite")
            .build()

        val options = MpFaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinDetectionConfidence(0.7f)
            .setResultListener(this::handleFaceDetectionResult)
            .build()

        faceDetector = try {
            MpFaceDetector.createFromOptions(context, options)
        } catch (e: Exception) {
            Timber.e(e, "Error creating face detector")
            throw RuntimeException("Failed to initialize face detector", e)
        }
        Timber.d("FaceDetector initialized successfully")
    }

    fun detectFaces(image: Mat, callback: (List<Face>) -> Unit) {
        try {
            if (image.empty()) {
                callback(emptyList())
                return
            }

            val bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(image, bitmap)

            val mpImage = BitmapImageBuilder(bitmap).build()
            faceDetector.detectAsync(mpImage, System.currentTimeMillis())
            currentCallback = callback
        } catch (e: Exception) {
            Timber.e(e, "Error detecting faces")
            callback(emptyList())
        }
    }

    private fun handleFaceDetectionResult(result: FaceDetectorResult, input: MPImage) {
        val faces = result.detections().mapNotNull { detection ->
            detection.boundingBox()?.let { boundingBox ->
                // Convert normalized coordinates to image coordinates
                val left = boundingBox.left * input.width
                val top = boundingBox.top * input.height
                val right = boundingBox.right * input.width
                val bottom = boundingBox.bottom * input.height

                val rect = Rect(
                    left.toInt(),
                    top.toInt(),
                    (right - left).toInt(),
                    (bottom - top).toInt()
                )

                val landmarks = mutableListOf<Point>()
                detection.keypoints().ifPresent { keypoints ->
                    keypoints.forEach { keypoint ->
                        landmarks.add(
                            Point(
                                (keypoint.x() * input.width).toDouble(),
                                (keypoint.y() * input.height).toDouble()
                            )
                        )
                    }
                }

                Face(rect, landmarks)
            }
        }

        currentCallback?.invoke(faces)
        currentCallback = null
    }

    fun close() {
        faceDetector.close()
    }
}