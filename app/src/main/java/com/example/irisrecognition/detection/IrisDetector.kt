package com.example.irisrecognition.detection

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import timber.log.Timber
import com.example.irisrecognition.detection.models.Iris
import com.example.irisrecognition.detection.models.IrisData
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.opencv.core.Point
import kotlin.math.sqrt
import kotlin.math.pow

class IrisDetector(context: Context) {
    private val faceLandmarker: FaceLandmarker
    private var resultListener: ((FaceLandmarkerResult, MPImage) -> Unit)? = null

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinFaceDetectionConfidence(0.7f)
            .setMinFacePresenceConfidence(0.7f)
            .setMinTrackingConfidence(0.7f)
            .setNumFaces(1)
            .setResultListener { result, input ->
                resultListener?.invoke(result, input)
            }
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(context, options)
    }

    fun detectIris(image: Bitmap, callback: (Iris) -> Unit) {
        try {
            resultListener = { result, _ ->
                processLandmarkResult(result, callback)
                resultListener = null
            }

            val mpImage = BitmapImageBuilder(image).build()
            faceLandmarker.detectAsync(mpImage, System.currentTimeMillis())
            Timber.d(message = "Detecting worked without an error")
        } catch (e: Exception) {
            Timber.e(e, "Error detecting iris")
            callback(Iris(null, null))
        }
    }

    private fun processLandmarkResult(result: FaceLandmarkerResult, callback: (Iris) -> Unit) {
        if (result.faceLandmarks().isEmpty()) {
            callback(Iris(null, null))
            return
        }

        val landmarks = result.faceLandmarks()[0]

        // MediaPipe iris landmarks:
        // Left iris: index 468 (center) + 4 surrounding points (469-472)
        // Right iris: index 473 (center) + 4 surrounding points (474-477)
        val leftIris = if (landmarks.size >= 473) {
            val irisLandmarks = listOf(landmarks[468]) + landmarks.subList(469, 473)
            createIrisData(irisLandmarks)
        } else null

        val rightIris = if (landmarks.size >= 478) {
            val irisLandmarks = listOf(landmarks[473]) + landmarks.subList(474, 478)
            createIrisData(irisLandmarks)
        } else null
        Timber.d(message = "Checked landmark results")
        callback(Iris(leftIris, rightIris))
    }

    private fun createIrisData(irisLandmarks: List<NormalizedLandmark>): IrisData {
        // Calculate center as average of all iris landmarks
        val centerX = irisLandmarks.map { it.x() }.average().toFloat()
        val centerY = irisLandmarks.map { it.y() }.average().toFloat()

        // Calculate radius as maximum distance from center to any landmark point
        val radius = irisLandmarks.maxOf { landmark ->
            sqrt(
                (landmark.x() - centerX).toDouble().pow(2) +
                        (landmark.y() - centerY).toDouble().pow(2)
            ).toFloat()
        }.coerceAtLeast(0.01f) // Ensure minimum radius

        // Extract features (normalized relative to center)
        val features = FloatArray(10).apply {
            irisLandmarks.forEachIndexed { i, landmark ->
                if (i * 2 < size) {
                    this[i * 2] = landmark.x() - centerX
                    this[i * 2 + 1] = landmark.y() - centerY
                }
            }
        }

        return IrisData(
            center = Point(centerX.toDouble(), centerY.toDouble()),
            radius = radius,
            features = features
        )
    }

    private fun extractIrisFeatures(landmarks: List<NormalizedLandmark>): FloatArray {
        val features = FloatArray(128)
        val centerX = landmarks.map { it.x() }.average().toFloat()
        val centerY = landmarks.map { it.y() }.average().toFloat()

        landmarks.forEachIndexed { i, landmark ->
            if (i * 2 < features.size) {
                features[i * 2] = landmark.x() - centerX
                features[i * 2 + 1] = landmark.y() - centerY
            }
        }

        return features
    }

    fun close() {
        faceLandmarker.close()
    }
}