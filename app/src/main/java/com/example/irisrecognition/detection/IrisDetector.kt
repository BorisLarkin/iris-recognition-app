package com.example.irisrecognition.detection

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import timber.log.Timber
import com.example.irisrecognition.detection.models.Iris
import com.example.irisrecognition.detection.models.IrisData
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import kotlin.math.*

class IrisDetector(context: Context) {
    private var eyeCascade: CascadeClassifier? = null

    init {
        try {
            eyeCascade = loadCascadeClassifier(context, "haarcascade_eye.xml")
            Timber.d("IrisDetector initialized successfully")
        } catch (e: Exception) {
            Timber.e(e, "Error initializing IrisDetector")
        }
    }

    private fun loadCascadeClassifier(context: Context, filename: String): CascadeClassifier {
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

    fun detectIris(image: Bitmap, callback: (Iris) -> Unit) {
        try {
            val mat = Mat()
            Utils.bitmapToMat(image, mat)

            val grayImage = Mat()
            Imgproc.cvtColor(mat, grayImage, Imgproc.COLOR_BGR2GRAY)
            Imgproc.equalizeHist(grayImage, grayImage)

            // Detect eyes in the image
            val eyes = MatOfRect()
            eyeCascade?.detectMultiScale(
                grayImage, eyes, 1.1, 2, 0,
                Size(30.0, 30.0), Size()
            )

            // Process detected eyes
            val eyeList = eyes.toList()
            val irisResult = if (eyeList.size >= 2) {
                // Assuming first two detections are left and right eyes
                val leftEye = eyeList[0]
                val rightEye = eyeList[1]

                Iris(
                    leftIris = createIrisData(leftEye, grayImage),
                    rightIris = createIrisData(rightEye, grayImage)
                )
            } else if (eyeList.size == 1) {
                // Only one eye detected
                Iris(
                    leftIris = createIrisData(eyeList[0], grayImage),
                    rightIris = null
                )
            } else {
                Iris(null, null)
            }

            callback(irisResult)
        } catch (e: Exception) {
            Timber.e(e, "Error detecting iris")
            callback(Iris(null, null))
        }
    }

    private fun createIrisData(eyeRect: Rect, grayImage: Mat): IrisData {
        val eyeROI = grayImage.submat(eyeRect)

        // Convert to color for iris color analysis
        val colorEyeROI = Mat()
        Imgproc.cvtColor(eyeROI, colorEyeROI, Imgproc.COLOR_GRAY2BGR)
        val hsvEyeROI = Mat()
        Imgproc.cvtColor(colorEyeROI, hsvEyeROI, Imgproc.COLOR_BGR2HSV)

        // Detect iris circle
        val circles = Mat()
        Imgproc.HoughCircles(
            eyeROI, circles, Imgproc.HOUGH_GRADIENT,
            2.0, eyeROI.rows().toDouble() / 8.0,
            100.0, 30.0,
            (eyeRect.width * 0.2).toInt(), (eyeRect.width * 0.5).toInt()
        )

        return if (circles.cols() > 0) {
            val circle = circles.get(0, 0)
            val center = Point(
                eyeRect.x + circle[0],
                eyeRect.y + circle[1]
            )
            val radius = circle[2].toFloat()

            // Extract color histogram features
            val features = extractIrisColorFeatures(hsvEyeROI, center, radius)

            IrisData(center, radius, features)
        } else {
            // Fallback with estimated iris position
            val center = Point(
                eyeRect.x + eyeRect.width * 0.5,
                eyeRect.y + eyeRect.height * 0.5
            )
            val radius = (eyeRect.width * 0.3).toFloat()

            IrisData(
                center,
                radius,
                extractIrisColorFeatures(hsvEyeROI, center, radius)
            )
        }
    }

    private fun extractIrisColorFeatures(hsvImage: Mat, center: Point, radius: Float): FloatArray {
        // Create mask for iris region
        val mask = Mat.zeros(hsvImage.size(), CvType.CV_8UC1)
        Imgproc.circle(mask, Point(center.x, center.y), radius.toInt(), Scalar(255.0), -1)

        // Calculate color histogram (Hue channel)
        val hist = Mat()
        val channels = MatOfInt(0) // Hue channel
        val histSize = MatOfInt(8) // 8 bins for hue
        val ranges = MatOfFloat(0f, 180f) // Hue range

        Imgproc.calcHist(
            listOf(hsvImage),
            channels,
            mask,
            hist,
            histSize,
            ranges
        )

        // Normalize histogram to get features
        Core.normalize(hist, hist, 1.0, 0.0, Core.NORM_L1)

        // Convert to float array
        val features = FloatArray(8)
        for (i in 0 until 8) {
            features[i] = hist.get(i, 0)[0].toFloat()
        }

        return features
    }

    private fun extractIrisFeatures(eyeROI: Mat, center: Point, radius: Float): FloatArray {
        // Simplified feature extraction
        val features = FloatArray(128)

        // Sample pixels around the iris
        for (i in 0 until 64) {
            val angle = 2 * Math.PI * i / 64
            val x = center.x + radius * cos(angle)
            val y = center.y + radius * sin(angle)

            if (x >= 0 && x < eyeROI.cols() && y >= 0 && y < eyeROI.rows()) {
                val pixelValue = eyeROI.get(y.toInt(), x.toInt())[0].toFloat() / 255.0f
                features[i] = pixelValue
                features[i + 64] = 1 - pixelValue // Complementary feature
            }
        }

        return features
    }

    fun close() {
        eyeCascade = null
    }
}