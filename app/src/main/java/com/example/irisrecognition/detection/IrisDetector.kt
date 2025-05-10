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

    fun detectIrisInImage(
        image: Bitmap,
        approximateFaceRegion: Rect? = null,
        callback: (Iris) -> Unit
    ) {
        try {
            val mat = Mat()
            Utils.bitmapToMat(image, mat)

            // If we have an approximate region, focus there
            val roi = approximateFaceRegion?.let { region ->
                // Expand the region by 20% for safety
                Rect(
                    (region.x - region.width * 0.2).toInt().coerceAtLeast(0),
                    (region.y - region.height * 0.1).toInt().coerceAtLeast(0),
                    (region.width * 1.4).toInt().coerceAtMost(mat.cols() - region.x),
                    (region.height * 1.2).toInt().coerceAtMost(mat.rows() - region.y)
                )
            } ?: Rect(0, 0, mat.cols(), mat.rows())

            // Work on grayscale image with enhanced contrast
            val gray = Mat()
            Imgproc.cvtColor(mat.submat(roi), gray, Imgproc.COLOR_BGR2GRAY)
            val clahe = Imgproc.createCLAHE(3.0, Size(8.0, 8.0))
            clahe.apply(gray, gray)

            // Detect eyes first (if no face region provided)
            val eyes = if (approximateFaceRegion == null) {
                val eyesDetected = MatOfRect()
                eyeCascade?.detectMultiScale(
                    gray, eyesDetected, 1.1, 3, 0,
                    Size(gray.cols() * 0.15, gray.rows() * 0.15),
                    Size()
                )
                eyesDetected.toList()
            } else emptyList()

            // If we found eyes, use those regions, otherwise scan whole ROI
            val irisResults = if (eyes.isNotEmpty()) {
                eyes.map { eyeRect ->
                    detectIrisInEyeRegion(gray.submat(eyeRect), eyeRect.x + roi.x, eyeRect.y + roi.y)
                }
            } else {
                // Full image iris detection
                detectIrisInRegion(gray, roi.x, roi.y)
            }

            // Pair results (left is leftmost iris)
            val pairedResult = when {
                irisResults.size >= 2 -> {
                    val sorted = irisResults.sortedBy { it.center.x }
                    Iris(leftIris = sorted[0], rightIris = sorted[1])
                }
                irisResults.size == 1 -> {
                    Iris(leftIris = irisResults[0], rightIris = null)
                }
                else -> Iris(null, null)
            }

            callback(pairedResult)
        } catch (e: Exception) {
            Timber.e(e, "Independent iris detection failed")
            callback(Iris(null, null))
        }
    }

    private fun detectIrisInEyeRegion(eyeRegion: Mat, offsetX: Int, offsetY: Int): IrisData {
        val circles = Mat()
        Imgproc.HoughCircles(
            eyeRegion, circles, Imgproc.HOUGH_GRADIENT,
            1.5, // dp
            eyeRegion.rows() / 4.0, // minDist (smaller for eye regions)
            80.0, // param1
            25.0, // param2
            (eyeRegion.rows() * 0.2).toInt(), // minRadius
            (eyeRegion.rows() * 0.4).toInt()  // maxRadius
        )

        return if (circles.cols() > 0) {
            val circle = circles.get(0, 0)
            val center = Point(
                circle[0] + offsetX,
                circle[1] + offsetY
            )
            val radius = circle[2].toFloat()
            IrisData(center, radius, extractIrisFeatures(eyeRegion, center, radius))
        } else {
            // Fallback to center of eye region
            val center = Point(
                eyeRegion.cols() / 2.0 + offsetX,
                eyeRegion.rows() / 2.0 + offsetY
            )
            val radius = (eyeRegion.rows() * 0.25).toFloat()
            IrisData(center, radius, FloatArray(128)) // Empty features
        }
    }

    private fun detectIrisInRegion(region: Mat, offsetX: Int, offsetY: Int): List<IrisData> {
        val circles = Mat()
        Imgproc.HoughCircles(
            region, circles, Imgproc.HOUGH_GRADIENT,
            1.5, // dp
            region.rows() / 8.0, // minDist
            80.0, // param1
            25.0, // param2
            15,   // minRadius
            45    // maxRadius
        )

        return (0 until circles.cols()).mapNotNull { i ->
            val circle = circles.get(0, i)
            val center = Point(
                circle[0] + offsetX,
                circle[1] + offsetY
            )
            val radius = circle[2].toFloat()

            // Filter by position (should be in upper half of image)
            if (center.y < region.rows() * 0.6) {
                IrisData(center, radius, extractIrisFeatures(region, center, radius))
            } else null
        }
    }

    fun detectIris(image: Bitmap, callback: (Iris) -> Unit) {
        try {
            val mat = Mat()
            Utils.bitmapToMat(image, mat)

            // Convert to grayscale with CLAHE for better contrast
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(gray, gray)

            // Detect circles with adjusted parameters
            val circles = Mat()
            Imgproc.HoughCircles(
                gray, circles, Imgproc.HOUGH_GRADIENT,
                1.5, // dp
                gray.rows() / 8.0, // minDist
                80.0, // param1 (reduced from 100)
                25.0, // param2 (reduced from 30)
                15,   // minRadius (increased)
                45    // maxRadius (reduced)
            )

            // Process detected circles
            val irisList = mutableListOf<IrisData>()
            for (i in 0 until circles.cols()) {
                val circle = circles.get(0, i)
                val center = Point(circle[0], circle[1])
                val radius = circle[2].toFloat()

                // Filter by position (should be in upper half of image)
                if (center.y < image.height / 2) {
                    irisList.add(IrisData(center, radius, extractIrisFeatures(mat, center, radius)))
                }
            }

            // Pair irises (left is the leftmost one)
            val result = when {
                irisList.size >= 2 -> {
                    val sorted = irisList.sortedBy { it.center.x }
                    Iris(leftIris = sorted[0], rightIris = sorted[1])
                }
                irisList.size == 1 -> {
                    Iris(leftIris = irisList[0], rightIris = null)
                }
                else -> Iris(null, null)
            }

            callback(result)
        } catch (e: Exception) {
            Timber.e(e, "Iris detection error")
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
            1.5, // dp
            eyeROI.rows() / 8.0, // minDist
            100.0, // param1
            30.0, // param2
            (eyeRect.width * 0.2).toInt(), // minRadius
            (eyeRect.width * 0.4).toInt() // maxRadius
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