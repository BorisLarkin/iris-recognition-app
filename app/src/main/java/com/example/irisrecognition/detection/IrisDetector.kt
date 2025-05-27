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

            // Convert to grayscale for eye detection
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)

            // Convert to HSV for color features
            val hsv = Mat()
            Imgproc.cvtColor(mat, hsv, Imgproc.COLOR_BGR2HSV)

            // Detect eyes first using the cascade classifier
            val eyes = MatOfRect()
            eyeCascade?.detectMultiScale(
                gray, eyes, 1.1, 3, 0,
                Size(30.0, 30.0), Size(200.0, 200.0)
            )

            val eyeRects = eyes.toList()
            val irises = mutableListOf<IrisData>()

            for (eyeRect in eyeRects) {
                // Extract eye region
                val eyeROI = gray.submat(eyeRect)
                val eyeHSV = hsv.submat(eyeRect)

                // Apply preprocessing
                val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
                clahe.apply(eyeROI, eyeROI)
                Imgproc.GaussianBlur(eyeROI, eyeROI, Size(5.0, 5.0), 0.0)

                // Detect circles (irises)
                val circles = Mat()
                Imgproc.HoughCircles(
                    eyeROI, circles, Imgproc.HOUGH_GRADIENT,
                    1.2, eyeROI.rows() / 8.0, 100.0, 30.0, 10, 50
                )

                // Process detected circles
                for (i in 0 until min(circles.cols(), 1)) { // Take only the most prominent circle per eye
                    val circle = circles.get(0, i)
                    // In IrisDetector.kt
                    val center = Point(
                        circle[0] + eyeRect.x, circle[1] + eyeRect.y
                    )
                    val radius = circle[2].toFloat() // Normalize radius too

                    // Extract both shape and color features
                    val shapeFeatures = extractIrisFeatures(eyeROI, Point(circle[0], circle[1]), radius)
                    val colorFeatures = extractIrisColorFeatures(eyeHSV, Point(circle[0], circle[1]), radius)

                    // Combine features
                    val combinedFeatures = shapeFeatures + colorFeatures

                    irises.add(IrisData(center, radius, combinedFeatures))
                }
            }

            // Pair irises (left is leftmost)
            val result = when {
                irises.size >= 2 -> {
                    val sorted = irises.sortedBy { it.center.x }
                    Iris(leftIris = sorted[0], rightIris = sorted[1])
                }
                irises.size == 1 -> {
                    Iris(leftIris = irises[0], rightIris = null)
                }
                else -> Iris(null, null)
            }

            callback(result)
        } catch (e: Exception) {
            Timber.e(e, "Iris detection failed")
            callback(Iris(null, null))
        }
    }

    private fun extractIrisFeatures(eyeROI: Mat, center: Point, radius: Float): FloatArray {
        // Enhanced feature extraction with Gabor filters and LBP
        val features = FloatArray(256) // Correct size: 128 + 128 = 256

        // 1. Polar coordinates texture features (similar to before but more robust)
        val steps = 16
        val rings = 8
        for (r in 0 until rings) {
            val currentRadius = radius * (r + 1) / rings
            for (a in 0 until steps) {
                val angle = 2 * Math.PI * a / steps
                val x = center.x + currentRadius * cos(angle)
                val y = center.y + currentRadius * sin(angle)

                if (x >= 0 && x < eyeROI.cols() && y >= 0 && y < eyeROI.rows()) {
                    val pixelValue = eyeROI.get(y.toInt(), x.toInt())[0].toFloat() / 255.0f
                    features[r * steps + a] = pixelValue
                }
            }
        }

        // 2. Local Binary Patterns (LBP) features - more robust to lighting
        val lbpFeatures = extractLBPFeatures(eyeROI, center, radius)
        System.arraycopy(lbpFeatures, 0, features, rings*steps, lbpFeatures.size)

        return features
    }

    private fun extractLBPFeatures(eyeROI: Mat, center: Point, radius: Float): FloatArray {
        val lbp = Mat(eyeROI.size(), eyeROI.type())
        val radius = 1
        val neighbors = 8
        val gridSize = 4 // 4x4 grid

        // Simple LBP implementation
        for (y in radius until eyeROI.rows()-radius) {
            for (x in radius until eyeROI.cols()-radius) {
                val centerVal = eyeROI.get(y, x)[0]
                var code = 0
                for (n in 0 until neighbors) {
                    val theta = 2 * Math.PI * n / neighbors
                    val xn = x + (radius * cos(theta)).toInt()
                    val yn = y - (radius * sin(theta)).toInt()
                    val neighborVal = eyeROI.get(yn, xn)[0]
                    if (neighborVal >= centerVal) {
                        code = code or (1 shl n)
                    }
                }
                lbp.put(y, x, code.toDouble())
            }
        }

        // Divide into grid and calculate histogram for each cell
        val features = FloatArray(gridSize * gridSize * 8) // 8 bins per histogram
        val cellHeight = lbp.rows() / gridSize
        val cellWidth = lbp.cols() / gridSize

        for (i in 0 until gridSize) {
            for (j in 0 until gridSize) {
                val cell = lbp.submat(
                    i * cellHeight, min((i+1)*cellHeight, lbp.rows()),
                    j * cellWidth, min((j+1)*cellWidth, lbp.cols())
                )

                val hist = Mat()
                val channels = MatOfInt(0)
                val histSize = MatOfInt(8)
                val ranges = MatOfFloat(0f, 256f)

                Imgproc.calcHist(listOf(cell), channels, Mat(), hist, histSize, ranges)
                Core.normalize(hist, hist, 1.0, 0.0, Core.NORM_L1)

                for (k in 0 until 8) {
                    features[(i*gridSize + j)*8 + k] = hist.get(k, 0)[0].toFloat()
                }
            }
        }

        return features
    }

    private fun extractIrisColorFeatures(hsvImage: Mat, center: Point, radius: Float): FloatArray {
        // Create mask for iris region
        val mask = Mat.zeros(hsvImage.size(), CvType.CV_8UC1)
        Imgproc.circle(mask, center, radius.toInt(), Scalar(255.0), -1)

        // Calculate color histogram for Hue and Saturation channels
        val hist = Mat()
        val channels = MatOfInt(0, 1) // Hue and Saturation
        val histSize = MatOfInt(8, 8)
        val ranges = MatOfFloat(0f, 180f, 0f, 256f)

        Imgproc.calcHist(
            listOf(hsvImage),
            channels,
            mask,
            hist,
            histSize,
            ranges
        )

        // Normalize histogram
        Core.normalize(hist, hist, 1.0, 0.0, Core.NORM_L1)

        // Convert histogram to float array
        val features = FloatArray(64).apply {
            var index = 0
            for (h in 0 until 8) {
                for (s in 0 until 8) {
                    this[index++] = hist.get(h, s)[0].toFloat()
                }
            }
        }

        // Calculate color moments (mean and stddev) - fixed version
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(hsvImage, mean, stddev, mask)

        // Extract values from Mats (HSV has 3 channels)
        val meanValues = mean.toArray()
        val stddevValues = stddev.toArray()

        // Create normalized color moments (using Hue [0] and Saturation [1] channels)
        val colorMoments = floatArrayOf(
            (meanValues[0] / 180).toFloat(),   // Hue mean
            (meanValues[1] / 255).toFloat(),   // Saturation mean
            (stddevValues[0] / 180).toFloat(), // Hue stddev
            (stddevValues[1] / 255).toFloat()  // Saturation stddev
        )

        return features + colorMoments
    }

    fun close() {
        eyeCascade = null
    }
}