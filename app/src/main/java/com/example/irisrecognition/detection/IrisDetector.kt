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

            // Convert to grayscale with CLAHE
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(gray, gray)

            // Blur to reduce noise
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

            // Edge detection
            val edges = Mat()
            Imgproc.Canny(gray, edges, 50.0, 150.0)

            // Hough Circles with optimized parameters
            val circles = Mat()
            Imgproc.HoughCircles(
                edges, circles, Imgproc.HOUGH_GRADIENT,
                1.2,  // dp
                gray.rows() / 8.0,  // minDist
                100.0,  // param1
                30.0,   // param2
                10,     // minRadius
                50      // maxRadius
            )

            // Process results
            val irises = mutableListOf<IrisData>()
            for (i in 0 until circles.cols()) {
                val circle = circles.get(0, i)
                val center = Point(circle[0], circle[1])
                val radius = circle[2].toFloat()

                // Only accept circles in upper half of image
                if (center.y < image.height * 0.6) {
                    irises.add(IrisData(center, radius, extractIrisFeatures(gray, center, radius)))
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
        // More robust feature extraction using polar coordinates
        val features = FloatArray(256)
        val steps = 16 // angular steps
        val rings = 8 // radial steps

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

        return features
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

    fun close() {
        eyeCascade = null
    }
}