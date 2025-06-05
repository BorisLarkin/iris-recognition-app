import android.content.Context
import android.graphics.BitmapFactory
import com.example.irisrecognition.detection.IrisDetector
import com.example.irisrecognition.detection.models.StoredIris
import timber.log.Timber
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class IrisDatabase(private val context: Context) {
    private val storedIrises = mutableListOf<StoredIris>()

    init {
        loadIrisImages()
    }

    private fun loadIrisImages() {
        try {
            // List all files in assets directory
            val files = context.assets.list("")?.filter {
                it.endsWith(".jpg") || it.endsWith(".png")
            } ?: emptyList()

            files.forEach { filename ->
                // Extract name from filename (e.g., "Jack.jpg" -> "Jack")
                val name = filename.substringBeforeLast('.')

                // Load and process image
                context.assets.open(filename).use { inputStream ->
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                    val irisDetector = IrisDetector(context)

                    irisDetector.detectIris(bitmap) { iris ->
                        iris.leftIris?.let { irisData ->
                            storedIrises.add(StoredIris(name, irisData.features))
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Error loading iris images from assets")
        }
    }

    fun findBestMatch(liveFeatures: FloatArray): Pair<String?, Float> {
        if (storedIrises.isEmpty()) return Pair(null, 0f)

        var bestMatch: String? = null
        var highestSimilarity = 0f

        // Updated feature sizes
        val colorFeatureLength = 18  // 16 (Hue histogram) + 2 (Hue moments)
        val textureFeatureLength = 128 // LBP portion
        val shapeFeatureLength = 128 // Polar coordinates portion

        storedIrises.forEach { stored ->
            // Split features into different types
            val liveColorFeatures = liveFeatures.copyOfRange(
                256, 256 + colorFeatureLength
            )
            val liveTextureFeatures = liveFeatures.copyOfRange(
                128, 128 + textureFeatureLength
            )
            val liveShapeFeatures = liveFeatures.copyOfRange(
                0, shapeFeatureLength
            )

            val storedColorFeatures = stored.features.copyOfRange(
                256, 256 + colorFeatureLength
            )
            val storedTextureFeatures = stored.features.copyOfRange(
                128, 128 + textureFeatureLength
            )
            val storedShapeFeatures = stored.features.copyOfRange(
                0, shapeFeatureLength
            )

            // Calculate similarities with proper normalization
            val colorSim = 1 - histogramDistance(
                liveColorFeatures.take(16).toFloatArray(),  // Histogram part
                storedColorFeatures.take(16).toFloatArray()
            )

            val textureSim = cosineSimilarity(
                liveTextureFeatures.normalize(),
                storedTextureFeatures.normalize()
            )

            val shapeSim = euclideanSimilarity(
                liveShapeFeatures.normalize(),
                storedShapeFeatures.normalize()
            )

            // Weighted combination
            val similarity = (colorSim * COLOR_WEIGHT +
                    textureSim * TEXTURE_WEIGHT +
                    shapeSim * SHAPE_WEIGHT)

            if (similarity > highestSimilarity && similarity >= MIN_CONFIDENCE) {
                highestSimilarity = similarity
                bestMatch = stored.name
            }
        }

        return Pair(bestMatch, highestSimilarity)
    }

    private fun FloatArray.normalize(): FloatArray {
        val norm = sqrt(this.sumOf { it.toDouble().pow(2) }.toFloat())
        return if (norm > 0) this.map { it / norm }.toFloatArray() else this
    }

    private fun histogramDistance(a: FloatArray, b: FloatArray): Float {
        // Proper Earth Mover's Distance approximation for normalized histograms
        require(a.size == b.size) { "Histograms must be of equal size" }

        var distance = 0f
        for (i in a.indices) {
            distance += abs(a[i] - b[i])
        }
        return abs(distance / 2f)  // Normalize to [0,1] range
    }


    private fun euclideanSimilarity(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) {
            sum += (a[i] - b[i]) * (a[i] - b[i])
        }
        val distance = sqrt(sum)
        // Convert distance to similarity (assuming normalized vectors)
        return 1 / (1 + distance)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            // Handle NaN values
            val ai = a[i].coerceIn(-1f, 1f)
            val bi = b[i].coerceIn(-1f, 1f)
            dotProduct += ai * bi
            normA += ai * ai
            normB += bi * bi
        }
        return when {
            normA == 0f || normB == 0f -> 0f
            else -> (dotProduct / (sqrt(normA) * sqrt(normB)))
                .coerceIn(-1f, 1f)
        }
    }

    // Adjust weights and thresholds
    private val COLOR_WEIGHT = 0.4f
    private val TEXTURE_WEIGHT = 0.5f
    private val SHAPE_WEIGHT = 0.10f
    private val MIN_CONFIDENCE = 0.6f
    private val COLOR_SIM_THRESHOLD = 0.6f
}