import android.content.Context
import android.graphics.BitmapFactory
import com.example.irisrecognition.detection.IrisDetector
import com.example.irisrecognition.detection.models.StoredIris
import timber.log.Timber
import kotlin.math.pow
import kotlin.math.sqrt

class IrisDatabase(private val context: Context) {
    private val storedIrises = mutableListOf<StoredIris>()
    private val SHAPE_WEIGHT = 0.4f // Weight for shape features
    private val COLOR_WEIGHT = 0.6f // Weight for color features
    private val MIN_CONFIDENCE = 0.8f // Minimum confidence to consider a match

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

        val normalizedLive = liveFeatures.normalize()
        var bestMatch: String? = null
        var highestSimilarity = 0f

        storedIrises.forEach { stored ->
            val similarity = cosineSimilarity(normalizedLive, stored.features.normalize())
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

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Feature vectors must have same length" }
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        return dotProduct / (sqrt(normA) * sqrt(normB))
    }
}