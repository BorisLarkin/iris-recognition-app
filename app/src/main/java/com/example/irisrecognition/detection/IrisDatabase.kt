import org.opencv.core.Core.sqrt
import kotlin.math.pow
import kotlin.math.sqrt

class IrisDatabase {
    private val database = mutableMapOf<String, Pair<FloatArray, FloatArray>>() // Stores both shape and color features separately
    private val SHAPE_WEIGHT = 0.4f // Weight for shape features
    private val COLOR_WEIGHT = 0.6f // Weight for color features
    private val MIN_CONFIDENCE = 0.8f // Minimum confidence to consider a match

    fun addUser(features: FloatArray, name: String) {
        // Split features into shape (first 256) and color (remaining 64)
        val shapeFeatures = features.copyOfRange(0, 256).normalize()
        val colorFeatures = features.copyOfRange(256, features.size).normalize()
        database[name] = Pair(shapeFeatures, colorFeatures)
    }

    fun findBestMatch(features: FloatArray): Pair<String?, Float> {
        if (database.isEmpty()) return Pair(null, 0f)

        // Split input features
        val inputShape = features.copyOfRange(0, 256).normalize()
        val inputColor = features.copyOfRange(256, features.size).normalize()

        var bestMatch: String? = null
        var highestSimilarity = 0f

        database.forEach { (userId, dbFeatures) ->
            val shapeSimilarity = cosineSimilarity(inputShape, dbFeatures.first)
            val colorSimilarity = cosineSimilarity(inputColor, dbFeatures.second)

            // Weighted combination of both similarities
            val combinedSimilarity = (shapeSimilarity * SHAPE_WEIGHT) + (colorSimilarity * COLOR_WEIGHT)

            if (combinedSimilarity > highestSimilarity && combinedSimilarity >= MIN_CONFIDENCE) {
                highestSimilarity = combinedSimilarity
                bestMatch = userId
            }
        }

        return Pair(bestMatch, highestSimilarity)
    }

    private fun FloatArray.normalize(): FloatArray {
        val norm = sqrt(this.sumOf { it.toDouble().pow(2) }.toFloat())
        return if (norm > 0) this.map { it / norm }.toFloatArray() else this
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Arrays must have same length" }
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