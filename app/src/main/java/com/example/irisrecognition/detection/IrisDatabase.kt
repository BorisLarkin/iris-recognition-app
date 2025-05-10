import org.opencv.core.Core.sqrt
import kotlin.math.pow
import kotlin.math.sqrt

class IrisDatabase {
    private val database = mutableMapOf<String, FloatArray>()
    private val MATCH_THRESHOLD = 0.7f // Adjusted threshold
    private val MIN_CONFIDENCE = 0.8f // Minimum confidence to consider a match

    fun addUser(features: FloatArray, name: String) {
        database[name] = features.normalize()
    }

    fun findBestMatch(features: FloatArray): Pair<String?, Float> {
        if (database.isEmpty()) return Pair(null, 0f)

        val normalizedFeatures = features.normalize()
        var bestMatch: String? = null
        var highestSimilarity = 0f

        database.forEach { (userId, dbFeatures) ->
            val similarity = cosineSimilarity(normalizedFeatures, dbFeatures)
            if (similarity > highestSimilarity && similarity >= MIN_CONFIDENCE) {
                highestSimilarity = similarity
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