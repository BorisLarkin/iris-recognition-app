package com.example.irisrecognition.detection

import kotlin.math.sqrt

class IrisDatabase {
    private val database = mutableMapOf<String, FloatArray>()
    private val MATCH_THRESHOLD = 0.8f // Similarity threshold

    fun addUser(features: FloatArray, name: String) {
        database[name] = features
    }

    fun findBestMatch(features: FloatArray): String? {
        if (database.isEmpty()) return null

        var maxSimilarity = 0f
        var bestMatch: String? = null

        database.forEach { (userId, dbFeatures) ->
            val similarity = cosineSimilarity(features, dbFeatures)
            if (similarity > maxSimilarity && similarity > MATCH_THRESHOLD) {
                maxSimilarity = similarity
                bestMatch = userId
            }
        }

        return bestMatch
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Arrays must have the same length" }

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