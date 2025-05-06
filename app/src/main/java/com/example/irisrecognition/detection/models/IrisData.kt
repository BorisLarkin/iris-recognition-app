package com.example.irisrecognition.detection.models

import org.opencv.core.Point
import kotlin.math.sqrt

data class IrisData(
    val center: Point,
    val radius: Float,
    val features: FloatArray,
    val confidence: Float = 0.9f
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as IrisData

        if (center != other.center) return false
        if (radius != other.radius) return false
        if (!features.contentEquals(other.features)) return false
        if (confidence != other.confidence) return false

        return true
    }

    override fun hashCode(): Int {
        var result = center.hashCode()
        result = 31 * result + radius.hashCode()
        result = 31 * result + features.contentHashCode()
        result = 31 * result + confidence.hashCode()
        return result
    }

    fun distanceTo(other: IrisData): Float {
        require(features.size == other.features.size) {
            "Feature vectors must have the same length"
        }

        var sum = 0f
        for (i in features.indices) {
            val diff = features[i] - other.features[i]
            sum += diff * diff
        }

        return sqrt(sum)
    }

    companion object {
        const val FEATURE_SIZE = 128
    }
}