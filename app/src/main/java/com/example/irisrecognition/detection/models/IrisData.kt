package com.example.irisrecognition.detection.models

import org.opencv.core.Point
import kotlin.math.sqrt

data class IrisData(
    val center: Point,
    val radius: Float,
    val features: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as IrisData

        if (center != other.center) return false
        if (radius != other.radius) return false
        if (!features.contentEquals(other.features)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = center.hashCode()
        result = 31 * result + radius.hashCode()
        result = 31 * result + features.contentHashCode()
        return result
    }
}