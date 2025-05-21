package com.example.irisrecognition.detection.models

// models/StoredIris.kt
data class StoredIris(
    val name: String,
    val features: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as StoredIris
        if (name != other.name) return false
        if (!features.contentEquals(other.features)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + features.contentHashCode()
        return result
    }
}