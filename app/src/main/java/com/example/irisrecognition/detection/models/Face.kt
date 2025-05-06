package com.example.irisrecognition.detection.models

import org.opencv.core.Point
import org.opencv.core.Rect

data class Face(
    val rect: Rect,
    val landmarks: List<Point>
)