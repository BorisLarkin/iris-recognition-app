package com.example.irisrecognition.detection.models

import org.opencv.core.Point

data class Iris(
    val leftIris: IrisData?,
    val rightIris: IrisData?
)