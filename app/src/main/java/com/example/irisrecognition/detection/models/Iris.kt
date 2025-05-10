package com.example.irisrecognition.detection.models

import org.opencv.core.Point

data class Iris(
    var leftIris: IrisData? = null,
    var rightIris: IrisData? = null
)