package com.example.irisrecognition.utils

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.Mat

fun ImageProxy.toMat(): Mat {
    val bitmap = this.toBitmap()
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)
    return mat
}

// Дополнительные утилиты для работы с изображениями