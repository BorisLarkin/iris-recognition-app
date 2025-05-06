package com.example.irisrecognition

import android.app.Application
import org.opencv.android.OpenCVLoader
import timber.log.Timber

class IrisRecognitionApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Timber.plant(Timber.DebugTree()) // Добавьте эту строку

        if (!OpenCVLoader.initDebug()) {
            Timber.e("OpenCV initialization failed!")
        } else {
            Timber.d("OpenCV initialized successfully")
        }
    }
}