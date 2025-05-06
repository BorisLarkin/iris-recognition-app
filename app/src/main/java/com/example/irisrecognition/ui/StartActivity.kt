package com.example.irisrecognition.ui

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import timber.log.Timber

class StartActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Timber.d("StartActivity created")

        setContent {
            StartScreen()
        }
    }
}

@Composable
fun StartScreen() {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(
            onClick = {
                Timber.d("Start button clicked")
                try {
                    context.startActivity(Intent(context, MainActivity::class.java))
                } catch (e: Exception) {
                    Timber.e(e, "Failed to start MainActivity")
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Start Iris Recognition")
        }
    }
}