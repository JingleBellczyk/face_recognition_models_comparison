package com.google.mediapipe.examples.facedetection.utils

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.examples.facedetection.database.AppDatabase
import com.google.mediapipe.examples.facedetection.database.user.UserEntity
import com.google.mediapipe.examples.facedetection.ml.FaceEmbeddingModel
import com.google.mediapipe.examples.facedetection.ml.ServerFaceEmbedder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class RegistrationCameraController(
    private val context: Context,
    private val userName: String,
    private val onFinish: () -> Unit
) {

    /** NOWA metoda - przyjmuje gotowy embedding */
    fun registerUser(embedding: FloatArray) {
        saveToDatabase(userName, embedding)
        onFinish()
    }

    private fun saveToDatabase(name: String, emb: FloatArray) {
        Log.d("REGISTER", "Zapisuję użytkownika: $name")
        Log.d("REGISTER", "Embedding: ${emb.take(5)} ...")

        val db = AppDatabase.getInstance(context)
        val userEntity = UserEntity(
            id = 0,  // Room automatycznie wygeneruje ID, jeśli używasz autoGenerate = true
            name = name,
            embedding = emb.toList() // FloatArray -> List<Float> dla Room
        )

        CoroutineScope(Dispatchers.IO).launch {
            db.userDao().insertUser(userEntity)
        }
    }
}
