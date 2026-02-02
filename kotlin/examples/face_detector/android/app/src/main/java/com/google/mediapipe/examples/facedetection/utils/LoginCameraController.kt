package com.google.mediapipe.examples.facedetection.utils

import android.content.Context
import android.util.Log
import com.google.mediapipe.examples.facedetection.database.AppDatabase
import com.google.mediapipe.examples.facedetection.database.user.UserEntity
import com.google.mediapipe.examples.facedetection.ml.FaceRecognitionConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

class LoginCameraController(
    private val scope: CoroutineScope,
    context: Context,
    private val onAuthenticated: (UserEntity) -> Unit,
    private val onFailed: () -> Unit
) {
    private val db = AppDatabase.getInstance(context.applicationContext)
    private val userDao = db.userDao()

    fun authenticateUser(embedding: FloatArray) {
        scope.launch(Dispatchers.IO) {
            val users = userDao.getAllUsers()
            if (users.isEmpty()) {
                withContext(Dispatchers.Main) { onFailed() }
                return@launch
            }

            var bestUser: UserEntity? = null
            var bestDistance = Float.MAX_VALUE

            for (user in users) {
                val dist = l2distance(embedding, user.embedding.toFloatArray())
                if (dist < bestDistance) {
                    bestDistance = dist
                    bestUser = user
                }
            }

            withContext(Dispatchers.Main) {
                if (bestUser != null && bestDistance < FaceRecognitionConfig.threshold) {
                    onAuthenticated(bestUser)
                } else {
                    onFailed()
                }
            }
        }
    }

    private fun l2distance(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }
        return kotlin.math.sqrt(sum)
    }
}
