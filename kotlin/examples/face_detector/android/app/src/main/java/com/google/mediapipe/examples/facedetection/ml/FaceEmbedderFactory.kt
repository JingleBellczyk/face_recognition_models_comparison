package com.google.mediapipe.examples.facedetection.ml

import android.content.Context

object FaceEmbedderFactory {

    fun create(context: Context): FaceEmbeddingModel =
        when (FaceRecognitionConfig.mode) {
            FaceModelMode.MOBILE_FACENET -> MobileFaceNetEmbedder(context)
            FaceModelMode.GHOST_FACENET -> GhostFaceNetEmbedder(context)
            FaceModelMode.SERVER_MODEL -> ServerFaceEmbedder()
        }
}
