package com.google.mediapipe.examples.facedetection.ml

object FaceRecognitionConfig {

    val mode: FaceModelMode = FaceModelMode.MOBILE_FACENET

    val threshold: Float
        get() = when (mode) {
            FaceModelMode.MOBILE_FACENET -> 8.0f
            FaceModelMode.GHOST_FACENET -> 25.0f
            FaceModelMode.SERVER_MODEL -> 0.40f
        }
}
