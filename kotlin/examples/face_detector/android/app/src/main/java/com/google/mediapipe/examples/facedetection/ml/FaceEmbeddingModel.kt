package com.google.mediapipe.examples.facedetection.ml

import android.graphics.Bitmap

interface FaceEmbeddingModel {
    val inputSize: Int
    fun getEmbedding(bitmap: Bitmap): FloatArray?
    fun close() {}
}
