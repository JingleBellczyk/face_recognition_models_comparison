package com.google.mediapipe.examples.facedetection.ml

import android.content.Context
import android.graphics.Bitmap
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.TimeUnit

class MobileFaceNetEmbedder(context: Context) : FaceEmbeddingModel {

    private val interpreter: Interpreter
    override val inputSize = 160
    private val outputSize = 128
    private val intValues = IntArray(inputSize * inputSize)

    init {
        val afd = context.assets.openFd("MobileFaceNet.tflite")
        val buffer = FileInputStream(afd.fileDescriptor)
            .channel
            .map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)

        interpreter = Interpreter(buffer)
    }

    override fun getEmbedding(bitmap: Bitmap): FloatArray {
        val input = preprocess(bitmap)
        val output = Array(1) { FloatArray(outputSize) }
        interpreter.run(input, output)
        return output[0]
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        scaled.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in intValues) {
            buffer.putFloat(((pixel shr 16 and 0xFF) - 127.5f) / 128f)
            buffer.putFloat(((pixel shr 8 and 0xFF) - 127.5f) / 128f)
            buffer.putFloat(((pixel and 0xFF) - 127.5f) / 128f)
        }
        return buffer
    }

    override fun close() = interpreter.close()
}
class GhostFaceNetEmbedder(context: Context) : FaceEmbeddingModel {

    private val interpreter: Interpreter
    override val inputSize = 112
    private val outputSize = 512
    private val imgData =
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())

    private val intValues = IntArray(inputSize * inputSize)

    init {
        val afd = context.assets.openFd("ghostfacenet_optimized.tflite")
        val buffer = FileInputStream(afd.fileDescriptor)
            .channel
            .map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)

        interpreter = Interpreter(buffer, Interpreter.Options().setNumThreads(4))
    }

    override fun getEmbedding(bitmap: Bitmap): FloatArray {
        preprocess(bitmap)
        val output = Array(1) { FloatArray(outputSize) }
        interpreter.run(imgData, output)
        return normalizeL2(output[0])
    }

    private fun preprocess(bitmap: Bitmap) {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        imgData.rewind()
        scaled.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (p in intValues) {
            imgData.putFloat((p shr 16 and 0xFF) / 255f)
            imgData.putFloat((p shr 8 and 0xFF) / 255f)
            imgData.putFloat((p and 0xFF) / 255f)
        }
    }

    private fun normalizeL2(embedding: FloatArray): FloatArray {
        var sum = 0.0f
        for (v in embedding) sum += v * v
        val norm = Math.sqrt(sum.toDouble()).toFloat()
        if (norm > 0) {
            for (i in embedding.indices) embedding[i] /= norm
        }
        return embedding
    }

    override fun close() = interpreter.close()
}
class ServerFaceEmbedder() : FaceEmbeddingModel {

    private val serverUrl = "http://192.168.0.101:5000/get_embedding"
    override val inputSize = 112

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    override fun getEmbedding(bitmap: Bitmap): FloatArray? {
        val jpeg = preprocess(bitmap)

        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                "face.jpg",
                jpeg.toRequestBody("image/jpeg".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder().url(serverUrl).post(body).build()
        val response = client.newCall(request).execute()

        if (!response.isSuccessful) return null

        val json = JSONObject(response.body!!.string())
        val arr = json.getJSONArray("embedding")

        return FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
    }

    private fun preprocess(bitmap: Bitmap): ByteArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val stream = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 95, stream)
        return stream.toByteArray()
    }

    override fun close() {
        client.dispatcher.executorService.shutdown()
    }
}
