package com.google.mediapipe.examples.facedetection.fragments

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import androidx.navigation.fragment.navArgs
import com.google.mediapipe.examples.facedetection.FaceDetectorHelper
import com.google.mediapipe.examples.facedetection.MainViewModel
import com.google.mediapipe.examples.facedetection.R
import com.google.mediapipe.examples.facedetection.databinding.FragmentCameraBinding
import com.google.mediapipe.examples.facedetection.ml.FaceEmbedderFactory
import com.google.mediapipe.examples.facedetection.ml.FaceEmbeddingModel
import com.google.mediapipe.examples.facedetection.utils.LoginCameraController
import com.google.mediapipe.examples.facedetection.utils.RegistrationCameraController
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraFragment : Fragment(), FaceDetectorHelper.DetectorListener {

    private val TAG = "CameraFragment"

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private val args: CameraFragmentArgs by navArgs()
    private val viewModel: MainViewModel by activityViewModels()

    private lateinit var faceDetectorHelper: FaceDetectorHelper
    private lateinit var faceEmbedder: FaceEmbeddingModel

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var embeddingExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null

    private var registrationController: RegistrationCameraController? = null
    private var loginController: LoginCameraController? = null

    private var isFragmentFinished = false
    private var hasNavigatedToNotes = false

    private var failedAttempts = 0
    private var lastEmbeddingTime = 0L

    private val MAX_ATTEMPTS = 30
    private val EMBEDDING_INTERVAL_MS = 500L
    private val CONFIDENCE_THRESHOLD_REGISTER = 0.96f
    private val CONFIDENCE_THRESHOLD_LOGIN = 0.93f

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        embeddingExecutor = Executors.newSingleThreadExecutor()

        val mode = args.mode
        val registerUserName = if (mode == "register") args.username else ""

        binding.tvRecognizing.text =
            if (mode == "register") "Rejestracja..."
            else "Logowanie..."

        // --- CONTROLLERY ---
        if (mode == "register") {
            registrationController = RegistrationCameraController(
                context = requireContext().applicationContext,
                userName = registerUserName,
                onFinish = {
                    Toast.makeText(requireContext(), "Zarejestrowano", Toast.LENGTH_SHORT).show()
                    val bundle = Bundle().apply {
                        putBoolean("showFaceNotRecognized", false)
                    }
                    findNavController().navigate(R.id.start_fragment, bundle)
                }
            )
        } else {
            loginController = LoginCameraController(
                scope = viewLifecycleOwner.lifecycleScope,
                context = requireContext().applicationContext,
                onAuthenticated = { user ->
                    if (!isAdded) return@LoginCameraController
                    Toast.makeText(requireContext(), "Zalogowano", Toast.LENGTH_SHORT).show()
                    navigateToNotes(user.id, user.name)
                },
                onFailed = {
                    failedAttempts++
                    if (failedAttempts >= MAX_ATTEMPTS) {
                        navigateToFaceNotRecognized()
                    }
                }
            )
        }

        // --- ML INIT (SYNC, PRZED KAMERĄ) ---
        faceEmbedder = FaceEmbedderFactory.create(requireContext())

        faceDetectorHelper = FaceDetectorHelper(
            context = requireContext(),
            threshold = viewModel.currentThreshold,
            currentDelegate = viewModel.currentDelegate,
            faceDetectorListener = this,
            runningMode = RunningMode.LIVE_STREAM
        )

        binding.viewFinder.post { setUpCamera() }
    }

    private fun setUpCamera() {
        val future = ProcessCameraProvider.getInstance(requireContext())
        future.addListener({
            cameraProvider = future.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return

        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, faceDetectorHelper::detectLivestreamFrame)
            }

        provider.unbindAll()
        provider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
    }

    override fun onResults(resultBundle: FaceDetectorHelper.ResultBundle) {
        activity?.runOnUiThread {
            if (!isAdded || _binding == null) return@runOnUiThread
            handleDetections(resultBundle)
            drawOverlay(resultBundle)
        }
    }

    private fun handleDetections(resultBundle: FaceDetectorHelper.ResultBundle) {
        if (isFragmentFinished) return
        val detections = resultBundle.results.firstOrNull()?.detections() ?: return
        if (detections.isEmpty()) return

        val face = detections[0]
        val confidence = face.categories()[0].score()

        if ((args.mode == "register" && confidence < CONFIDENCE_THRESHOLD_REGISTER) ||
            (args.mode == "login" && confidence < CONFIDENCE_THRESHOLD_LOGIN)
        ) return

        val now = System.currentTimeMillis()
        if (now - lastEmbeddingTime < EMBEDDING_INTERVAL_MS) return
        lastEmbeddingTime = now

        val bitmap = faceDetectorHelper.latestBitmap ?: return
        val box = face.boundingBox()

        val size = (maxOf(box.width(), box.height()) * 1.3f).toInt()
        val left = (box.centerX() - size / 2).toInt().coerceIn(0, bitmap.width - size)
        val top = (box.centerY() - size / 2).toInt().coerceIn(0, bitmap.height - size)

        val faceBitmap = Bitmap.createBitmap(bitmap, left, top, size, size)
        val inputSize = faceEmbedder.inputSize

        embeddingExecutor.execute {
            val scaled = Bitmap.createScaledBitmap(faceBitmap, inputSize, inputSize, true)
            val embedding = faceEmbedder.getEmbedding(scaled)

            activity?.runOnUiThread {
                if (!isAdded || _binding == null) return@runOnUiThread
                embedding?.let {
                    if (args.mode == "register") {
                        isFragmentFinished = true      // ⬅️ DODAJ
                        registrationController?.registerUser(it)
                    } else {
                        loginController?.authenticateUser(it)
                    }
                }
            }
        }
    }

    private fun drawOverlay(resultBundle: FaceDetectorHelper.ResultBundle) {
        binding.overlay.setResults(
            resultBundle.results[0],
            resultBundle.inputImageHeight,
            resultBundle.inputImageWidth
        )
        binding.overlay.invalidate()
    }

    override fun onError(error: String, errorCode: Int) {
        Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
    }

    private fun navigateToFaceNotRecognized() {
        if (args.mode == "login") {
            val bundle = Bundle().apply {
                putBoolean("showFaceNotRecognized", true)
            }
            findNavController().navigate(R.id.start_fragment, bundle)
        }
    }

    override fun onPause() {
        super.onPause()
        imageAnalyzer?.clearAnalyzer()
        faceDetectorHelper.clearFaceDetector()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        isFragmentFinished = true

        cameraExecutor.shutdown()
        embeddingExecutor.shutdown()
        embeddingExecutor.awaitTermination(300, TimeUnit.MILLISECONDS)

        faceEmbedder.close()
        cameraProvider?.unbindAll()
        _binding = null
    }
    private fun navigateToNotes(userId: Int, userName: String) {
        if (!isAdded) return

        val navController = findNavController()
        if (navController.currentDestination?.id == R.id.camera_fragment) {
            val action =
                CameraFragmentDirections.actionCameraToNotes(userId, userName)
            navController.navigate(action)
        }
    }

}
