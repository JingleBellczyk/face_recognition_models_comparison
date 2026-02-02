//package com.google.mediapipe.examples.facedetection.fragments
//
//import android.graphics.Bitmap
//import android.graphics.ImageDecoder
//import android.net.Uri
//import android.os.Build
//import android.os.Bundle
//import android.os.SystemClock
//import android.provider.MediaStore
//import android.view.LayoutInflater
//import android.view.View
//import android.view.ViewGroup
//import android.widget.Toast
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.core.content.ContextCompat
//import androidx.fragment.app.Fragment
//import androidx.fragment.app.activityViewModels
//import androidx.navigation.fragment.findNavController
//import androidx.navigation.ui.setupWithNavController
//import com.google.mediapipe.examples.facedetection.FaceDetectorHelper
//import com.google.mediapipe.examples.facedetection.MainViewModel
//import com.google.mediapipe.examples.facedetection.R
//import com.google.mediapipe.examples.facedetection.databinding.FragmentGalleryBinding
//import com.google.mediapipe.examples.facedetection.ml.FaceEmbedder
//import com.google.mediapipe.examples.facedetection.utils.LoginCameraController
//import com.google.mediapipe.examples.facedetection.utils.RegistrationCameraController
//import com.google.mediapipe.tasks.vision.core.RunningMode
//import java.util.concurrent.Executors
//import java.util.concurrent.ScheduledExecutorService
//import java.util.concurrent.TimeUnit
//
//class GalleryFragment : Fragment(), FaceDetectorHelper.DetectorListener {
//
//    enum class MediaType { IMAGE, VIDEO, UNKNOWN }
//
//    private var _binding: FragmentGalleryBinding? = null
//    private val binding get() = _binding!!
//
//    private lateinit var faceDetectorHelper: FaceDetectorHelper
//    private lateinit var faceEmbedder: FaceEmbedder
//    private lateinit var backgroundExecutor: ScheduledExecutorService
//
//    private val viewModel: MainViewModel by activityViewModels()
//
//    private var embeddingCalculated = false
//    private var lastEmbedding: FloatArray? = null
//
//    private var registrationController: RegistrationCameraController? = null
//    private var loginController: LoginCameraController? = null
//
//    private var currentMediaType: MediaType = MediaType.UNKNOWN
//    private var currentBitmap: Bitmap? = null
//    private var currentVideoUri: Uri? = null
//
//    // ---- wybór pliku ----
//    private val getContent =
//        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
//            uri?.let { mediaUri ->
//                when (val type = loadMediaType(mediaUri)) {
//                    MediaType.IMAGE -> runDetectionOnImage(mediaUri)
//                    MediaType.VIDEO -> runDetectionOnVideo(mediaUri)
//                    MediaType.UNKNOWN -> {
//                        updateDisplayView(MediaType.UNKNOWN)
//                        Toast.makeText(
//                            requireContext(),
//                            "Unsupported data type.",
//                            Toast.LENGTH_SHORT
//                        ).show()
//                    }
//                }
//            }
//        }
//
//    override fun onCreateView(
//        inflater: LayoutInflater,
//        container: ViewGroup?,
//        savedInstanceState: Bundle?
//    ): View {
//        _binding = FragmentGalleryBinding.inflate(inflater, container, false)
//        return binding.root
//    }
//
//    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
//        super.onViewCreated(view, savedInstanceState)
//
//        backgroundExecutor = Executors.newSingleThreadScheduledExecutor()
//        faceEmbedder = FaceEmbedder(requireContext())
//
//        // ---- 1) BottomNavigationView ----
//        binding.navigation.setupWithNavController(findNavController())
//        binding.navigation.setOnNavigationItemReselectedListener { /* ignore */ }
//
//        // ---- 2) FAB ----
//        binding.fabGetContent.setOnClickListener {
//            getContent.launch(
//                arrayOf(
//                    "image/*",
//                    "video/*"
//                )
//            )
//        }
//
//        // ---- 3) Bottom sheet / threshold ----
////        initBottomSheetControls()
//
//        // ---- 4) Przygotowanie przycisku Dalej ----
//        binding.buttonNext.setOnClickListener {
//            if (embeddingCalculated && lastEmbedding != null) {
//                if (registrationController != null) {
//                    registrationController?.registerUser(lastEmbedding!!)
//                    navigateToStartScreen()
//                } else if (loginController != null) {
//                    loginController?.authenticateUser(lastEmbedding!!)
//                }
//            } else {
//                Toast.makeText(
//                    requireContext(),
//                    "Obliczanie embeddingu w toku...",
//                    Toast.LENGTH_SHORT
//                ).show()
//            }
//        }
//    }
//
//    override fun onPause() {
//        super.onPause()
//        binding.overlay.clear()
//        if (binding.videoView.isPlaying) binding.videoView.stopPlayback()
//    }
//
//    override fun onDestroyView() {
//        _binding = null
//        backgroundExecutor.shutdown()
//        super.onDestroyView()
//    }
//
//    // ==================== IMAGE ====================
//    private fun runDetectionOnImage(uri: Uri) {
//        setUiEnabled(false)
//        updateDisplayView(MediaType.IMAGE)
//        currentMediaType = MediaType.IMAGE
//
//        val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            val source = ImageDecoder.createSource(requireContext().contentResolver, uri)
//            ImageDecoder.decodeBitmap(source)
//        } else {
//            MediaStore.Images.Media.getBitmap(requireContext().contentResolver, uri)
//        }.copy(Bitmap.Config.ARGB_8888, true)
//
//        currentBitmap = bitmap
//        binding.imageResult.setImageBitmap(bitmap)
//
//        backgroundExecutor.execute {
//            faceDetectorHelper = FaceDetectorHelper(
//                context = requireContext(),
//                threshold = viewModel.currentThreshold,
//                currentDelegate = viewModel.currentDelegate,
//                runningMode = RunningMode.IMAGE,
//                faceDetectorListener = this
//            )
//            faceDetectorHelper.detectImage(bitmap)?.let { result -> handleDetections(result) }
//            faceDetectorHelper.clearFaceDetector()
//        }
//    }
//
//    // ==================== VIDEO ====================
//    private fun runDetectionOnVideo(uri: Uri) {
//        setUiEnabled(false)
//        updateDisplayView(MediaType.VIDEO)
//        currentMediaType = MediaType.VIDEO
//        currentVideoUri = uri
//
//        with(binding.videoView) {
//            setVideoURI(uri)
//            setOnPreparedListener { it.setVolume(0f, 0f) }
//        }
//
//        backgroundExecutor.execute {
//            faceDetectorHelper = FaceDetectorHelper(
//                context = requireContext(),
//                threshold = viewModel.currentThreshold,
//                currentDelegate = viewModel.currentDelegate,
//                runningMode = RunningMode.VIDEO,
//                faceDetectorListener = this
//            )
//
//            faceDetectorHelper.detectVideoFile(uri, VIDEO_INTERVAL_MS)?.let { result ->
//                displayVideoResult(result)
//            }
//
//            faceDetectorHelper.clearFaceDetector()
//        }
//    }
//
//    private fun displayVideoResult(result: FaceDetectorHelper.ResultBundle) {
//        activity?.runOnUiThread {
//            binding.videoView.visibility = View.VISIBLE
//            binding.progress.visibility = View.GONE
//            binding.videoView.start()
//        }
//
//        val videoStartTimeMs = SystemClock.uptimeMillis()
//
//        backgroundExecutor.scheduleAtFixedRate({
//            activity?.runOnUiThread {
//                val elapsed = SystemClock.uptimeMillis() - videoStartTimeMs
//                val index = (elapsed / VIDEO_INTERVAL_MS).toInt()
//                if (index >= result.results.size) {
//                    backgroundExecutor.shutdown()
//                } else {
//                    handleDetections(result, index)
//                }
//            }
//        }, 0, VIDEO_INTERVAL_MS, TimeUnit.MILLISECONDS)
//    }
//
//    // ==================== DETECTIONS ====================
//    private fun handleDetections(resultBundle: FaceDetectorHelper.ResultBundle, index: Int = 0) {
//        val detectionResult = resultBundle.results[index]
//        val detections = detectionResult.detections()
//        if (detections.isEmpty()) {
//            embeddingCalculated = false
//            lastEmbedding = null
//            setNextButtonEnabled(false)
//            return
//        }
//
//        val face = detections[0]
//        val confidence = face.categories()[0].score()
//        val CONFIDENCE_THRESHOLD = 0.93f
//
//        if (confidence < CONFIDENCE_THRESHOLD) {
//            embeddingCalculated = false
//            lastEmbedding = null
//            setNextButtonEnabled(false)
//            return
//        }
//
//        val bitmap = currentBitmap ?: return
//        val box = face.boundingBox()
//        val left = box.left.coerceAtLeast(0f).toInt()
//        val top = box.top.coerceAtLeast(0f).toInt()
//        val width = box.width().coerceAtMost((bitmap.width - left).toFloat()).toInt()
//        val height = box.height().coerceAtMost((bitmap.height - top).toFloat()).toInt()
//        val faceBitmap = Bitmap.createBitmap(bitmap, left, top, width, height)
//
//        backgroundExecutor.execute {
//            val scaledFace = Bitmap.createScaledBitmap(faceBitmap, 160, 160, true)
//            val embedding = faceEmbedder.getEmbedding(scaledFace)
//            lastEmbedding = embedding
//            embeddingCalculated = true
//
//            activity?.runOnUiThread { setNextButtonEnabled(true) }
//        }
//
//        // overlay
//        activity?.runOnUiThread {
//            binding.overlay.setResults(
//                detectionResult,
//                resultBundle.inputImageHeight,
//                resultBundle.inputImageWidth
//            )
//        }
//    }
//
//    private fun setNextButtonEnabled(enabled: Boolean) {
//        binding.buttonNext.isEnabled = enabled
//        binding.buttonNext.setBackgroundColor(
//            ContextCompat.getColor(
//                requireContext(),
//                if (enabled) R.color.mp_variant else R.color.grey
//            )
//        )
//    }
//
//    // ==================== BOTTOM SHEET / THRESHOLD ====================
////    private fun initBottomSheetControls() {
////        binding.bottomSheetLayout.thresholdMinus.setOnClickListener {
////            if (viewModel.currentThreshold >= 0.1) {
////                viewModel.setThreshold(viewModel.currentThreshold - 0.1f)
////            }
////        }
////        binding.bottomSheetLayout.thresholdPlus.setOnClickListener {
////            if (viewModel.currentThreshold <= 0.8) {
////                viewModel.setThreshold(viewModel.currentThreshold + 0.1f)
////            }
////        }
////    }
//
//    // ==================== UTILS ====================
//    private fun updateDisplayView(type: MediaType) {
//        binding.overlay.clear()
//        binding.imageResult.visibility = if (type == MediaType.IMAGE) View.VISIBLE else View.GONE
//        binding.videoView.visibility = if (type == MediaType.VIDEO) View.VISIBLE else View.GONE
//        binding.tvPlaceholder.visibility =
//            if (type == MediaType.UNKNOWN) View.VISIBLE else View.GONE
//    }
//
//    private fun loadMediaType(uri: Uri): MediaType {
//        val mime = context?.contentResolver?.getType(uri) ?: return MediaType.UNKNOWN
//        return when {
//            mime.startsWith("image") -> MediaType.IMAGE
//            mime.startsWith("video") -> MediaType.VIDEO
//            else -> MediaType.UNKNOWN
//        }
//    }
//
//    private fun setUiEnabled(enabled: Boolean) {
//        binding.fabGetContent.isEnabled = enabled
//    }
//
//    override fun onError(error: String, errorCode: Int) {
//        activity?.runOnUiThread {
//            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
//            setUiEnabled(true)
//            updateDisplayView(MediaType.UNKNOWN)
//        }
//    }
//
//    override fun onResults(resultBundle: FaceDetectorHelper.ResultBundle) {
//        // no-op
//    }
//
//    private fun navigateToStartScreen() {
//        findNavController().popBackStack(R.id.start_fragment, false)
//    }
//
//    companion object {
//        private const val VIDEO_INTERVAL_MS = 300L
//    }
//}
