package com.google.mediapipe.examples.facedetection.fragments

import android.widget.Button
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.mediapipe.examples.facedetection.FaceDetectorHelper
import com.google.mediapipe.examples.facedetection.MainViewModel
import com.google.mediapipe.examples.facedetection.R
import com.google.mediapipe.examples.facedetection.OverlayView

abstract class BaseFaceFragment :
    Fragment(),
    FaceDetectorHelper.DetectorListener {

    protected val viewModel: MainViewModel by activityViewModels()

    protected val CONFIDENCE_THRESHOLD = 0.95f

    protected fun handleFaceResults(
        resultBundle: FaceDetectorHelper.ResultBundle,
        overlay: OverlayView,
        onFaceValid: () -> Unit,
        onFaceInvalid: () -> Unit
    ) {
        val detections = resultBundle.results[0].detections()

        if (detections.isEmpty()) {
            onFaceInvalid()
            return
        }

        val confidence = detections[0].categories()[0].score()

        if (confidence >= CONFIDENCE_THRESHOLD) {
            onFaceValid()
        } else {
            onFaceInvalid()
        }

        overlay.setResults(
            resultBundle.results[0],
            resultBundle.inputImageHeight,
            resultBundle.inputImageWidth
        )
        overlay.invalidate()
    }

    protected fun updateNextButton(
        button: Button,
        enabled: Boolean
    ) {
        button.isEnabled = enabled
        button.setBackgroundColor(
            ContextCompat.getColor(
                requireContext(),
                if (enabled) R.color.mp_variant else R.color.grey
            )
        )
    }

    override fun onError(error: String, errorCode: Int) {
        Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
    }
}
