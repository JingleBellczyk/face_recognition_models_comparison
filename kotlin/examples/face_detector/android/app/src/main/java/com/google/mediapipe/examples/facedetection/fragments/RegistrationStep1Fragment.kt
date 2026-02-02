package com.google.mediapipe.examples.facedetection.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.google.mediapipe.examples.facedetection.databinding.FragmentRegistrationStep1Binding

class RegistrationStep1Fragment : Fragment() {

    private var _binding: FragmentRegistrationStep1Binding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentRegistrationStep1Binding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Kliknięcie przycisku DALEJ
        binding.btnNext.setOnClickListener {
            val username = binding.inputUsername.text.toString().trim()

            if (username.isEmpty()) {
                binding.inputUsername.error = "Wpisz nazwę użytkownika"
                return@setOnClickListener
            }

            // przejście do ekranu kamerki
            val action = RegistrationStep1FragmentDirections
                .actionRegistrationStep1FragmentToCameraFragment(username)

            findNavController().navigate(action)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
