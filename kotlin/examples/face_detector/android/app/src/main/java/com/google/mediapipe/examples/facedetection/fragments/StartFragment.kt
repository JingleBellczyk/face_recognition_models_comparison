package com.google.mediapipe.examples.facedetection.fragments

import android.content.Context
import android.content.Context.MODE_PRIVATE
import android.database.sqlite.SQLiteDatabase.openOrCreateDatabase
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.google.android.material.button.MaterialButton
import com.google.mediapipe.examples.facedetection.R
import com.google.mediapipe.examples.facedetection.database.AppDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class StartFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_start, container, false)

        // Przycisk Rejestracja
        val btnRegister = view.findViewById<MaterialButton>(R.id.btnRegister)
        btnRegister.setOnClickListener {
            findNavController().navigate(R.id.registrationStep1Fragment)

//            lifecycleScope.launch(Dispatchers.IO) {
//                clearDatabase(requireContext())
//                // po wyczyszczeniu danych wracamy na główny wątek, żeby nawigować
//                withContext(Dispatchers.Main) {
//                    findNavController().navigate(R.id.registrationStep1Fragment)
//                }
//            }
        }

        // Przycisk Logowanie
        val btnLogin = view.findViewById<MaterialButton>(R.id.btnLogin)
        btnLogin.setOnClickListener {
            findNavController().navigate(R.id.action_start_to_login)
        }

        // Tekst informujący o nieudanej detekcji twarzy
        val tvFaceNotRecognized = view.findViewById<TextView>(R.id.tvFaceNotRecognized)
        // domyślnie niewidoczny
        tvFaceNotRecognized.visibility = View.GONE

        // Sprawdzenie argumentu showFaceNotRecognized z NavArgs
        val showFaceNotRecognized = arguments?.getBoolean("showFaceNotRecognized", false) ?: false
        if (showFaceNotRecognized) {
            tvFaceNotRecognized.visibility = View.VISIBLE
        }

        return view
    }
    fun clearDatabase(context: Context) {
        val db = AppDatabase.getInstance(context)
        db.clearAllTables()  // usuwa wszystkie dane ze wszystkich tabel
        Log.w("USUWAM", "Usunieta baza danych!")

    }
}
