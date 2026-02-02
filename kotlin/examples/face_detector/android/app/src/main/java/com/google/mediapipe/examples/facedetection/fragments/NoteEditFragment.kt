package com.google.mediapipe.examples.facedetection.fragments

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.navArgs
import com.google.mediapipe.examples.facedetection.database.AppDatabase
import com.google.mediapipe.examples.facedetection.database.notes.NoteEntity
import com.google.mediapipe.examples.facedetection.databinding.FragmentNoteEditBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class NoteEditFragment : Fragment() {

    private var _binding: FragmentNoteEditBinding? = null
    private val binding get() = _binding!!

    private val args: NoteEditFragmentArgs by navArgs()

    private var note: NoteEntity? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentNoteEditBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val db = AppDatabase.getInstance(requireContext())

        if (args.noteId == -1) {
            // nowa notatka → puste pola
            note = NoteEntity(
                userId = args.userId, // poprawny użytkownik
                title = "",
                content = "",
                timestamp = System.currentTimeMillis()
            )
        } else {
            // istniejąca notatka
            viewLifecycleOwner.lifecycleScope.launch {
                val loadedNote = withContext(Dispatchers.IO) {
                    db.noteDao().getNoteById(args.noteId)
                }
                note = loadedNote
                binding.titleEdit.setText(loadedNote.title)
                binding.contentEdit.setText(loadedNote.content)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        saveNote()
    }

    private fun saveNote() {
        val title = binding.titleEdit.text.toString().trim()
        val content = binding.contentEdit.text.toString().trim()

        // NIE zapisujemy pustej notatki
        if (title.isEmpty() && content.isEmpty()) return

        val db = AppDatabase.getInstance(requireContext())
        Log.d("userid", "userid: ${args.userId}")


        lifecycleScope.launch(Dispatchers.IO) {
            if (args.noteId == -1) {
                // NOWA NOTATKA
                val newNote = NoteEntity(
                    userId = args.userId,
                    title = title,
                    content = content,
                    timestamp = System.currentTimeMillis()
                )
                // wstawiamy i pobieramy ID wygenerowane przez Room
                val newId = db.noteDao().insertNoteReturnId(newNote).toInt()
                // aktualizujemy note z nowym id
                note = newNote.copy(id = newId)
            } else {
                // ISTNIEJĄCA NOTATKA
                val hasChanged = title != note!!.title || content != note!!.content
                if (!hasChanged) return@launch

                val updated = note!!.copy(
                    title = title,
                    content = content,
                    timestamp = System.currentTimeMillis()
                )
                db.noteDao().updateNote(updated)
                note = updated
            }
        }
    }

    override fun onDestroyView() {
        _binding = null
        super.onDestroyView()
    }
}
