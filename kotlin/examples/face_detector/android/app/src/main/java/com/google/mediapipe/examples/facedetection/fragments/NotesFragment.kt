package com.google.mediapipe.examples.facedetection.fragments

import android.app.AlertDialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.OnBackPressedCallback
import androidx.core.os.bundleOf
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import androidx.navigation.fragment.navArgs
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.mediapipe.examples.facedetection.database.AppDatabase
import com.google.mediapipe.examples.facedetection.database.notes.NoteEntity
import com.google.mediapipe.examples.facedetection.databinding.FragmentNotesBinding
import com.google.mediapipe.examples.facedetection.utils.NotesAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.google.mediapipe.examples.facedetection.R
import android.widget.PopupMenu
import android.widget.Toast

class NotesFragment : Fragment() {

    private var _binding: FragmentNotesBinding? = null
    private val binding get() = _binding!!

    private val args: NotesFragmentArgs by navArgs() // userId i opcjonalnie userName

    private var backPressCount = 0

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentNotesBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Nagłówek z nazwą użytkownika
        val userName = args.userName ?: "Użytkownik"
        binding.tvUserNotesHeader.text = "Notatki użytkownika $userName"

        // RecyclerView
        binding.notesRecyclerView.layoutManager = LinearLayoutManager(requireContext())

        // Wczytanie notatek
        loadNotesForUser(args.userId)

        // FloatingActionButton – dodawanie notatki
        binding.fabAddNote.setOnClickListener {
            val bundle = bundleOf(
                "noteId" to -1,
                "userId" to args.userId
            )
            findNavController().navigate(R.id.noteEditFragment, bundle)
        }

        // Blokada BACK jeśli NotesFragment jest startDestination
        requireActivity().onBackPressedDispatcher.addCallback(
            viewLifecycleOwner,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    // blokujemy back
                }
            }
        )

        //
        binding.btnMenu.setOnClickListener {
            showMenu(it)
        }

        requireActivity().onBackPressedDispatcher.addCallback(
            viewLifecycleOwner,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    backPressCount++
                    if (backPressCount >= 2) {
                        logoutUser() // wylogowanie po 2 kliknięciach
                    } else {
                        // opcjonalnie: pokaz toast informacyjny
                        Toast.makeText(
                            requireContext(),
                            "Naciśnij jeszcze raz, aby się wylogować",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        )
    }

    override fun onResume() {
        super.onResume()
        loadNotesForUser(args.userId)
    }

    private fun loadNotesForUser(userId: Int) {
        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
            val notes = AppDatabase.getInstance(requireContext())
                .noteDao()
                .getNotesForUser(userId)

            withContext(Dispatchers.Main) {
                showNotes(notes)
            }
        }
    }

    private fun showNotes(notes: List<NoteEntity>) {
        if (notes.isEmpty()) {
            binding.tvNoNotes.visibility = View.VISIBLE
            binding.notesRecyclerView.visibility = View.GONE
        } else {
            binding.tvNoNotes.visibility = View.GONE
            binding.notesRecyclerView.visibility = View.VISIBLE
            binding.notesRecyclerView.adapter = NotesAdapter(
                notes,
                onClick = { note ->
                    val bundle = bundleOf(
                        "noteId" to note.id,
                        "userId" to args.userId
                    )
                    findNavController().navigate(R.id.noteEditFragment, bundle)
                },
                onLongClick = { note ->
                    showDeleteDialog(note)
                }
            )
        }
    }

    private fun showDeleteDialog(note: NoteEntity) {
        AlertDialog.Builder(requireContext())
            .setTitle("Usuń notatkę")
            .setMessage("Czy na pewno chcesz usunąć tę notatkę?")
            .setPositiveButton("Usuń") { _, _ ->
                deleteNote(note)
            }
            .setNegativeButton("Anuluj", null)
            .show()
    }

    private fun deleteNote(note: NoteEntity) {
        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
            val db = AppDatabase.getInstance(requireContext())
            db.noteDao().deleteNote(note)

            val notes = db.noteDao().getNotesForUser(args.userId)
            withContext(Dispatchers.Main) {
                showNotes(notes)
            }
        }
    }

    override fun onDestroyView() {
        _binding = null
        super.onDestroyView()
    }

    private fun showMenu(anchor: View) {
        val popup = PopupMenu(requireContext(), anchor)
        popup.menuInflater.inflate(R.menu.notes_menu, popup.menu)

        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_logout -> {
                    logoutUser()
                    true
                }
                R.id.action_delete_account -> {
                    showDeleteAccountDialog()
                    true
                }
                else -> false
            }
        }
        popup.show()
    }

    private fun logoutUser() {
        // TODO: wyczyść sesję / zapis loginu
        findNavController().navigate(R.id.action_notesFragment_to_startFragment)
    }

    private fun showDeleteAccountDialog() {
        AlertDialog.Builder(requireContext())
            .setTitle("Usuń konto")
            .setMessage("Czy na pewno chcesz usunąć konto?")
            .setPositiveButton("Tak") { _, _ ->
                deleteAccount()
            }
            .setNegativeButton("Nie", null)
            .show()
    }

    private fun deleteAccount() {
        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
            val db = AppDatabase.getInstance(requireContext())

            // USUŃ NOTATKI
            db.noteDao().deleteNotesForUser(args.userId)

            // USUŃ USERA (jeśli masz UserDao)
            db.userDao().deleteUserById(args.userId)

            withContext(Dispatchers.Main) {
                findNavController().navigate(R.id.action_notesFragment_to_startFragment)
            }
        }
    }

}

