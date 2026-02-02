package com.google.mediapipe.examples.facedetection.utils

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.mediapipe.examples.facedetection.R
import com.google.mediapipe.examples.facedetection.database.notes.NoteEntity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class NotesAdapter(
    private val notes: List<NoteEntity>,
    private val onClick: (NoteEntity) -> Unit,
    private val onLongClick: (NoteEntity) -> Unit
) : RecyclerView.Adapter<NotesAdapter.NoteViewHolder>() {

    class NoteViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleText: TextView = view.findViewById(R.id.noteTitle)
        val contentText: TextView = view.findViewById(R.id.noteContent)
        val timestampText: TextView = view.findViewById(R.id.noteTimestamp)
    }


    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NoteViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_note, parent, false)
        return NoteViewHolder(view)
    }

    override fun onBindViewHolder(holder: NoteViewHolder, position: Int) {
        val note = notes[position]
        holder.titleText.text = note.title        // pole title
        holder.titleText.text =
            if (note.title.length > 70)
                note.title.take(70) + "..."
            else
                note.title
//        holder.contentText.text = note.content    // pole content
        val preview = note.content
            .replace("\n", " ")
            .replace("\r", " ")
            .trim()

        holder.contentText.text =
            if (preview.length > 34)
                preview.take(34) + "..."
            else
                preview

        holder.timestampText.text =
            SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                .format(Date(note.timestamp))

        holder.itemView.setOnClickListener {
            onClick(note)
        }

        holder.itemView.setOnLongClickListener {
            onLongClick(note)
            true
        }

    }

    override fun getItemCount(): Int = notes.size
}
