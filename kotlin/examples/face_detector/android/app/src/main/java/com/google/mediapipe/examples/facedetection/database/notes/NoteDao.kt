package com.google.mediapipe.examples.facedetection.database.notes

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update

@Dao
interface NoteDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertNote(note: NoteEntity): Long

    @Query("SELECT * FROM notes WHERE userId = :userId ORDER BY timestamp DESC")
    suspend fun getNotesForUser(userId: Int): List<NoteEntity>

//    @Query("DELETE FROM notes WHERE id = :noteId")
//    suspend fun deleteNote(noteId: Int)

    @Query("SELECT * FROM notes WHERE id = :noteId LIMIT 1")
    suspend fun getNoteById(noteId: Int): NoteEntity

    @Update
    suspend fun updateNote(note: NoteEntity)

    @Delete
    suspend fun deleteNote(note: NoteEntity)

    @Insert
    suspend fun insertNoteReturnId(note: NoteEntity): Long

    @Query("DELETE FROM notes WHERE userId = :userId")
    suspend fun deleteNotesForUser(userId: Int)
}
