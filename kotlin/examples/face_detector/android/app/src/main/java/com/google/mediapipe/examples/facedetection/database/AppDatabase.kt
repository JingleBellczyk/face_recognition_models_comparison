package com.google.mediapipe.examples.facedetection.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.google.mediapipe.examples.facedetection.database.notes.NoteDao
import com.google.mediapipe.examples.facedetection.database.notes.NoteEntity
import com.google.mediapipe.examples.facedetection.database.user.UserDao
import com.google.mediapipe.examples.facedetection.database.user.UserEntity

@Database(
    entities = [UserEntity::class, NoteEntity::class],
    version = 1,
    exportSchema = true
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {

    abstract fun userDao(): UserDao
    abstract fun noteDao(): NoteDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "face_app_db"
                ).build().also { INSTANCE = it }
            }
        }
    }
}
