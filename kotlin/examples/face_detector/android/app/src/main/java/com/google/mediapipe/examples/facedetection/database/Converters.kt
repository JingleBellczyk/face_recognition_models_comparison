package com.google.mediapipe.examples.facedetection.database

import androidx.room.TypeConverter

object Converters {
    @TypeConverter
    fun fromFloatList(list: List<Float>): String {
        return list.joinToString(",")
    }

    @TypeConverter
    fun toFloatList(data: String): List<Float> {
        if (data.isEmpty()) return emptyList()
        return data.split(",").map { it.toFloat() }
    }
}
