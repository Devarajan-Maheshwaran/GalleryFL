package com.fgt.galleryfl.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        FeedbackEntity::class,
        OperationLogEntity::class,
        ScanResultEntity::class,
        MediaStateEntity::class
    ],
    version = 3
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun feedbackDao(): FeedbackDao
    abstract fun operationLogDao(): OperationLogDao
    abstract fun scanResultDao(): ScanResultDao
    abstract fun mediaStateDao(): MediaStateDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        // v1 -> v2: add the operation_log table used by Organize / Undo.
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL(
                    "CREATE TABLE IF NOT EXISTS operation_log (" +
                            "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                            "operationType TEXT NOT NULL, " +
                            "timestamp INTEGER NOT NULL, " +
                            "detailsJson TEXT NOT NULL)"
                )
            }
        }

        // v2 -> v3: add scan_result (persisted Scan & Group output) and
        // media_state (Favorites / Archive / Trash) tables.
        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL(
                    "CREATE TABLE IF NOT EXISTS scan_result (" +
                            "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                            "classIndex INTEGER NOT NULL, " +
                            "tagName TEXT NOT NULL, " +
                            "folderPath TEXT NOT NULL, " +
                            "imageUri TEXT NOT NULL, " +
                            "confidence REAL NOT NULL, " +
                            "positionInAlbum INTEGER NOT NULL, " +
                            "avgConfidence REAL NOT NULL, " +
                            "thresholdUsed REAL NOT NULL)"
                )
                database.execSQL(
                    "CREATE TABLE IF NOT EXISTS media_state (" +
                            "uri TEXT NOT NULL PRIMARY KEY, " +
                            "favorite INTEGER NOT NULL DEFAULT 0, " +
                            "archived INTEGER NOT NULL DEFAULT 0, " +
                            "trashed INTEGER NOT NULL DEFAULT 0, " +
                            "updatedAt INTEGER NOT NULL DEFAULT 0)"
                )
            }
        }

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "fgt_database"
                ).addMigrations(MIGRATION_1_2, MIGRATION_2_3).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
