# msp/state/database_manager.py

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .core_states import ChatMessage
from .document_states import UploadedFileState

DATABASE_FILE = "msp_chat_state.db"

def _get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    """Helper function to get column names of a table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

def init_db():
    """Initialize database with all required tables and columns."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create chat_messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            message_id TEXT NOT NULL UNIQUE
        )
    ''')
    
    # Create app_sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_sessions (
            session_id TEXT PRIMARY KEY,
            current_input TEXT,
            is_bot_typing INTEGER,
            last_active TEXT,
            report_data_json TEXT,
            report_is_generating INTEGER,
            report_generation_progress TEXT,
            report_generation_complete INTEGER,
            report_error_message TEXT,
            clinical_analysis_json TEXT,
            ehr_data_json TEXT,
            uploaded_documents_json TEXT,
            patient_data_json TEXT,
            active_report_tab TEXT,
            bias_monitoring_json TEXT
        )
    ''')
    
    # Create clinical_reports table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            report_type TEXT NOT NULL,
            report_data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    ''')
    
    # Create uploaded_files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            file_data BLOB,
            upload_timestamp TEXT NOT NULL,
            file_category TEXT DEFAULT 'medical_document'
        )
    ''')
    
    # Schema migration
    existing_columns = _get_table_columns(cursor, "app_sessions")
    new_columns_to_add = {
        "report_data_json": "TEXT",
        "report_is_generating": "INTEGER",
        "report_generation_progress": "TEXT",
        "report_generation_complete": "INTEGER",
        "report_error_message": "TEXT",
        "clinical_analysis_json": "TEXT",
        "ehr_data_json": "TEXT",
        "uploaded_documents_json": "TEXT",
        "patient_data_json": "TEXT",
        "active_report_tab": "TEXT",
        "bias_monitoring_json": "TEXT"
    }
    
    for col_name, col_type in new_columns_to_add.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE app_sessions ADD COLUMN {col_name} {col_type};")
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column '{col_name}': {e}")
    
    conn.commit()
    conn.close()

def save_message_to_db(session_id: str, message: ChatMessage):
    """Save chat message to database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_messages (session_id, role, content, timestamp, message_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, message.role, message.content, message.timestamp, message.message_id))
    
    conn.commit()
    conn.close()

def load_chat_history_from_db(session_id: str) -> List[ChatMessage]:
    """Load chat history from database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content, timestamp, message_id FROM chat_messages
        WHERE session_id = ? ORDER BY id ASC
    ''', (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [ChatMessage(role=row[0], content=row[1], timestamp=row[2], message_id=row[3]) for row in rows]

def save_comprehensive_session_data_to_db(session_data: Dict[str, Any]):
    """Save complete session state to database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO app_sessions (
            session_id, current_input, is_bot_typing, last_active,
            report_data_json, report_is_generating, report_generation_progress,
            report_generation_complete, report_error_message,
            clinical_analysis_json, ehr_data_json, uploaded_documents_json,
            patient_data_json, active_report_tab, bias_monitoring_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_data["session_id"],
        session_data["current_input"],
        1 if session_data["is_bot_typing"] else 0,
        datetime.now().isoformat(),
        session_data["report_data_json"],
        1 if session_data["report_is_generating"] else 0,
        session_data["report_generation_progress"],
        1 if session_data["report_generation_complete"] else 0,
        session_data["report_error_message"],
        session_data["clinical_analysis_json"],
        session_data["ehr_data_json"],
        session_data["uploaded_documents_json"],
        session_data["patient_data_json"],
        session_data["active_report_tab"],
        session_data["bias_monitoring_json"]
    ))
    
    conn.commit()
    conn.close()

def load_session_data_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session data from database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM app_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None

def save_file_to_db(session_id: str, file_state: UploadedFileState, file_data: bytes):
    """Save uploaded file data to database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO uploaded_files (
            file_id, session_id, original_filename, file_size,
            mime_type, file_data, upload_timestamp, file_category
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        file_state.file_id,
        session_id,
        file_state.original_filename,
        file_state.file_size,
        file_state.mime_type,
        file_data,
        file_state.upload_timestamp,
        file_state.file_category
    ))
    
    conn.commit()
    conn.close()

# Initialize database when module is loaded
init_db()
