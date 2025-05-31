
import mesop as me
from typing import List, Literal, Optional
from datetime import datetime
import sqlite3
import json  # Import json for serialization/deserialization
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()

Role = Literal["user", "assistant", "system"]

DATABASE_FILE = "msp_chat_state.db"

def _get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    """Helper function to get column names of a table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Create chat_messages table (if not exists)
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

    # Create app_sessions table (if not exists)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_sessions (
            session_id TEXT PRIMARY KEY,
            current_input TEXT,
            is_bot_typing INTEGER,
            last_active TEXT
            /* New columns will be added below if they don't exist */
        )
    ''')

    # --- Simple Schema Migration for app_sessions table ---
    existing_columns = _get_table_columns(cursor, "app_sessions")
    
    new_columns_to_add = {
        "report_data_json": "TEXT",
        "report_is_generating": "INTEGER",
        "report_generation_progress": "TEXT",
        "report_generation_complete": "INTEGER",
        "report_error_message": "TEXT"
    }

    for col_name, col_type in new_columns_to_add.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE app_sessions ADD COLUMN {col_name} {col_type};")
                print(f"Added column '{col_name}' to 'app_sessions' table.")
            except sqlite3.OperationalError as e:
                # This might happen if the column was partially added or due to other issues
                print(f"Warning: Could not add column '{col_name}': {e}")

    conn.commit()
    conn.close()

init_db()  # Initialize DB when module is loaded

@me.stateclass
class ChatMessage:
    role: Role = "user"
    content: str = ""
    timestamp: str = ""
    message_id: str = ""

class ReportContentStructure:  # Conceptual model, not a stateclass
    title: str = "Medical Assessment Report"
    sections: list[dict] = None
    is_generating: bool = False
    error: Optional[str] = None

@me.stateclass
class AppState:
    current_page: str = "chat"
    chat_history: list[ChatMessage] = None
    current_input: str = ""
    is_bot_typing: bool = False
    session_id: str = "default_session"

    report_data_json: Optional[str] = None
    report_is_generating: bool = False
    report_generation_progress: str = ""
    report_generation_complete: bool = False
    report_error_message: Optional[str] = None
    home_page_viewed: bool = False
    selected_feature: str = ""
    newsletter_subscribed: bool = False

def initialize_app_state(session_id: str = "default_session"):
    state = me.state(AppState)
    state.session_id = session_id

    if state.chat_history is None:
        state.chat_history = load_chat_history_from_db(session_id)

    if not state.chat_history:
        add_chat_message("assistant", "Hello! I'm your NHS Digital Triage assistant. How can I help you today?", session_id)

    session_data = load_session_data_from_db(session_id)
    if session_data:
        state.current_input = session_data.get("current_input", "")
        state.is_bot_typing = bool(session_data.get("is_bot_typing", 0))
        state.report_data_json = session_data.get("report_data_json")
        state.report_is_generating = bool(session_data.get("report_is_generating", 0))
        state.report_generation_progress = session_data.get("report_generation_progress", "")
        state.report_generation_complete = bool(session_data.get("report_generation_complete", 0))
        state.report_error_message = session_data.get("report_error_message")
    else:
        save_session_data_to_db(
            session_id,
            state.current_input,
            state.is_bot_typing,
            state.report_data_json,
            state.report_is_generating,
            state.report_generation_progress,
            state.report_generation_complete,
            state.report_error_message
        )


def add_chat_message(role: Role, content: str, session_id: str):
    state = me.state(AppState)
    if state.chat_history is None:
        state.chat_history = []
        
    timestamp_str = datetime.now().strftime("%I:%M %p").lstrip("0")
    message_id_str = f"{session_id}_{datetime.now().timestamp()}_{len(state.chat_history)}"
    
    new_message = ChatMessage(
        role=role,
        content=content,
        timestamp=timestamp_str,
        message_id=message_id_str
    )
    state.chat_history.append(new_message)
    save_message_to_db(session_id, new_message)

def save_message_to_db(session_id: str, message: ChatMessage):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_messages (session_id, role, content, timestamp, message_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, message.role, message.content, message.timestamp, message.message_id))
    conn.commit()
    conn.close()

def load_chat_history_from_db(session_id: str) -> list[ChatMessage]:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT role, content, timestamp, message_id FROM chat_messages
        WHERE session_id = ? ORDER BY id ASC
    ''', (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [ChatMessage(role=row[0], content=row[1], timestamp=row[2], message_id=row[3]) for row in rows]

def save_session_data_to_db(
    session_id: str,
    current_input: str,
    is_bot_typing: bool,
    report_data_json: Optional[str],
    report_is_generating: bool,
    report_generation_progress: str,
    report_generation_complete: bool,
    report_error_message: Optional[str]
):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Ensure the SQL for INSERT OR REPLACE matches the schema, including all new columns
    cursor.execute('''
        INSERT OR REPLACE INTO app_sessions (
            session_id, current_input, is_bot_typing, last_active,
            report_data_json, report_is_generating, report_generation_progress,
            report_generation_complete, report_error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (  # Ensure 9 placeholders match 9 values
        session_id, current_input, 1 if is_bot_typing else 0, datetime.now().isoformat(),
        report_data_json,
        1 if report_is_generating else 0,
        report_generation_progress,
        1 if report_generation_complete else 0,
        report_error_message
    ))
    conn.commit()
    conn.close()

def load_session_data_from_db(session_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Ensure all columns exist before trying to select them
    existing_columns = [col[1] for col in cursor.execute("PRAGMA table_info(app_sessions);").fetchall()]
    
    # Columns to select based on what's expected by AppState and what actually exists
    select_cols = ["current_input", "is_bot_typing"]
    report_cols_to_check = [
        "report_data_json", "report_is_generating",
        "report_generation_progress", "report_generation_complete",
        "report_error_message"
    ]
    for col in report_cols_to_check:
        if col in existing_columns:
            select_cols.append(col)
        else:  # If column doesn't exist, we won't try to select it
            print(f"Debug: Column '{col}' not found in 'app_sessions' for loading, will use default.")

    query = f"SELECT {', '.join(select_cols)} FROM app_sessions WHERE session_id = ?"
        
    cursor.execute(query, (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_report_state_in_app_and_db(
    report_data_dict: Optional[dict] = None,
    is_generating: Optional[bool] = None,
    generation_progress: Optional[str] = None,
    generation_complete: Optional[bool] = None,
    error_message: Optional[str] = None
):
    state = me.state(AppState)
    
    # Update AppState fields
    if report_data_dict is not None:
        state.report_data_json = json.dumps(report_data_dict)
    if is_generating is not None:
        state.report_is_generating = is_generating
    if generation_progress is not None:
        state.report_generation_progress = generation_progress
    if generation_complete is not None:
        state.report_generation_complete = generation_complete
    # Handle error_message, ensuring it can be None
    state.report_error_message = error_message if error_message is not None else state.report_error_message  # Keep existing if None passed
    if error_message == "":  # Explicitly clear error message if empty string passed
        state.report_error_message = None


    # Persist the updated state to DB
    save_session_data_to_db(
        state.session_id,
        state.current_input,
        state.is_bot_typing,
        state.report_data_json,
        state.report_is_generating,
        state.report_generation_progress,
        state.report_generation_complete,
        state.report_error_message
    )
