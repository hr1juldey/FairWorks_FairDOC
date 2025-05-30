import mesop as me
from typing import List, Literal, Optional
from datetime import datetime
import sqlite3
import utils.path_setup # Ensures paths are set up

Role = Literal["user", "assistant", "system"]

DATABASE_FILE = "msp_chat_state.db"

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_sessions (
            session_id TEXT PRIMARY KEY,
            current_input TEXT,
            is_bot_typing INTEGER, -- boolean stored as 0 or 1
            last_active TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db() # Initialize DB when module is loaded

@me.stateclass
class ChatMessage:
    role: Role = "user"
    content: str = ""
    timestamp: str = ""
    message_id: str = ""

@me.stateclass
class ReportContent:
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
    session_id: str = "default_session" # This should be made dynamic
    report_content: ReportContent = None

def initialize_app_state(session_id: str = "default_session"):
    state = me.state(AppState)
    state.session_id = session_id # Assign session_id
    
    if state.chat_history is None: # Load from DB or initialize
        state.chat_history = load_chat_history_from_db(session_id)
    
    if not state.chat_history: # If still empty after DB load
        add_chat_message("assistant", "Hello! I'm your NHS Digital Triage assistant. How can I help you today?", session_id)
    
    if state.report_content is None:
        state.report_content = ReportContent()
    
    # Load session specific state like current_input
    session_data = load_session_data_from_db(session_id)
    if session_data:
        state.current_input = session_data.get("current_input", "")
        state.is_bot_typing = bool(session_data.get("is_bot_typing", 0))
    else: # If no session data, ensure session is created
        save_session_data_to_db(session_id, state.current_input, state.is_bot_typing)


def add_chat_message(role: Role, content: str, session_id: str):
    state = me.state(AppState)
    timestamp_str = datetime.now().strftime("%I:%M %p").lstrip("0")
    message_id_str = f"{session_id}_{datetime.now().timestamp()}_{len(state.chat_history)}"
    
    new_message = ChatMessage(
        role=role,
        content=content,
        timestamp=timestamp_str,
        message_id=message_id_str
    )
    if state.chat_history is None: # Should be initialized by now, but as a safe guard
        state.chat_history = []
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

def save_session_data_to_db(session_id: str, current_input: str, is_bot_typing: bool):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO app_sessions (session_id, current_input, is_bot_typing, last_active)
        VALUES (?, ?, ?, ?)
    ''', (session_id, current_input, 1 if is_bot_typing else 0, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_session_data_from_db(session_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # Access columns by name
    cursor = conn.cursor()
    cursor.execute('''
        SELECT current_input, is_bot_typing FROM app_sessions
        WHERE session_id = ?
    ''', (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None