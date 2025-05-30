import os #noqa
from pathlib import Path

# --- File Contents ---

PATH_SETUP_PY_CONTENT = """
import sys
import os
from pathlib import Path

_APP_ROOT_SETUP_DONE = False

def setup_project_paths():
    global _APP_ROOT_SETUP_DONE
    if _APP_ROOT_SETUP_DONE:
        return

    # Assuming this file is in msp/utils/
    # Project root is two levels up from here (msp/)
    project_root = Path(__file__).resolve().parent.parent
    
    paths_to_add = [
        str(project_root),  # Adds 'msp' directory to sys.path
    ]
    
    for path_str in paths_to_add:
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            # print(f"Added to sys.path: {path_str}") # Optional: for debugging

    _APP_ROOT_SETUP_DONE = True

# Call setup when this module is imported for the first time
setup_project_paths()
"""

WHATSAPP_DARK_THEME_PY_CONTENT = """
import mesop as me

# WhatsApp Web Dark Mode Palette (approximated)
WHATSAPP_DARK_COLORS = {
    "app_bg": "#0B141A",
    "sidebar_bg": "#111B21",
    "header_bg": "#202C33",
    "active_chat_bg": "#2A3942",
    "message_sent_bg": "#005C4B",
    "message_sent_text": "#E9EDEF",
    "message_received_bg": "#202C33",
    "message_received_text": "#E9EDEF",
    "input_field_bg": "#2A3942",
    "input_field_text": "#E9EDEF",
    "input_placeholder_text": "#8696A0",
    "icon_color": "#AEBAC1",
    "icon_active_color": "#00A884", # WhatsApp Green for send
    "timestamp_text": "#8696A0",
    "link_text": "#53BDEB",
    "divider_color": "#2C3E46",
    "primary_text": "#E9EDEF",
    "secondary_text": "#AEBAC1",
    "green_accent": "#00A884", # Used for active send icon
    "error_text": "#F37A7A", # A common error red for dark themes
    "error_bg": "#4B2226",  # Dark red background for error messages
}

# --- Core Layout Styles ---
def app_container_style():
    return me.Style(
        height="100vh",
        width="100vw",
        display="flex",
        flex_direction="column",
        background=WHATSAPP_DARK_COLORS["app_bg"],
        font_family="'Segoe UI', Helvetica, Arial, sans-serif",
        color=WHATSAPP_DARK_COLORS["primary_text"],
        overflow="hidden"
    )

def chat_area_style():
    return me.Style(
        flex_grow=1,
        display="flex",
        flex_direction="column",
        overflow_y="auto",
        padding=me.Padding.symmetric(horizontal=0, vertical=10), # No horizontal padding here
        background=WHATSAPP_DARK_COLORS["app_bg"]
    )

# --- Chat Header Styles ---
def chat_header_style():
    return me.Style(
        background=WHATSAPP_DARK_COLORS["header_bg"],
        padding=me.Padding.symmetric(horizontal=16, vertical=10),
        display="flex",
        align_items="center",
        justify_content="space-between",
        border=me.Border(bottom=me.BorderSide(width=1, style="solid", color=WHATSAPP_DARK_COLORS["divider_color"]))
    )

def chat_header_avatar_style():
    return me.Style(
        width=40, 
        height=40, 
        border_radius="50%", 
        background=WHATSAPP_DARK_COLORS["active_chat_bg"],
        margin=me.Margin(right=12)
    )

def chat_header_text_style():
    return me.Style(font_size="16px", font_weight="500", color=WHATSAPP_DARK_COLORS["primary_text"])

def chat_header_status_style():
    return me.Style(font_size="13px", color=WHATSAPP_DARK_COLORS["secondary_text"])

def chat_header_icons_style():
    return me.Style(display="flex", gap=20)

# --- Chat Bubble Styles ---
def message_bubble_base_style():
    return me.Style(
        padding=me.Padding(top=6, bottom=8, left=9, right=9),
        border_radius=8,
        max_width="65%",
        box_shadow="0 1px 0.5px rgba(0,0,0,0.3)",
        margin=me.Margin(bottom=3)
    )

def sent_bubble_style():
    return me.Style(
        background=WHATSAPP_DARK_COLORS["message_sent_bg"],
        color=WHATSAPP_DARK_COLORS["message_sent_text"],
        align_self="flex-end",
        margin=me.Margin(left="auto", bottom=3, right=10, top=3), # Added top margin
        # Inherit base styles, Pydantic v1 way; adjust if Mesop's Style is not a simple dict
        **{k: v for k, v in message_bubble_base_style().__dict__.items() if not k.startswith('_')}
    )


def received_bubble_style():
    return me.Style(
        background=WHATSAPP_DARK_COLORS["message_received_bg"],
        color=WHATSAPP_DARK_COLORS["message_received_text"],
        align_self="flex-start",
        margin=me.Margin(right="auto", bottom=3, left=10, top=3), # Added top margin
        # Inherit base styles
        **{k: v for k, v in message_bubble_base_style().__dict__.items() if not k.startswith('_')}
    )

def message_text_style():
    return me.Style(font_size="14.2px", line_height="19px", white_space="pre-wrap", word_wrap="break-word")

def timestamp_style():
    return me.Style(
        font_size="11px",
        color=WHATSAPP_DARK_COLORS["timestamp_text"],
        margin=me.Margin(top=4, left=8),
        align_self="flex-end"
    )

# --- Chat Input Styles ---
def chat_input_bar_style():
    return me.Style(
        background=WHATSAPP_DARK_COLORS["header_bg"],
        padding=me.Padding.symmetric(horizontal=10, vertical=10),
        display="flex",
        align_items="center",
        gap=12,
        width="100%" # Ensure it spans full width
    )

def input_field_wrapper_style():
    return me.Style(
        flex_grow=1,
        background=WHATSAPP_DARK_COLORS["input_field_bg"],
        border_radius=8, # WhatsApp uses less rounded input fields
        padding=me.Padding.symmetric(horizontal=12),
        display="flex",
        align_items="center",
        height=42 # Typical height for WhatsApp input area
    )

def native_textarea_style():
    return me.Style(
        background="transparent",
        border=me.Border.all(me.BorderSide(style="none")),
        outline="none",
        width="100%",
        color=WHATSAPP_DARK_COLORS["input_field_text"],
        font_size="15px",
        line_height="20px",
        overflow_y="hidden", # Manage scroll within the single line if needed
        padding=me.Padding(top=8, bottom=8), # Adjust to vertically center text
        height="22px", # To make it appear single line
        vertical_align="middle",
        font_family="'Segoe UI', Helvetica, Arial, sans-serif"
    )

def icon_button_style():
    return me.Style(padding=me.Padding.all(8), cursor="pointer")
"""

STATE_MANAGER_PY_CONTENT = """
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
"""

CHAT_HEADER_PY_CONTENT = """
import mesop as me
import utils.path_setup # Ensures paths are set up
from styles.whatsapp_dark_theme import (
    WHATSAPP_DARK_COLORS, 
    chat_header_style, 
    chat_header_avatar_style,
    chat_header_text_style,
    chat_header_status_style,
    chat_header_icons_style
)

@me.component
def render_chat_header(contact_name: str, status: str):
    with me.box(style=chat_header_style()):
        with me.box(style=me.Style(display="flex", align_items="center")):
            me.box(style=chat_header_avatar_style()) 
            with me.box():
                me.text(contact_name, style=chat_header_text_style())
                me.text(status, style=chat_header_status_style())
        
        with me.box(style=chat_header_icons_style()):
            me.icon("search", style=me.Style(color=WHATSAPP_DARK_COLORS["icon_color"], cursor="pointer"))
            me.icon("more_vert", style=me.Style(color=WHATSAPP_DARK_COLORS["icon_color"], cursor="pointer"))
"""

CHAT_BUBBLE_PY_CONTENT = """
import mesop as me
import utils.path_setup # Ensures paths are set up
from styles.whatsapp_dark_theme import (
    sent_bubble_style, 
    received_bubble_style, 
    message_text_style, 
    timestamp_style,
    WHATSAPP_DARK_COLORS
)
from state.state_manager import ChatMessage

@me.component
def render_chat_bubble(message: ChatMessage):
    is_user = message.role == "user"
    bubble_style_func = sent_bubble_style if is_user else received_bubble_style

    with me.box(style=bubble_style_func()):
        me.text(message.content, style=message_text_style())
        with me.box(style=me.Style(display="flex", align_items="center", justify_content="flex-end", margin=me.Margin(top=5))):
            me.text(message.timestamp, style=timestamp_style())
            if is_user:
                me.icon("done_all", style=me.Style(font_size="16px", color=WHATSAPP_DARK_COLORS["green_accent"], margin=me.Margin(left=4)))
"""

CHAT_INPUT_PY_CONTENT = """
import mesop as me
import utils.path_setup # Ensures paths are set up
from styles.whatsapp_dark_theme import (
    WHATSAPP_DARK_COLORS,
    chat_input_bar_style, 
    input_field_wrapper_style,
    native_textarea_style,
    icon_button_style
)
from state.state_manager import AppState

InputEventHandler = me.InputEventHandler
BlurEventHandler = me.InputBlurEventHandler
ClickEventHandler = me.MouseEventHandler

@me.component
def render_chat_input(
    current_input_value: str,
    on_input_change: InputEventHandler,
    on_input_blur: BlurEventHandler,
    on_send_click: ClickEventHandler,
    is_bot_typing: bool
):
    with me.box(style=chat_input_bar_style()):
        with me.content_button(type="icon", style=icon_button_style()):
            me.icon("mood", style=me.Style(color=WHATSAPP_DARK_COLORS["icon_color"]))
        
        with me.content_button(type="icon", style=icon_button_style()):
            me.icon("attach_file", style=me.Style(color=WHATSAPP_DARK_COLORS["icon_color"]))

        with me.box(style=input_field_wrapper_style()):
            me.native_textarea(
                key="chat_main_input", # Stable key
                value=current_input_value,
                placeholder="Type a message",
                autosize=False,
                min_rows=1,
                max_rows=1,
                style=native_textarea_style(),
                on_input=on_input_change,
                on_blur=on_input_blur,
                # Added on_enter for quick send
                shortcuts={me.Shortcut(key="Enter"): on_send_click}
            )
        
        can_send = current_input_value.strip() != "" and not is_bot_typing
        send_icon_color = WHATSAPP_DARK_COLORS["green_accent"] if can_send else WHATSAPP_DARK_COLORS["icon_color"]
        
        with me.content_button(
            type="icon", 
            on_click=on_send_click, 
            disabled=not can_send,
            style=icon_button_style()
        ):
            me.icon("send", style=me.Style(color=send_icon_color))
"""

CHAT_PAGE_PY_CONTENT = """
import mesop as me
import asyncio # For me.effects
import utils.path_setup 

from components.chat_header import render_chat_header
from components.chat_bubble import render_chat_bubble
from components.chat_input import render_chat_input
from state.state_manager import AppState, add_chat_message, initialize_app_state, save_session_data_to_db
from styles.whatsapp_dark_theme import chat_area_style, WHATSAPP_DARK_COLORS
from llm_mock import get_ollama_response

@me.page(path="/chat", title="WhatsApp Chat") # Ensure it's registered as a page
def chat_page_content(): # Renamed to avoid conflict with module name if any
    # Session ID should be managed, e.g., from URL param or global state for simplicity
    # For now, using the default_session from AppState
    state = me.state(AppState)
    initialize_app_state(state.session_id) # Pass current session_id or default

    render_chat_header(contact_name="NHS AI Assistant", status="Online")

    with me.box(style=chat_area_style(), key="chat_scroll_area"): # Key for scrolling
        if state.chat_history:
            for message_data in state.chat_history:
                render_chat_bubble(message=message_data)
        
        if state.is_bot_typing:
            with me.box(style=me.Style(align_self="flex-start", margin=me.Margin(left=10, bottom=5))):
                 me.text("Assistant is typing...", style=me.Style(font_style="italic", font_size="12px", color=WHATSAPP_DARK_COLORS["secondary_text"]))
        
        # Empty box at the end to scroll to
        with me.box(key="chat_end_anchor", style=me.Style(height=1)):
            pass


    render_chat_input(
        current_input_value=state.current_input,
        on_input_change=handle_input_change,
        on_input_blur=handle_input_blur,
        on_send_click=handle_send_message_click, # Pass the correct handler
        is_bot_typing=state.is_bot_typing
    )
    
    # Scroll to bottom effect
    me.scroll_into_view(selector="#chat_end_anchor", options={"behavior": "smooth", "block": "end"})


def handle_input_change(event: me.InputEvent):
    state = me.state(AppState)
    state.current_input = event.value
    # No yield, save to DB on blur or send

def handle_input_blur(event: me.InputBlurEvent): # Corrected type
    state = me.state(AppState)
    state.current_input = event.value
    save_session_data_to_db(state.session_id, state.current_input, state.is_bot_typing)
    yield # Allow UI to update if needed (e.g., send button state)

def handle_send_message_click(event: me.ClickEvent): # Or me.TextareaShortcutEvent if from Enter
    state = me.state(AppState)
    user_message_content = state.current_input.strip()

    if not user_message_content:
        return

    add_chat_message(role="user", content=user_message_content, session_id=state.session_id)
    state.current_input = "" 
    state.is_bot_typing = True
    save_session_data_to_db(state.session_id, state.current_input, state.is_bot_typing) # Save cleared input
    yield 

    # Prepare history for LLM (ensure it's serializable if needed by get_ollama_response)
    llm_history = [{"role": msg.role, "content": msg.content} for msg in state.chat_history[:-1]]

    bot_response_content = yield from me.effects(
        lambda: get_ollama_response(user_message_content, llm_history), ()
    )

    add_chat_message(role="assistant", content=bot_response_content, session_id=state.session_id)
    state.is_bot_typing = False
    save_session_data_to_db(state.session_id, state.current_input, state.is_bot_typing)
    yield 
"""

REPORT_PAGE_PY_CONTENT = """
import mesop as me
import utils.path_setup 
from state.state_manager import AppState, ReportContent
from styles.whatsapp_dark_theme import WHATSAPP_DARK_COLORS

@me.page(path="/report", title="Medical Report")
def report_page_content(): # Renamed
    state = me.state(AppState)
    if state.report_content is None: 
        state.report_content = ReportContent(
            sections=[{"header": "Placeholder", "text": "Report generation not yet fully implemented."}]
        )

    with me.box(style=me.Style(padding=me.Padding.all(20), background=WHATSAPP_DARK_COLORS["app_bg"], height="100vh", color=WHATSAPP_DARK_COLORS["primary_text"])):
        me.text(state.report_content.title, type="headline-4", style=me.Style(margin=me.Margin(bottom=20)))

        if state.report_content.is_generating:
            me.progress_spinner()
            me.text("Generating your detailed medical report...", style=me.Style(margin=me.Margin(top=10)))
        elif state.report_content.error:
            me.text(f"Error: {state.report_content.error}", style=me.Style(color=WHATSAPP_DARK_COLORS["error_text"]))
        elif state.report_content.sections:
            for section in state.report_content.sections:
                me.text(section.get("header", "Section"), type="headline-5", style=me.Style(margin=me.Margin(top=15, bottom=5)))
                me.markdown(section.get("text", "No content for this section."))
        else:
            me.text("No report data available. Please complete a chat session first.")

        with me.button("Back to Chat", on_click=lambda e: me.navigate("/chat")): # Navigate to /chat
            pass
"""

LLM_MOCK_PY_CONTENT = """
import asyncio
import logging
import utils.path_setup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
    logger.info("Ollama client imported successfully.")
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not found. Chatbot will use mock responses.")

async def get_ollama_response(prompt: str, history: list = None) -> str:
    if not OLLAMA_AVAILABLE:
        await asyncio.sleep(0.5) # Simulate network delay
        mock_response = f"Mock response to: '{prompt[:50]}...'. (Ollama unavailable)"
        logger.info(f"Using mock response: {mock_response}")
        return mock_response

    client = AsyncClient() 
    messages = []
    if history: # history is expected to be list of dicts like {'role': 'user', 'content': 'Hi'}
        for msg_dict in history:
            if isinstance(msg_dict, dict) and "role" in msg_dict and "content" in msg_dict:
                 messages.append({'role': msg_dict['role'], 'content': msg_dict['content']})
            else:
                logger.warning(f"Skipping malformed history item: {msg_dict}")
                 
    messages.append({'role': 'user', 'content': prompt})

    try:
        logger.info(f"Sending to Ollama (gemma:4b) - Prompt: '{prompt}' - History items: {len(messages)-1}")
        response = await client.chat(
            model='gemma:4b', 
            messages=messages,
            stream=False 
        )
        assistant_response = response['message']['content']
        logger.info(f"Received from Ollama: '{assistant_response[:100]}...'")
        return assistant_response
    except Exception as e:
        logger.error(f"Error communicating with Ollama: {e}", exc_info=True)
        return "I'm having trouble connecting to my AI brain right now. Please try again in a moment."

if __name__ == "__main__":
    async def main_test():
        test_prompt = "What are common symptoms of the flu?"
        print(f"Testing Ollama with prompt: {test_prompt}")
        response = await get_ollama_response(test_prompt)
        print(f"Ollama Test Response: {response}")
    asyncio.run(main_test())
"""

MAIN_PY_CONTENT = """
import mesop as me
import utils.path_setup # CRITICAL: This must be the first Fairdoc/msp import

# Import pages (after path setup)
from pages import chat_page  # Import the module
from pages import report_page # Import the module
from state.state_manager import initialize_app_state, AppState
from styles.whatsapp_dark_theme import app_container_style

# Global on_load for all pages, or define per page if needed
def on_load_main(e: me.LoadEvent):
    me.set_theme_mode("dark") 
    # Initialize state with a session_id. In a real app, this might come from URL or auth.
    initialize_app_state(session_id="user_session_123") 

@me.page(
    path="/", # Default page, will render chat_page content
    title="NHS AI Triage",
    on_load=on_load_main # Apply the on_load handler
)
def main_app_page():
    # This page can act as a router or directly render the default page.
    # For simplicity, we'll call the chat_page's rendering function directly.
    # Ensure chat_page.chat_page_content is the function that builds the UI.
    chat_page.chat_page_content()


if __name__ == "__main__":
    print("Starting Mesop app. Ensure you run with `mesop main.py` or `mesop run main.py`")
    # To run directly with `python main.py`, you'd typically use Mesop's CLI execution.
    # For development, `mesop main.py` is standard.
"""

# --- File Generation Logic ---
BASE_DIR = "msp"

# Ensure the base directory 'msp' exists or is created
base_path_obj = Path(BASE_DIR)
base_path_obj.mkdir(parents=True, exist_ok=True)

structure = {
    "main.py": MAIN_PY_CONTENT,
    "llm_mock.py": LLM_MOCK_PY_CONTENT,
    "styles/whatsapp_dark_theme.py": WHATSAPP_DARK_THEME_PY_CONTENT,
    "state/state_manager.py": STATE_MANAGER_PY_CONTENT,
    "components/chat_header.py": CHAT_HEADER_PY_CONTENT,
    "components/chat_bubble.py": CHAT_BUBBLE_PY_CONTENT,
    "components/chat_input.py": CHAT_INPUT_PY_CONTENT,
    "pages/chat_page.py": CHAT_PAGE_PY_CONTENT,
    "pages/report_page.py": REPORT_PAGE_PY_CONTENT,
    "utils/path_setup.py": PATH_SETUP_PY_CONTENT,
}

# Create __init__.py files
init_files = [
    "styles/__init__.py",
    "state/__init__.py",
    "components/__init__.py",
    "pages/__init__.py",
    "utils/__init__.py",
    "__init__.py" # For the 'msp' directory itself
]

for relative_path_str, content in structure.items():
    # Correctly join BASE_DIR with the relative_path_str
    file_path_obj = base_path_obj / Path(relative_path_str)
    
    # Ensure parent directory exists
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path_obj, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Generated: {file_path_obj}")

for init_file_rel_path in init_files:
    init_file_path_obj = base_path_obj / Path(init_file_rel_path)
    init_file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not init_file_path_obj.exists():
        with open(init_file_path_obj, "w", encoding="utf-8") as f:
            f.write("# This file makes Python treat the directory as a package.\n")
        print(f"Generated __init__.py for: {init_file_path_obj.parent}")

print(f"\nProject structure generated in '{BASE_DIR}' directory.")
print("To run the Mesop application, navigate to the directory containing 'msp' (e.g., your project root)")
print("Then run: cd msp")
print("And then: mesop main.py")

