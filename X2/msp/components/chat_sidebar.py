# msp/components/chat_sidebar.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import List, Dict, Any, Optional
from styles.government_digital_styles import GOVERNMENT_COLORS
from state.state_manager import AppState

@me.stateclass
class ChatSidebarState:
    is_expanded: bool = True
    search_term: str = ""
    active_chat_session_id: Optional[str] = None

def render_chat_sidebar():
    """Render Signal-style chat sidebar with chat history and controls"""
    state = me.state(AppState)
    sidebar_state = me.state(ChatSidebarState)
    
    with me.box(style=me.Style(
        width=320 if sidebar_state.is_expanded else 80,
        background=GOVERNMENT_COLORS["bg_secondary"],
        border=me.Border(right=me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        display="flex",
        flex_direction="column",
        height="100vh",
        transition="width 0.3s ease"
    )):
        render_sidebar_header(sidebar_state)
        if sidebar_state.is_expanded:
            render_sidebar_search(sidebar_state)
        render_chat_list(state, sidebar_state)
        if sidebar_state.is_expanded:
            render_sidebar_footer()

def render_sidebar_header(sidebar_state: ChatSidebarState):
    """Render sidebar header with toggle button and title"""
    with me.box(style=me.Style(
        padding=me.Padding.all(16),
        display="flex",
        align_items="center",
        gap=12,
        border=me.Border(bottom=me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["light_grey"]))
    )):
        # Menu button (always visible)
        with me.content_button(
            type="icon",
            on_click=toggle_sidebar,
            style=me.Style(color=GOVERNMENT_COLORS["primary"]),
            key="sidebar_toggle_button"
        ):
            me.icon(icon="menu" if sidebar_state.is_expanded else "menu_open")
        
        # FIXED: Title and New Chat button only show when expanded
        if sidebar_state.is_expanded:
            me.text("Chats", type="headline-6", style=me.Style(
                color=GOVERNMENT_COLORS["text_primary"],
                flex_grow=1  # Allow title to take available space
            ))
            
            # FIXED: New chat button moved INSIDE the expanded block
            with me.tooltip(message="Start New Chat"):
                with me.content_button(
                    type="icon",
                    on_click=start_new_chat_session,
                    style=me.Style(color=GOVERNMENT_COLORS["primary"]),
                    key="new_chat_button_expanded"  # Different key to avoid conflicts
                ):
                    me.icon(icon="add_comment")

def render_sidebar_search(sidebar_state: ChatSidebarState):
    """Render chat search bar"""
    with me.box(style=me.Style(
        padding=me.Padding.symmetric(horizontal=16, vertical=12),
        display="flex",
        align_items="center",
        gap=8,
        background="white",
        border_radius="8px"
    )):
        me.icon("search", style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))
        me.input(
            label="Search chats...",
            value=sidebar_state.search_term,
            on_input=update_search_term,
            style=me.Style(
                width="100%",
                border=None,
                outline="none",
                background="transparent"
            )
        )

def render_chat_list(app_state: AppState, sidebar_state: ChatSidebarState):
    """Render list of chat sessions"""
    chat_sessions = get_recent_chat_sessions()
    with me.box(style=me.Style(flex_grow=1, overflow_y="auto")):
        if chat_sessions:
            for session in chat_sessions:
                render_chat_list_item(session, sidebar_state)
        elif sidebar_state.is_expanded:
            render_no_chats_message()

def render_chat_list_item(session_data: Dict[str, Any], sidebar_state: ChatSidebarState):
    """Render individual chat list item"""
    session_id = session_data.get("session_id", "unknown")
    is_active = session_id == sidebar_state.active_chat_session_id
    
    # Use only active state for background (no hover since Mesop doesn't support it)
    background_color = GOVERNMENT_COLORS["primary"] if is_active else "transparent"
    
    with me.box(
        on_click=lambda e, sid=session_id: select_chat_session(e, sid),
        style=me.Style(
            padding=me.Padding.symmetric(horizontal=16, vertical=12),
            display="flex",
            align_items="center",
            gap=12,
            cursor="pointer",
            background=background_color,
            border=me.Border(bottom=me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["light_grey"]))
        )
    ):
        if sidebar_state.is_expanded:
            render_chat_avatar(session_data)
            render_chat_item_details(session_data, is_active)
        else:
            render_collapsed_chat_icon(session_data, is_active)

def render_chat_avatar(session_data: Dict[str, Any]):
    """Render chat avatar"""
    with me.box(style=me.Style(
        width=40,
        height=40,
        background=GOVERNMENT_COLORS["medium_grey"],
        border_radius="50%",
        display="flex",
        align_items="center",
        justify_content="center"
    )):
        me.text("P", style=me.Style(color="white", font_size="1.2rem", font_weight="600"))

def render_chat_item_details(session_data: Dict[str, Any], is_active: bool):
    """Render chat item details (name, last message, timestamp)"""
    # Only use active state for text color
    text_color = "white" if is_active else GOVERNMENT_COLORS["text_primary"]
    muted_color = "rgba(255,255,255,0.8)" if is_active else GOVERNMENT_COLORS["text_muted"]
    
    with me.box(style=me.Style(flex_grow=1)):
        me.text(
            session_data.get("patient_name", "Unknown Patient"),
            style=me.Style(font_weight="600", color=text_color, margin=me.Margin(bottom=4))
        )
        
        last_message = session_data.get("last_message", "No messages yet")
        me.text(
            last_message[:30] + "..." if len(last_message) > 30 else last_message,
            style=me.Style(font_size="0.9rem", color=muted_color)
        )
    
    with me.box(style=me.Style(text_align="right")):
        timestamp = session_data.get("timestamp", "")
        me.text(timestamp, style=me.Style(font_size="0.8rem", color=muted_color, margin=me.Margin(bottom=4)))
        
        unread_count = session_data.get("unread_count", 0)
        if unread_count > 0:
            render_unread_badge(unread_count)

def render_collapsed_chat_icon(session_data: Dict[str, Any], is_active: bool):
    """Render icon for collapsed chat item"""
    icon = "chat_bubble"
    if session_data.get("unread_count", 0) > 0:
        icon = "mark_chat_unread"
    
    me.icon(icon, style=me.Style(
        font_size="24px",
        color=GOVERNMENT_COLORS["primary"] if is_active else GOVERNMENT_COLORS["text_secondary"]
    ))

def render_unread_badge(count: int):
    """Render unread message count badge"""
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["error"],
        color="white",
        font_size="0.7rem",
        font_weight="600",
        padding=me.Padding.symmetric(horizontal=6, vertical=2),
        border_radius="10px",
        min_width="20px",
        text_align="center"
    )):
        me.text(str(count))

def render_no_chats_message():
    """Render message when no chats are available"""
    with me.box(style=me.Style(
        padding=me.Padding.all(20),
        text_align="center",
        color=GOVERNMENT_COLORS["text_muted"]
    )):
        me.icon("chat_bubble_outline", style=me.Style(font_size="48px", margin=me.Margin(bottom=16)))
        me.text("No active chats", style=me.Style(font_weight="600", margin=me.Margin(bottom=8)))
        me.text("Start a new conversation to begin.")

def render_sidebar_footer():
    """Render sidebar footer with settings and profile"""
    with me.box(style=me.Style(
        padding=me.Padding.all(16),
        border=me.Border(top=me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["light_grey"])),
        display="flex",
        justify_content="space-between",
        align_items="center"
    )):
        with me.tooltip(message="Settings"):
            with me.content_button(
                type="icon",
                on_click=open_settings,
                style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]),
                key="settings_button"
            ):
                me.icon(icon="settings")
        
        with me.box(style=me.Style(
            width=32,
            height=32,
            background=GOVERNMENT_COLORS["primary"],
            border_radius="50%",
            display="flex",
            align_items="center",
            justify_content="center",
            cursor="pointer"
        )):
            me.text("U", style=me.Style(color="white", font_weight="600"))

# Event Handlers
def toggle_sidebar(e: me.ClickEvent):
    """Toggle sidebar expansion"""
    state = me.state(ChatSidebarState)
    state.is_expanded = not state.is_expanded

def update_search_term(e: me.InputEvent):
    """Update search term in sidebar state"""
    state = me.state(ChatSidebarState)
    state.search_term = e.value

def start_new_chat_session(e: me.ClickEvent):
    """Handle starting a new chat session"""
    # TODO: Implement new chat session logic
    pass

def select_chat_session(e: me.ClickEvent, session_id: str):
    """Handle selecting an existing chat session"""
    sidebar_state = me.state(ChatSidebarState)
    sidebar_state.active_chat_session_id = session_id
    # TODO: Load selected chat session

def open_settings(e: me.ClickEvent):
    """Handle opening settings panel"""
    # TODO: Implement settings panel
    pass

# Helper to fetch chat sessions (placeholder)
def get_recent_chat_sessions() -> List[Dict[str, Any]]:
    """Placeholder function to get recent chat sessions"""
    return [
        {"session_id": "1", "patient_name": "John Doe", "last_message": "Okay, thank you!", "timestamp": "10:30 AM", "unread_count": 0},
        {"session_id": "2", "patient_name": "Jane Smith", "last_message": "I'm feeling much better now.", "timestamp": "Yesterday", "unread_count": 2},
        {"session_id": "3", "patient_name": "Anonymous User", "last_message": "Hello?", "timestamp": "Mon", "unread_count": 0}
    ]
