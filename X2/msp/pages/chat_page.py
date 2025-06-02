# msp/pages/chat_page.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, Any
from datetime import datetime

# Import state and utilities
from state.state_manager import AppState, add_chat_message, initialize_app_state
from utils.extensions.webcomponents import get_webcomponent_security_policy, smooth_scroll_web_component, whatsapp_smooth_scroll, handle_scroll_start, handle_scroll_complete
from llm_mock import get_ollama_response

# Import components
from components.chat_sidebar import render_chat_sidebar, ChatSidebarState
from components.chat_header import render_chat_header
from components.chat_bubble import render_chat_bubble
from components.chat_input import render_chat_input

# Import styles
from styles.government_digital_styles import GOVERNMENT_COLORS, base_page_style

def on_chat_page_load(e: me.LoadEvent):
    """Initialize chat page state following Mesop best practices"""
    me.set_theme_mode("system")
    
    # Initialize main app state
    state = me.state(AppState)
    if not state.session_id:
        state.session_id = "chat_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    initialize_app_state(state.session_id)
    
    # Initialize sidebar state - Mesop automatically handles initialization
    sidebar_state = me.state(ChatSidebarState)
    # Set default expanded state if not already set
    if not hasattr(sidebar_state, '_initialized'):
        sidebar_state.is_expanded = True
        sidebar_state._initialized = True

@me.page(
    path="/chat",
    title="Fairdoc AI - Secure Chat",
    on_load=on_chat_page_load,
    security_policy=get_webcomponent_security_policy()
)
def chat_page():
    """Main chat page with Signal-style UI and Fairdoc backend integration"""
    with me.box(style=me.Style(
        display="flex",
        height="100vh",
        overflow="hidden"
    )):
        # Chat Sidebar (Signal-style)
        render_chat_sidebar()
        
        # Main Chat Area (Signal-style)
        render_main_chat_content()

def render_main_chat_content():
    """Render the main chat content area following Mesop patterns"""
    state = me.state(AppState)
    
    with me.box(style=me.Style(
        display="flex",
        flex_direction="column",
        flex_grow=1,
        height="100vh",
        background=GOVERNMENT_COLORS["bg_primary"]
    )):
        # Smooth scroll component
        smooth_scroll_web_component(
            key="chat_smooth_scroller",
            on_scroll_start=handle_scroll_start,
            on_scroll_complete=handle_scroll_complete
        )
        
        # Chat Header
        render_chat_header(contact_name="NHS AI Assistant", status="Online")
        
        # Chat Message Area
        render_chat_message_area()
        
        # Chat Input Area
        render_chat_input(
            current_input_value=state.current_input,
            on_input_change=handle_input_change,
            on_input_blur=handle_input_blur,
            on_send_click=handle_send_message_click,
            is_bot_typing=state.is_bot_typing
        )

def render_chat_message_area():
    """Render chat messages area with proper state handling"""
    state = me.state(AppState)
    
    with me.box(style=me.Style(
        flex_grow=1,
        overflow_y="auto",
        padding=me.Padding.all(20),
        background=GOVERNMENT_COLORS["bg_primary"],
        display="flex",
        flex_direction="column"
    )):
        # Render messages if they exist
        if state.chat_history:
            for message_data in state.chat_history:
                render_chat_bubble(message=message_data)
        else:
            render_empty_chat_placeholder()
        
        # Show typing indicator when bot is responding
        if state.is_bot_typing:
            render_typing_indicator()
        
        # Scroll anchor for smooth scrolling
        with me.box(key="chat_end_anchor", style=me.Style(height=1)):
            pass

def render_empty_chat_placeholder():
    """Render placeholder when no messages exist"""
    with me.box(style=me.Style(
        flex_grow=1,
        display="flex",
        flex_direction="column",
        align_items="center",
        justify_content="center",
        color=GOVERNMENT_COLORS["text_muted"]
    )):
        me.icon("chat", style=me.Style(font_size="64px", margin=me.Margin(bottom=20)))
        me.text("No messages yet", style=me.Style(font_size="1.2rem", font_weight="500"))
        me.text("Start the conversation by typing a message below.")

def render_typing_indicator():
    """Render typing indicator following Mesop patterns"""
    with me.box(style=me.Style(
        align_self="flex-start",
        margin=me.Margin(left=10, bottom=5),
        display="flex",
        align_items="center",
        gap=8
    )):
        # Simple typing indicator - could be enhanced with animation
        me.progress_spinner(size="small")
        me.text("Assistant is typing...", style=me.Style(
            font_style="italic",
            font_size="0.9rem",
            color=GOVERNMENT_COLORS["text_secondary"]
        ))

# Event Handlers following Mesop best practices

def handle_input_change(event: me.InputEvent):
    """Handle input change events - regular function pattern"""
    state = me.state(AppState)
    state.current_input = event.value

def handle_input_blur(event: me.InputBlurEvent):
    """Handle input blur events - generator function pattern for state persistence"""
    state = me.state(AppState)
    current_val = event.value
    
    if state.current_input != current_val:
        state.current_input = current_val
        # Save state to database
        from state.state_manager import save_comprehensive_session_data_to_db
        save_comprehensive_session_data_to_db()
        yield  # Yield to update UI after state change

def handle_send_message_click(event: me.ClickEvent):
    """Handle send message - generator function pattern for streaming responses"""
    state = me.state(AppState)
    user_message_content = state.current_input.strip()

    # Early return if no content
    if not user_message_content:
        return

    # Step 1: Add user message and update UI
    add_chat_message(role="user", content=user_message_content, session_id=state.session_id)
    state.current_input = ""
    state.is_bot_typing = True
    
    # Save state and yield to update UI
    from state.state_manager import save_comprehensive_session_data_to_db
    save_comprehensive_session_data_to_db()
    yield  # Show user message, clear input, show typing indicator

    # Step 2: Smooth scroll to bottom
    yield from whatsapp_smooth_scroll("chat_end_anchor")

    # Step 3: Prepare chat history for LLM (exclude last message to prevent echo)
    history_for_llm = [
        {"role": msg.role, "content": msg.content} 
        for msg in state.chat_history[:-1]  # Exclude the just-added user message
    ]

    # Step 4: Get bot response using Mesop's effects pattern
    try:
        bot_response_content = yield from me.effects(
            lambda: get_ollama_response(user_message_content, history_for_llm), 
            ()  # No dependencies
        )
    except Exception as e:
        # Handle errors gracefully
        bot_response_content = f"I apologize, but I encountered an error: {str(e)}. Please try again."

    # Step 5: Add bot response and update UI
    add_chat_message(role="assistant", content=bot_response_content, session_id=state.session_id)
    state.is_bot_typing = False
    save_comprehensive_session_data_to_db()
    yield  # Show bot message, hide typing indicator

    # Step 6: Final smooth scroll to bottom
    yield from whatsapp_smooth_scroll("chat_end_anchor")

# Sidebar management functions following Mesop component patterns

def toggle_sidebar_expanded():
    """Toggle sidebar expansion state"""
    sidebar_state = me.state(ChatSidebarState)
    sidebar_state.is_expanded = not sidebar_state.is_expanded

def update_sidebar_search(search_term: str):
    """Update sidebar search term"""
    sidebar_state = me.state(ChatSidebarState)
    sidebar_state.search_term = search_term

def select_chat_from_sidebar(session_id: str):
    """Select a chat session from sidebar"""
    # Update sidebar state
    sidebar_state = me.state(ChatSidebarState)
    sidebar_state.active_chat_session_id = session_id
    
    # Update main app state
    app_state = me.state(AppState)
    if app_state.session_id != session_id:
        # Load the selected chat session
        app_state.session_id = session_id
        initialize_app_state(session_id)

# Utility functions for chat management

def start_new_chat_session():
    """Start a new chat session following Mesop state patterns"""
    app_state = me.state(AppState)
    sidebar_state = me.state(ChatSidebarState)
    
    # Generate new session ID
    new_session_id = "chat_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update states
    app_state.session_id = new_session_id
    sidebar_state.active_chat_session_id = new_session_id
    
    # Initialize new session
    initialize_app_state(new_session_id)

def get_chat_sessions_for_sidebar():
    """Get chat sessions for sidebar display - placeholder function"""
    # This would typically fetch from database
    # Following the pattern from the fancy chat example
    return [
        {
            "session_id": "1", 
            "patient_name": "John Doe", 
            "last_message": "Thank you for your help!", 
            "timestamp": "10:30 AM", 
            "unread_count": 0
        },
        {
            "session_id": "2", 
            "patient_name": "Jane Smith", 
            "last_message": "I'm feeling better now.", 
            "timestamp": "Yesterday", 
            "unread_count": 2
        }
    ]
