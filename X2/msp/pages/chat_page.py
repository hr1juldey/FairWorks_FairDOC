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