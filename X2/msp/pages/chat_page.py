import mesop as me
import asyncio
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()

from components.chat_header import render_chat_header
from components.chat_bubble import render_chat_bubble
from components.chat_input import render_chat_input
from state.state_manager import AppState, add_chat_message, initialize_app_state
from styles.whatsapp_dark_theme import chat_area_style
from llm_mock import get_ollama_response # For Ollama integration

@me.page(path="/chat", title="WhatsApp Chat")
def chat_page():
    initialize_app_state() # Ensure state is ready
    state = me.state(AppState)

    render_chat_header(contact_name="NHS AI Assistant", status="Online")

    with me.box(style=chat_area_style()):
        if state.chat_history:
            for message_data in state.chat_history:
                render_chat_bubble(message=message_data)
        
        if state.is_bot_typing:
            with me.box(style=me.Style(align_self="flex-start", margin=me.Margin(left=10))): # Typing indicator
                me.text("Assistant is typing...", style=me.Style(font_style="italic", font_size="12px"))


    render_chat_input(
        current_input_value=state.current_input,
        on_input_change=handle_input_change,
        on_input_blur=handle_input_blur,
        on_send_click=handle_send_message_click,
        is_bot_typing=state.is_bot_typing
    )

def handle_input_change(event: me.InputEvent):
    state = me.state(AppState)
    state.current_input = event.value
    # No yield here for better typing performance

def handle_input_blur(event: me.InputBlurEvent): # Corrected type
    state = me.state(AppState)
    state.current_input = event.value # Finalize value on blur
    # Optionally yield if send button state depends on this
    yield

def handle_send_message_click(event: me.ClickEvent):
    state = me.state(AppState)
    user_message_content = state.current_input.strip()

    if not user_message_content:
        return

    add_chat_message(role="user", content=user_message_content)
    state.current_input = "" # Clear input field in state
    state.is_bot_typing = True
    yield # Update UI to show user message and clear input, show typing

    # Prepare history for LLM
    llm_history = [{"role": msg.role, "content": msg.content} for msg in state.chat_history[:-1]] # Exclude current user msg

    # Call Ollama
    bot_response_content = yield from me.effects(
        lambda: get_ollama_response(user_message_content, llm_history), ()
    )

    add_chat_message(role="assistant", content=bot_response_content)
    state.is_bot_typing = False
    yield # Update UI to show bot message and hide typing
