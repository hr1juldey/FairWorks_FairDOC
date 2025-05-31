import mesop as me
import asyncio  # For me.effects
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()

from components.chat_header import render_chat_header
from components.chat_bubble import render_chat_bubble
from components.chat_input import render_chat_input
from state.state_manager import AppState, add_chat_message, initialize_app_state, save_session_data_to_db
# Import app_container_style as well
from styles.whatsapp_dark_theme import chat_area_style, WHATSAPP_DARK_COLORS, app_container_style
from llm_mock import get_ollama_response

# This on_load is specific to this page if navigated to directly
def on_chat_page_load(e: me.LoadEvent):
    state = me.state(AppState)
    me.set_theme_mode("dark")  # Ensure theme is set
    initialize_app_state(state.session_id if state.session_id else "direct_chat_session")  # Use existing or new session_id

@me.page(
    path="/chat",
    title="WhatsApp Chat",
    on_load=on_chat_page_load  # Add on_load here
)
def chat_page_content():
    state = me.state(AppState)
    # initialize_app_state(state.session_id) # Moved to on_load for direct navigation

    # WRAP THE ENTIRE PAGE CONTENT
    with me.box(style=app_container_style()):  # APPLY THE FULL PAGE CONTAINER STYLE
        render_chat_header(contact_name="NHS AI Assistant", status="Online")

        with me.box(style=chat_area_style(), key="chat_scroll_area"):
            if state.chat_history:
                for message_data in state.chat_history:
                    render_chat_bubble(message=message_data)
            
            if state.is_bot_typing:
                with me.box(style=me.Style(align_self="flex-start", margin=me.Margin(left=10, bottom=5))):
                    me.text("Assistant is typing...", style=me.Style(font_style="italic", font_size="12px", color=WHATSAPP_DARK_COLORS["secondary_text"]))
            
            with me.box(key="chat_end_anchor", style=me.Style(height=1)):
                pass

        render_chat_input(
            current_input_value=state.current_input,
            on_input_change=handle_input_change,
            on_input_blur=handle_input_blur,
            on_send_click=handle_send_message_click,
            is_bot_typing=state.is_bot_typing
        )
        
        # me.scroll_into_view(key="#chat_end_anchor")
        # If you want to try passing options for smooth scroll:
        me.scroll_into_view(key="chat_end_anchor", scroll_into_view_options={"block": "end", "behavior": "smooth"})
        # However, the primary fix is using `key` instead of `selector`.

# Event handlers remain the same
def handle_input_change(event: me.InputEvent):
    state = me.state(AppState)
    state.current_input = event.value

def handle_input_blur(event: me.InputBlurEvent):
    state = me.state(AppState)
    state.current_input = event.value
    save_session_data_to_db(state.session_id, state.current_input, state.is_bot_typing)
    yield

def handle_send_message_click(event: me.ClickEvent):
    state = me.state(AppState)
    user_message_content = state.current_input.strip()

    if not user_message_content:
        return

    add_chat_message(role="user", content=user_message_content, session_id=state.session_id)
    state.current_input = ""
    state.is_bot_typing = True
    save_session_data_to_db(state.session_id, state.current_input, state.is_bot_typing)
    yield

    llm_history = [{"role": msg.role, "content": msg.content} for msg in state.chat_history[:-1]]

    bot_response_content = yield from me.effects(
        lambda: get_ollama_response(user_message_content, llm_history), ()
    )

    add_chat_message(role="assistant", content=bot_response_content, session_id=state.session_id)
    state.is_bot_typing = False
    save_session_data_to_db(state.session_id, state.current_input, state.is_bot_typing)
    yield
