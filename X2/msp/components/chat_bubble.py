import mesop as me
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()
from styles.whatsapp_dark_theme import (
    sent_bubble_style, 
    received_bubble_style, 
    message_text_style, 
    timestamp_style,
    WHATSAPP_DARK_COLORS # For tick icon color
)
from state.state_manager import ChatMessage # Ensure this import works

@me.component
def render_chat_bubble(message: ChatMessage): # Use ChatMessage type hint
    is_user = message.role == "user"
    bubble_style_func = sent_bubble_style if is_user else received_bubble_style

    with me.box(style=bubble_style_func()):
        # Main message content
        me.text(message.content, style=message_text_style())
        
        # Timestamp and status ticks (like WhatsApp)
        with me.box(style=me.Style(display="flex", align_items="center", justify_content="flex-end", margin=me.Margin(top=5))):
            me.text(message.timestamp, style=timestamp_style())
            if is_user: # Show ticks for user's sent messages
                me.icon("done_all", style=me.Style(font_size="16px", color=WHATSAPP_DARK_COLORS["icon_active_color"], margin=me.Margin(left=4)))
