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