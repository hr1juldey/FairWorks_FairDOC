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