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