import mesop as me
from typing import Callable, Any, Generator  # Import Callable
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()
from styles.whatsapp_dark_theme import (
    WHATSAPP_DARK_COLORS,
    chat_input_bar_style,
    input_field_wrapper_style,
    native_textarea_style,
    icon_button_style
)
# from state.state_manager import AppState # AppState is not directly used here if values are passed

@me.component
def render_chat_input(
    current_input_value: str,
    # Use typing.Callable with the correct event types that the HANDLER functions expect
    on_input_change: Callable[[me.InputEvent], Any | Generator[None, None, None]],
    on_input_blur: Callable[[me.InputBlurEvent], Any | Generator[None, None, None]],
    on_send_click: Callable[[me.ClickEvent], Any | Generator[None, None, None]],  # Assuming it's a click event from content_button
    is_bot_typing: bool
):
    with me.box(style=chat_input_bar_style()):
        with me.content_button(type="icon", style=icon_button_style()):
            me.icon("mood", style=me.Style(color=WHATSAPP_DARK_COLORS["icon_color"]))

        with me.content_button(type="icon", style=icon_button_style()):  # Placeholder for file upload
            me.icon("attach_file", style=me.Style(color=WHATSAPP_DARK_COLORS["icon_color"]))
 
        with me.box(style=input_field_wrapper_style()):  # This container defines the visual bounds
            me.native_textarea(
                key="chat_main_input",
                value=current_input_value,
                placeholder="Type a message",
                autosize=False,
                min_rows=2,
                max_rows=12,
                style=native_textarea_style(),
                on_input=on_input_change,  # This expects a function like: def handler(e: me.InputEvent): ...
                on_blur=on_input_blur,   # This expects a function like: def handler(e: me.InputBlurEvent): ...
                # If using Enter to send via shortcuts, on_send_click would need to handle TextareaShortcutEvent
                # shortcuts={me.Shortcut(key="Enter"): on_send_click}
            )

        can_send = current_input_value.strip() != "" and not is_bot_typing
        send_icon_color = WHATSAPP_DARK_COLORS["green_accent"] if can_send else WHATSAPP_DARK_COLORS["icon_color"]

        with me.content_button(
            type="icon",
            on_click=on_send_click,  # This expects a function like: def handler(e: me.ClickEvent): ...
            disabled=not can_send,
            style=icon_button_style()
        ):
            me.icon("send", style=me.Style(color=send_icon_color))
