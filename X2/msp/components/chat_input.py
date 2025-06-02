# msp/components/chat_input.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Callable
from styles.government_digital_styles import GOVERNMENT_COLORS

def render_chat_input(
    current_input_value: str,
    on_input_change: Callable,
    on_input_blur: Callable,
    on_send_click: Callable,
    is_bot_typing: bool
):
    """Render Signal-style chat input area with full-width textarea."""
    with me.box(style=me.Style(
        padding=me.Padding.symmetric(horizontal=16, vertical=12),
        background=GOVERNMENT_COLORS["bg_secondary"],
        border=me.Border(top=me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["light_grey"])),
        display="flex",
        align_items="flex-end",  # Align items to bottom for multi-line input that grows
        gap=12
    )):
        # Attachment button (placeholder, can be added later)
        # with me.content_button(type="icon", on_click=lambda e: print("Attach clicked")):
        #     me.icon("attach_file")

        # Text input area container - this box needs to grow
        with me.box(style=me.Style(
            flex_grow=1,  # Key: Allow this box to take up available horizontal space
            background="white",
            border_radius="20px",  # Rounded like Signal/WhatsApp
            border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
            padding=me.Padding.symmetric(horizontal=12, vertical=6),  # Adjusted padding for better fit
            display="flex",  # Use flex to align textarea and potential emoji button
            align_items="flex-end"  # Align items to bottom within this rounded box
        )):
            me.native_textarea(
                value=current_input_value,
                on_blur=on_input_blur,
                on_input=on_input_change,
                placeholder="Type a message...",
                autosize=True,
                min_rows=1,
                max_rows=5,  # Limit max rows like Signal
                style=me.Style(
                    width="100%",  # Textarea will take full width of its parent flex item
                    background="transparent",
                    border=None,  # Remove default textarea border
                    outline="none",  # Remove default textarea outline
                    font_size="1rem",
                    line_height="1.4",
                    color=GOVERNMENT_COLORS["text_primary"],
                    padding=me.Padding.all(0)  # Remove default padding of native_textarea if any conflicts
                )
            )
            # Placeholder for Emoji button (if you add it later)
            # with me.content_button(type="icon", style=me.Style(margin=me.Margin(left=8))):
            #     me.icon("sentiment_satisfied")

        # Send button (or mic button logic if input is empty)
        send_enabled = len(current_input_value.strip()) > 0 and not is_bot_typing
        with me.content_button(
            type="icon",
            on_click=on_send_click,
            disabled=not send_enabled,
            style=me.Style(
                background=GOVERNMENT_COLORS["primary"] if send_enabled else GOVERNMENT_COLORS["medium_grey"],
                color="white",
                width=40,  # Fixed size for icon button
                height=40,  # Fixed size for icon button
                border_radius="50%",  # Circular button
                flex_shrink=0  # Prevent send button from shrinking
            )
        ):
            me.icon("send")
