# msp/components/chat_message_area.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import List, Dict, Any
from styles.government_digital_styles import GOVERNMENT_COLORS
from state.state_manager import AppState, ChatMessage
from components.chat_bubble import render_chat_bubble  # Existing component

def render_chat_message_area():
    """Render the main chat message display area (Signal-style)"""
    
    state = me.state(AppState)
    
    with me.box(style=me.Style(
        flex_grow=1,
        overflow_y="auto",
        padding=me.Padding.all(20),
        background=GOVERNMENT_COLORS["bg_primary"],  # Use primary background
        display="flex",
        flex_direction="column"
    )):
        if state.chat_history and len(state.chat_history) > 0:
            render_chat_messages(state.chat_history)
        else:
            render_empty_chat_placeholder()
        
        # Typing indicator
        if state.is_bot_typing:
            render_typing_indicator()
        
        # Scroll anchor for new messages
        with me.box(key="chat_end_anchor", style=me.Style(height=1)):
            pass

def render_chat_messages(chat_history: List[ChatMessage]):
    """Render all chat messages"""
    
    for i, message in enumerate(chat_history):
        # Check if this is the start of a new day
        if i == 0 or is_new_day(chat_history[i - 1].timestamp, message.timestamp):
            render_date_separator(message.timestamp)
        
        # Render message bubble
        render_chat_bubble(message=message)  # Using existing component

def render_date_separator(timestamp_str: str):
    """Render date separator like in Signal"""
    
    try:
        # Parse timestamp (assuming "%I:%M %p" or similar)
        # For simplicity, just show a static date for now
        date_display = "Today"  # Replace with actual date logic
    except Exception:
        date_display = "Date Unknown"
        
    with me.box(style=me.Style(
        text_align="center",
        margin=me.Margin.symmetric(vertical=16)
    )):
        with me.box(style=me.Style(
            display="inline-block",
            background=GOVERNMENT_COLORS["light_grey"],
            color=GOVERNMENT_COLORS["text_muted"],
            padding=me.Padding.symmetric(horizontal=12, vertical=4),
            border_radius="12px",
            font_size="0.8rem"
        )):
            me.text(date_display)

def is_new_day(prev_timestamp: str, current_timestamp: str) -> bool:
    """Check if current message is on a new day"""
    # Placeholder logic - implement actual date comparison
    return False  # For now, assume all messages are on the same day

def render_empty_chat_placeholder():
    """Render placeholder when chat is empty"""
    
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
    """Render typing indicator similar to Signal"""
    
    with me.box(style=me.Style(
        padding=me.Padding.symmetric(horizontal=12, vertical=8),
        display="flex",
        align_items="center",
        gap=8,
        align_self="flex-start"  # For bot typing indicator
    )):
        # Avatar (placeholder)
        with me.box(style=me.Style(
            width=24,
            height=24,
            background=GOVERNMENT_COLORS["primary"],
            border_radius="50%"
        )):
            pass
        
        # Typing dots animation
        with me.box(style=me.Style(
            background=GOVERNMENT_COLORS["light_grey"],
            border_radius="16px",
            padding=me.Padding.symmetric(horizontal=12, vertical=8)
        )):
            # Use a simple text for now - animations would need web components
            me.text("Typing...", style=me.Style(
                font_style="italic",
                font_size="0.9rem",
                color=GOVERNMENT_COLORS["text_secondary"]
            ))
