# File: Fairdoc\frontend\msp\main.py

"""
Fairdoc AI - NHS Digital Triage Frontend
WhatsApp-style medical triage interface using Mesop
"""

import mesop as me
from components.chat_interface import render_chat_interface
from utils.state_manager import get_state, add_message
from styles.material_theme import MD3_COLORS

@me.page(
    path="/",
    title="NHS Digital Triage - Fairdoc AI",
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://fairdoc.ai"]
    )
)
def home_page():
    """Main application page"""
    state = get_state()
    
    # Initialize welcome message if first visit
    if not state.messages:
        add_message(
            "Welcome to NHS Digital Triage! I'm here to help assess your symptoms. Please describe your current symptoms to get started.",
            is_user=False
        )
    
    # Render main interface
    with me.box(style=me.Style(
        height="100vh",
        font_family="Roboto, sans-serif",
        background=MD3_COLORS["background"]
    )):
        render_chat_interface()

@me.page(
    path="/emergency",
    title="Emergency Response - NHS Digital Triage"
)
def emergency_page():
    """Emergency response page"""
    with me.box(style=me.Style(
        height="100vh",
        background=MD3_COLORS["error"],
        color=MD3_COLORS["on_error"],
        display="flex",
        flex_direction="column",
        justify_content="center",
        align_items="center",
        text_align="center",
        padding=me.Padding.all(32)
    )):
        me.text("🚨", style=me.Style(font_size="64px"))
        me.text("EMERGENCY ALERT", style=me.Style(
            font_size="32px",
            font_weight="bold",
            margin=me.Margin(bottom=16)
        ))
        me.text("Emergency services have been contacted", style=me.Style(
            font_size="18px",
            margin=me.Margin(bottom=32)
        ))
        
        me.button(
            "Return to Chat",
            on_click=lambda e: me.navigate("/"),
            style=me.Style(
                background=MD3_COLORS["on_error"],
                color=MD3_COLORS["error"],
                padding=me.Padding.symmetric(horizontal=24, vertical=12),
                border="none",
                border_radius=24,
                font_size="16px",
                font_weight="500"
            )
        )

if __name__ == "__main__":
    me.run()

