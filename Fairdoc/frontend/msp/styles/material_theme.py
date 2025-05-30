# File: FairDOC\frontend\msp\styles\material_theme.py

"""
Material Design 3 Theme for Fairdoc AI
WhatsApp-inspired medical triage interface
"""

import mesop as me

# Material Design 3 Color Tokens
MD3_COLORS = {
    # Primary colors
    "primary": "#006A6B",
    "on_primary": "#FFFFFF",
    "primary_container": "#6FF7F8",
    "on_primary_container": "#002020",
    
    # Secondary colors
    "secondary": "#4A6363",
    "on_secondary": "#FFFFFF",
    "secondary_container": "#CCE8E8",
    "on_secondary_container": "#051F1F",
    
    # Tertiary colors
    "tertiary": "#456179",
    "on_tertiary": "#FFFFFF",
    "tertiary_container": "#CCE5FF",
    "on_tertiary_container": "#001E30",
    
    # Error colors
    "error": "#BA1A1A",
    "on_error": "#FFFFFF",
    "error_container": "#FFDAD6",
    "on_error_container": "#410002",
    
    # Neutral colors
    "outline": "#6F7979",
    "background": "#F4FFFE",
    "on_background": "#161D1D",
    "surface": "#F4FFFE",
    "on_surface": "#161D1D",
    "surface_variant": "#DAE5E5",
    "on_surface_variant": "#3F4948",
    
    # WhatsApp-inspired
    "chat_bg": "#E5DDD5",
    "message_sent": "#DCF8C6",
    "message_received": "#FFFFFF",
    "header_bg": "#075E54",
    "input_bg": "#F0F0F0"
}

# Typography Scale
MD3_TYPOGRAPHY = {
    "display_large": me.Style(
        font_size="57px",
        font_weight="400",
        line_height="64px",
        letter_spacing="-0.25px"
    ),
    "headline_large": me.Style(
        font_size="32px",
        font_weight="400",
        line_height="40px"
    ),
    "headline_medium": me.Style(
        font_size="28px",
        font_weight="400",
        line_height="36px"
    ),
    "title_large": me.Style(
        font_size="22px",
        font_weight="400",
        line_height="28px"
    ),
    "body_large": me.Style(
        font_size="16px",
        font_weight="400",
        line_height="24px",
        letter_spacing="0.5px"
    ),
    "body_medium": me.Style(
        font_size="14px",
        font_weight="400",
        line_height="20px",
        letter_spacing="0.25px"
    ),
    "label_large": me.Style(
        font_size="14px",
        font_weight="500",
        line_height="20px",
        letter_spacing="0.1px"
    )
}

# Component Styles
def chat_container_style():
    return me.Style(
        background=MD3_COLORS["chat_bg"],
        height="100vh",
        display="flex",
        flex_direction="column",
        font_family="Roboto, sans-serif"
    )

def header_style():
    return me.Style(
        background=MD3_COLORS["header_bg"],
        color=MD3_COLORS["on_primary"],
        padding=me.Padding.all(16),
        display="flex",
        align_items="center",
        box_shadow="0 2px 4px rgba(0,0,0,0.1)"
    )

def message_sent_style():
    return me.Style(
        background=MD3_COLORS["message_sent"],
        color=MD3_COLORS["on_surface"],
        padding=me.Padding.all(12),
        margin=me.Margin(left=40, right=8, top=4, bottom=4),
        border_radius=18,
        border_top_right_radius=4,
        max_width="80%",
        align_self="flex-end",
        box_shadow="0 1px 2px rgba(0,0,0,0.1)"
    )

def message_received_style():
    return me.Style(
        background=MD3_COLORS["message_received"],
        color=MD3_COLORS["on_surface"],
        padding=me.Padding.all(12),
        margin=me.Margin(left=8, right=40, top=4, bottom=4),
        border_radius=18,
        border_top_left_radius=4,
        max_width="80%",
        align_self="flex-start",
        box_shadow="0 1px 2px rgba(0,0,0,0.1)"
    )

def input_container_style():
    return me.Style(
        background=MD3_COLORS["surface"],
        padding=me.Padding.all(16),
        display="flex",
        align_items="center",
        gap=8,
        border_top=f"1px solid {MD3_COLORS['outline']}"
    )

def input_field_style():
    return me.Style(
        flex="1",
        padding=me.Padding.all(12),
        border_radius=24,
        border=f"1px solid {MD3_COLORS['outline']}",
        background=MD3_COLORS["input_bg"],
        outline="none"
    )

def send_button_style():
    return me.Style(
        background=MD3_COLORS["primary"],
        color=MD3_COLORS["on_primary"],
        border="none",
        border_radius=24,
        padding=me.Padding.symmetric(horizontal=16, vertical=12),
        cursor="pointer",
        font_weight="500"
    )

def emergency_alert_style():
    return me.Style(
        background=MD3_COLORS["error"],
        color=MD3_COLORS["on_error"],
        padding=me.Padding.all(16),
        border_radius=12,
        margin=me.Margin.all(8),
        box_shadow="0 4px 8px rgba(186, 26, 26, 0.3)",
        animation="pulse 2s infinite"
    )

def file_upload_style():
    return me.Style(
        border=f"2px dashed {MD3_COLORS['outline']}",
        border_radius=12,
        padding=me.Padding.all(24),
        text_align="center",
        background=MD3_COLORS["surface_variant"],
        cursor="pointer"
    )
