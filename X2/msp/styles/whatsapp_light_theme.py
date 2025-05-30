# msp/styles/whatsapp_light_theme.py

import mesop as me

# WhatsApp Web Light Mode Palette (approximated from image [5])
WHATSAPP_LIGHT_COLORS = {
    "app_bg": "#F0F2F5",                # Overall app background (very light grey)
    "chat_area_bg": "#E5DDD5",          # Chat area background (WhatsApp doodle pattern color) [5]
    "sidebar_bg": "#FFFFFF",            # Sidebar background (white)
    "header_bg": "#00A884",             # Main header green (top bar of chat, not the chat header itself)
                                        # For individual chat header, it's typically #F0F2F5 or #FFFFFF
    "chat_header_bg": "#F0F2F5",        # Header above messages (light grey) [5]
    "active_chat_bg": "#F0F2F5",        # Selected chat in sidebar (light grey) [5]
    
    "message_sent_bg": "#D9FDD3",       # Light green for sent messages [5]
    "message_sent_text": "#0F0F0F",     # Dark text for sent messages
    
    "message_received_bg": "#FFFFFF",   # White for received messages [5]
    "message_received_text": "#0F0F0F",  # Dark text for received messages
    
    "input_bar_bg": "#F0F2F5",          # Input bar background (light grey) [5]
    "input_field_bg": "#FFFFFF",        # Text input field itself (white) [5]
    "input_field_text": "#3B4A54",      # Text typed in input field (dark grey)
    "input_placeholder_text": "#8696A0",  # Placeholder text color (medium grey)
    
    "icon_color": "#54656F",            # Icons in input bar, headers (grey) [5]
    "icon_active_color": "#00A884",     # WhatsApp Green for active send icon
    
    "timestamp_text": "#667781",        # Grey for timestamps [5]
    "link_text": "#0077CC",             # Standard link blue
    
    "divider_color": "#D1D7DB",         # Subtle dividers (light grey)
    "primary_text": "#111B21",          # Main dark text color
    "secondary_text": "#54656F",        # Secondary grey text (like status)
    "green_accent": "#00A884",          # WhatsApp Green
    "error_text": "#D32F2F",            # Standard error red
    "error_bg": "#FFDDE1",              # Light red background for error messages
}

# --- Core Layout Styles ---
def app_container_style():
    return me.Style(
        height="100vh",
        width="100vw",
        display="flex",
        flex_direction="column",
        background=WHATSAPP_LIGHT_COLORS["app_bg"],  # Overall background
        font_family="'Segoe UI', Helvetica, Arial, sans-serif",
        color=WHATSAPP_LIGHT_COLORS["primary_text"],
        overflow="hidden"
    )

def chat_area_style():  # The main content area for messages
    return me.Style(
        flex_grow=1,
        display="flex",
        flex_direction="column",
        overflow_y="auto",
        padding=me.Padding.symmetric(horizontal=0, vertical=10),
        background=WHATSAPP_LIGHT_COLORS["chat_area_bg"]  # Doodle background color [5]
        # If you have the doodle image:
        # background_image="url('/static/whatsapp_doodle_light.png')",
        # background_repeat="repeat"
    )

# --- Chat Header Styles ---
def chat_header_style():  # Header for an individual chat screen
    return me.Style(
        background=WHATSAPP_LIGHT_COLORS["chat_header_bg"],  # Light grey [5]
        padding=me.Padding.symmetric(horizontal=16, vertical=10),
        display="flex",
        align_items="center",
        justify_content="space-between",
        border=me.Border(bottom=me.BorderSide(width=1, style="solid", color=WHATSAPP_LIGHT_COLORS["divider_color"]))
    )

def chat_header_avatar_style():
    return me.Style(
        width=40,
        height=40,
        border_radius="50%",
        background="#DFE5E7",  # Placeholder color for avatar
        margin=me.Margin(right=12)
    )

def chat_header_text_style():
    return me.Style(font_size="16px", font_weight="500", color=WHATSAPP_LIGHT_COLORS["primary_text"])

def chat_header_status_style():
    return me.Style(font_size="13px", color=WHATSAPP_LIGHT_COLORS["secondary_text"])

def chat_header_icons_style():  # Icons on the right of the chat header
    return me.Style(display="flex", gap=20)

# --- Chat Bubble Styles ---
def message_bubble_base_style():  # Common properties for both sent and received
    return {  # Return as dict for easier merging
        "padding": me.Padding(top=6, bottom=8, left=9, right=9),
        "border_radius": 7.5,  # WhatsApp has slightly less rounded bubbles [5]
        "max_width": "65%",
        "box_shadow": "0 1px 0.5px rgba(0,0,0,0.13)",  # Subtle shadow [5]
        "margin_bottom": 3,
        "margin_top": 3,
    }

def sent_bubble_style():
    base = message_bubble_base_style()
    return me.Style(
        **base,  # Unpack base style dictionary
        background=WHATSAPP_LIGHT_COLORS["message_sent_bg"],
        color=WHATSAPP_LIGHT_COLORS["message_sent_text"],
        align_self="flex-end",
        margin_left="auto",
        margin_right=10
    )

def received_bubble_style():
    base = message_bubble_base_style()
    return me.Style(
        **base,  # Unpack base style dictionary
        background=WHATSAPP_LIGHT_COLORS["message_received_bg"],
        color=WHATSAPP_LIGHT_COLORS["message_received_text"],
        align_self="flex-start",
        margin_right="auto",
        margin_left=10
    )

def message_text_style():
    return me.Style(font_size="14.2px", line_height="19px", white_space="pre-wrap", word_wrap="break-word")

def timestamp_style():  # Timestamp inside the bubble
    return me.Style(
        font_size="11px",
        color=WHATSAPP_LIGHT_COLORS["timestamp_text"],
        margin=me.Margin(top=4, left=8),
        # align_self="flex-end" # This is handled by the container below
    )

# --- Chat Input Styles ---
def chat_input_bar_style():  # The overall bar at the bottom
    return me.Style(
        background=WHATSAPP_LIGHT_COLORS["input_bar_bg"],  # Light grey [5]
        padding=me.Padding.symmetric(horizontal=10, vertical=10),  # Symmetrical padding
        display="flex",
        align_items="center",  # Align items vertically in the middle
        gap=12,  # Gap between icons and input field wrapper
        width="100%",  # Ensure it spans full width
        border=me.Border(top=me.BorderSide(width=1, style="solid", color=WHATSAPP_LIGHT_COLORS["divider_color"]))  # Top border
    )

def input_field_wrapper_style():  # The rounded container for the text area
    return me.Style(
        flex_grow=1,
        background=WHATSAPP_LIGHT_COLORS["input_field_bg"],  # White [5]
        border_radius=20,  # WhatsApp input field is quite rounded [5]
        padding=me.Padding.symmetric(horizontal=12),  # Padding inside the white box
        display="flex",
        align_items="center",
        height=42  # Typical height for WhatsApp input area
    )

def native_textarea_style():
    return me.Style(
        background="transparent",
        border=me.Border.all(me.BorderSide(style="none")),
        outline="none",
        width="100%",
        color=WHATSAPP_LIGHT_COLORS["input_field_text"],  # Dark grey text
        font_size="15px",
        line_height="20px",
        overflow_y="hidden",
        padding=me.Padding(top=8, bottom=8),  # Vertical padding for text inside text area
        height="22px",  # To make it appear single line and centered
        vertical_align="middle",  # Helps center text in some browsers
        font_family="'Segoe UI', Helvetica, Arial, sans-serif",
        # Placeholder styling needs to be handled by the component itself or CSS if Mesop doesn't expose it directly
    )

def icon_button_style():  # For attach, emoji, send icons
    return me.Style(padding=me.Padding.all(8), cursor="pointer")  # Adequate touch/click area
