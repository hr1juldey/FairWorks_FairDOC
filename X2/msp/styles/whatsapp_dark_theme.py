import mesop as me

# WhatsApp Web Dark Mode Palette (approximated)
WHATSAPP_DARK_COLORS = {
    "app_bg": "#0B141A",
    "sidebar_bg": "#111B21",
    "header_bg": "#202C33",
    "active_chat_bg": "#2A3942",
    "message_sent_bg": "#005C4B",
    "message_sent_text": "#E9EDEF",
    "message_received_bg": "#202C33",
    "message_received_text": "#E9EDEF",
    "input_field_bg": "#2A3942",
    "input_field_text": "#E9EDEF",
    "input_placeholder_text": "#8696A0",
    "icon_color": "#AEBAC1",
    "icon_active_color": "#00A884",  # WhatsApp Green for send
    "timestamp_text": "#8696A0",
    "link_text": "#53BDEB",
    "divider_color": "#2C3E46",
    "primary_text": "#E9EDEF",
    "secondary_text": "#AEBAC1",
    "green_accent": "#00A884",  # Used for active send icon
    "error_text": "#F37A7A",
    "error_bg": "#4B2226",
}

# --- Core Layout Styles ---
def app_container_style():
    return me.Style(
        height="100vh",
        width="100vw",
        display="flex",
        flex_direction="column",
        background=WHATSAPP_DARK_COLORS["app_bg"],
        font_family="'Segoe UI', Helvetica, Arial, sans-serif",
        color=WHATSAPP_DARK_COLORS["primary_text"],
        overflow="hidden"
    )

def chat_area_style():
    return me.Style(
        flex_grow=1,
        display="flex",
        flex_direction="column",
        overflow_y="auto",
        padding=me.Padding.symmetric(horizontal=0, vertical=10),
        background=WHATSAPP_DARK_COLORS["app_bg"]
    )

# --- Chat Header Styles ---
def chat_header_style():
    return me.Style(
        background=WHATSAPP_DARK_COLORS["header_bg"],
        padding=me.Padding.symmetric(horizontal=16, vertical=10),
        display="flex",
        align_items="center",
        justify_content="space-between",
        border=me.Border(bottom=me.BorderSide(width=1, style="solid", color=WHATSAPP_DARK_COLORS["divider_color"]))
    )

def chat_header_avatar_style():
    return me.Style(
        width=40,
        height=40,
        border_radius="50%",
        background=WHATSAPP_DARK_COLORS["active_chat_bg"],
        margin=me.Margin(right=12)
    )

def chat_header_text_style():
    return me.Style(font_size="16px", font_weight="500", color=WHATSAPP_DARK_COLORS["primary_text"])

def chat_header_status_style():
    return me.Style(font_size="13px", color=WHATSAPP_DARK_COLORS["secondary_text"])

def chat_header_icons_style():
    return me.Style(display="flex", gap=20)

# --- Chat Bubble Styles ---
# REMOVED message_bubble_base_style() function to avoid complex inheritance issues

def sent_bubble_style():
    return me.Style(
        # Base properties directly inlined:
        padding=me.Padding(top=6, bottom=8, left=9, right=9),
        border_radius=8,
        max_width="65%",
        box_shadow="0 1px 0.5px rgba(0,0,0,0.3)",
        # margin=me.Margin(bottom=3), # This will be overridden by more specific margin below

        # Sent-specific properties:
        background=WHATSAPP_DARK_COLORS["message_sent_bg"],
        color=WHATSAPP_DARK_COLORS["message_sent_text"],
        align_self="flex-end", # Align to right
        margin=me.Margin(left="auto", bottom=3, right=10, top=3) # Specific margin for sent
    )

def received_bubble_style():
    return me.Style(
        # Base properties directly inlined:
        padding=me.Padding(top=6, bottom=8, left=9, right=9),
        border_radius=8,
        max_width="65%",
        box_shadow="0 1px 0.5px rgba(0,0,0,0.3)",
        # margin=me.Margin(bottom=3), # This will be overridden by more specific margin below

        # Received-specific properties:
        background=WHATSAPP_DARK_COLORS["message_received_bg"],
        color=WHATSAPP_DARK_COLORS["message_received_text"],
        align_self="flex-start", # Align to left
        margin=me.Margin(right="auto", bottom=3, left=10, top=3) # Specific margin for received
    )

def message_text_style():
    return me.Style(font_size="14.2px", line_height="19px", white_space="pre-wrap", word_wrap="break-word")

def timestamp_style():
    return me.Style(
        font_size="11px",
        color=WHATSAPP_DARK_COLORS["timestamp_text"],
        margin=me.Margin(top=4, left=8), # This margin is for the text itself
        # align_self="flex-end" # This should be on the container holding the timestamp text and ticks, if any
    )

# --- Chat Input Styles ---
def chat_input_bar_style():
    return me.Style(
        background=WHATSAPP_DARK_COLORS["header_bg"],
        padding=me.Padding.symmetric(horizontal=10, vertical=10),
        display="flex",
        align_items="center",
        gap=12,
        width="100%"
    )

def input_field_wrapper_style():
    return me.Style(
        flex_grow=1,
        background=WHATSAPP_DARK_COLORS["input_field_bg"],
        border_radius=8,
        padding=me.Padding.symmetric(horizontal=12),
        display="flex",
        align_items="center",
        height=42
    )

def native_textarea_style():
    return me.Style(
        background="transparent",
        border=me.Border.all(me.BorderSide(style="none")),
        outline="none",
        width="100%",
        color=WHATSAPP_DARK_COLORS["input_field_text"],
        font_size="15px",
        line_height="20px",
        overflow_y="hidden",
        padding=me.Padding(top=6, bottom=12),
        height="32px",
        vertical_align="middle",
        font_family="'Segoe UI', Helvetica, Arial, sans-serif"
    )

def icon_button_style():
    return me.Style(padding=me.Padding.all(8), cursor="pointer")