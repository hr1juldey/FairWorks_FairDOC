# msp/components/shared_navigation.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from styles.government_digital_styles import GOVERNMENT_COLORS
from utils.home_page_helpers import navigate_to_chat, navigate_to_report

def render_shared_navigation():
    """Shared navigation header for all pages"""
    with me.box(style=me.Style(
        background="rgba(255, 255, 255, 0.95)",
        backdrop_filter="blur(10px)",
        border=me.Border(bottom=me.BorderSide(width=1, style="solid", color="rgba(19, 102, 217, 0.1)")),
        padding=me.Padding.symmetric(vertical=16),
        position="sticky",
        top="0",
        z_index=100
    )):
        with me.box(style=me.Style(
            display="flex",
            justify_content="space-between",
            align_items="center",
            max_width="1200px",
            margin=me.Margin.symmetric(horizontal="auto"),
            padding=me.Padding.symmetric(horizontal=24)
        )):
            # Brand Logo
            render_brand_section()
            
            # Navigation Links
            render_nav_links()
            
            # CTA Section
            render_nav_cta()

def render_brand_section():
    """Render brand/logo section"""
    with me.box(style=me.Style(display="flex", align_items="center", gap=12)):
        me.text("🏥", style=me.Style(font_size="2rem"))
        me.button(
            "Fairdoc AI",
            on_click=lambda e: me.navigate("/"),
            style=me.Style(
                background="transparent",
                # REMOVED: border="none", - not needed in Mesop
                font_size="1.5rem",
                font_weight="700",
                color=GOVERNMENT_COLORS["primary"],
                cursor="pointer"
            )
        )

def render_nav_links():
    """Render navigation links"""
    with me.box(style=me.Style(display="flex", gap=8, align_items="center")):
        me.button(
            "🏠 Home",
            on_click=lambda e: me.navigate("/"),
            style=me.Style(
                background="transparent",
                # REMOVED: border="none", - not needed in Mesop
                color=GOVERNMENT_COLORS["text_secondary"],
                font_weight="500",
                padding=me.Padding.symmetric(horizontal=16, vertical=8),
                border_radius="6px",
                cursor="pointer"
            )
        )
        me.button(
            "💬 Triage Chat",
            on_click=navigate_to_chat,
            style=me.Style(
                background="transparent",
                # REMOVED: border="none", - not needed in Mesop
                color=GOVERNMENT_COLORS["text_secondary"],
                font_weight="500",
                padding=me.Padding.symmetric(horizontal=16, vertical=8),
                border_radius="6px",
                cursor="pointer"
            )
        )
        me.button(
            "📊 View Reports",
            on_click=navigate_to_report,
            style=me.Style(
                background="transparent",
                # REMOVED: border="none", - not needed in Mesop
                color=GOVERNMENT_COLORS["text_secondary"],
                font_weight="500",
                padding=me.Padding.symmetric(horizontal=16, vertical=8),
                border_radius="6px",
                cursor="pointer"
            )
        )

def render_nav_cta():
    """Render navigation CTA button"""
    me.button(
        "🚀 Start Triage",
        on_click=navigate_to_chat,
        style=me.Style(
            background=f"linear-gradient(135deg, {GOVERNMENT_COLORS['primary']} 0%, {GOVERNMENT_COLORS['primary_dark']} 100%)",
            color="white",
            padding=me.Padding.symmetric(horizontal=20, vertical=8),
            border_radius="6px",
            font_weight="600",
            # REMOVED: border="none", - not needed in Mesop
            cursor="pointer"
        )
    )
