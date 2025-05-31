# msp/styles/government_digital_styles.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me

# Government Digital Infrastructure Color Palette
GOVERNMENT_COLORS = {
    "primary": "#1366D9",          # Government Blue
    "primary_dark": "#0D47A1",     # Darker Blue  
    "secondary": "#00A884",        # WhatsApp Green (trust/health)
    "accent": "#FF6B35",           # Saffron Orange (energy)
    "success": "#4CAF50",          # Success Green
    "warning": "#FFC107",          # Warning Amber
    "error": "#F44336",            # Error Red
    
    # Neutral Colors
    "white": "#FFFFFF",
    "light_grey": "#F5F7FA", 
    "medium_grey": "#8A9BA8",
    "dark_grey": "#2C3E50",
    "black": "#1A1A1A",
    
    # Background Colors
    "bg_primary": "#FFFFFF",
    "bg_secondary": "#F8FAFC",
    "bg_accent": "#E3F2FD",
    
    # Text Colors
    "text_primary": "#1A1A1A",
    "text_secondary": "#4A5568", 
    "text_muted": "#718096",
}

# Base Styles
def base_page_style():
    """Base style for the entire page"""
    return me.Style(
        min_height="100vh",
        font_family="'Inter', 'Segoe UI', 'Roboto', -apple-system, sans-serif",
        background=GOVERNMENT_COLORS["white"]
    )

# Navigation Styles
def navigation_container_style():
    """Main navigation container style"""
    return me.Style(
        display="flex",
        justify_content="space-between",
        align_items="center",
        max_width="1200px",
        margin=me.Margin.symmetric(horizontal="auto"),
        padding=me.Padding.symmetric(horizontal=24)
    )

def nav_brand_container_style():
    """Navigation brand/logo container"""
    return me.Style(
        display="flex", 
        align_items="center", 
        gap=12
    )

def nav_brand_text_style():
    """Navigation brand text style"""
    return me.Style(
        font_size="1.5rem", 
        font_weight="700", 
        color=GOVERNMENT_COLORS["primary"]
    )

def nav_links_container_style():
    """Navigation links container"""
    return me.Style(
        display="flex", 
        gap=8, 
        align_items="center"
    )

def nav_link_button_style():
    """Style for navigation link buttons"""
    return me.Style(
        background="transparent",
        color=GOVERNMENT_COLORS["text_secondary"],
        font_weight="500",
        padding=me.Padding.symmetric(horizontal=16, vertical=8),
        border_radius="6px",
        cursor="pointer"
    )

def nav_cta_button_style():
    """Style for navigation CTA button"""
    return me.Style(
        background=f"linear-gradient(135deg, {GOVERNMENT_COLORS['primary']} 0%, {GOVERNMENT_COLORS['primary_dark']} 100%)",
        color="white",
        padding=me.Padding.symmetric(horizontal=20, vertical=8),
        border_radius="6px",
        font_weight="600",
        cursor="pointer"
    )

# Hero Section Styles
def hero_section_style():
    """Hero section container style"""
    return me.Style(
        padding=me.Padding.symmetric(horizontal=24, vertical=80),
        text_align="center",
        background="linear-gradient(135deg, rgba(19, 102, 217, 0.05) 0%, rgba(0, 168, 132, 0.05) 50%, rgba(255, 107, 53, 0.05) 100%)"
    )

def hero_content_container_style():
    """Hero content container"""
    return me.Style(
        max_width="800px", 
        margin=me.Margin.symmetric(horizontal="auto")
    )

def hero_badge_style():
    """Hero badge style"""
    return me.Style(
        background="rgba(19, 102, 217, 0.1)",
        color=GOVERNMENT_COLORS["primary"],
        padding=me.Padding.symmetric(horizontal=16, vertical=8),
        border_radius="20px",
        font_size="0.875rem",
        font_weight="600",
        display="inline-block",
        margin=me.Margin(bottom=24)
    )

def hero_headline_style():
    """Hero main headline style"""
    return me.Style(
        font_size="3.5rem",
        font_weight="700",
        line_height="1.1",
        color=GOVERNMENT_COLORS["text_primary"],
        margin=me.Margin(bottom=24)
    )

def hero_subtitle_style():
    """Hero subtitle style"""
    return me.Style(
        font_size="1.25rem",
        line_height="1.6",
        color=GOVERNMENT_COLORS["text_secondary"],
        margin=me.Margin(bottom=40)
    )

def hero_buttons_container_style():
    """Hero buttons container"""
    return me.Style(
        display="flex", 
        gap=16, 
        justify_content="center", 
        flex_wrap="wrap"
    )

def hero_primary_button_style():
    """Hero primary CTA button"""
    return me.Style(
        background=f"linear-gradient(135deg, {GOVERNMENT_COLORS['primary']} 0%, {GOVERNMENT_COLORS['primary_dark']} 100%)",
        color="white",
        padding=me.Padding.symmetric(horizontal=32, vertical=16),
        border_radius="8px",
        font_weight="600",
        font_size="1.1rem",
        cursor="pointer",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)"
    )

def hero_secondary_button_style():
    """Hero secondary button"""
    return me.Style(
        background="transparent",
        color=GOVERNMENT_COLORS["primary"],
        padding=me.Padding.symmetric(horizontal=32, vertical=16),
        # FIXED: Use proper me.Border object instead of string
        border=me.Border.all(me.BorderSide(width=2, style="solid", color=GOVERNMENT_COLORS["primary"])),
        border_radius="8px",
        font_weight="600",
        font_size="1.1rem",
        cursor="pointer"
    )

# Feature Section Styles
def features_section_style():
    """Features section container"""
    return me.Style(
        padding=me.Padding.symmetric(horizontal=24, vertical=80), 
        background=GOVERNMENT_COLORS["bg_secondary"]
    )

def features_container_style():
    """Features content container"""
    return me.Style(
        max_width="1200px", 
        margin=me.Margin.symmetric(horizontal="auto")
    )

def section_headline_style():
    """Section headline style"""
    return me.Style(
        text_align="center", 
        margin=me.Margin(bottom=16), 
        color=GOVERNMENT_COLORS["text_primary"], 
        font_weight="600"
    )

def section_subtitle_style():
    """Section subtitle style"""
    return me.Style(
        text_align="center", 
        margin=me.Margin(bottom=60), 
        color=GOVERNMENT_COLORS["text_secondary"], 
        font_size="1.1rem"
    )

def features_grid_style():
    """Features grid container"""
    return me.Style(
        display="grid",
        grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))",
        gap=32
    )

def feature_card_style():
    """Individual feature card style"""
    return me.Style(
        background="white",
        border_radius="12px",
        padding=me.Padding.all(32),
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
        # FIXED: Use proper me.Border object
        border=me.Border.all(me.BorderSide(width=1, style="solid", color="rgba(19, 102, 217, 0.1)"))
    )

def feature_icon_style():
    """Feature card icon style"""
    return me.Style(
        width="64px",
        height="64px",
        background=f"linear-gradient(135deg, {GOVERNMENT_COLORS['primary']} 0%, {GOVERNMENT_COLORS['secondary']} 100%)",
        border_radius="16px",
        display="flex",
        align_items="center",
        justify_content="center",
        margin=me.Margin(bottom=24),
        font_size="24px"
    )

def feature_title_style():
    """Feature card title style"""
    return me.Style(
        margin=me.Margin(bottom=12), 
        color=GOVERNMENT_COLORS["text_primary"], 
        font_weight="600"
    )

def feature_description_style():
    """Feature card description style"""
    return me.Style(
        color=GOVERNMENT_COLORS["text_secondary"], 
        line_height="1.6"
    )

# Statistics Section Styles
def stats_section_style():
    """Statistics section container"""
    return me.Style(
        padding=me.Padding.symmetric(horizontal=24, vertical=80), 
        background="white"
    )

def stats_container_style():
    """Statistics content container"""
    return me.Style(
        max_width="1000px", 
        margin=me.Margin.symmetric(horizontal="auto"), 
        text_align="center"
    )

def stats_grid_style():
    """Statistics grid container"""
    return me.Style(
        display="grid",
        grid_template_columns="repeat(auto-fit, minmax(200px, 1fr))",
        gap=40
    )

def stat_container_style():
    """Individual statistic container"""
    return me.Style(
        text_align="center", 
        padding=me.Padding.all(24)
    )

def stat_number_style():
    """Statistic number style"""
    return me.Style(
        font_size="3rem",
        font_weight="700",
        color=GOVERNMENT_COLORS["primary"],
        line_height="1",
        margin=me.Margin(bottom=8)
    )

def stat_label_style():
    """Statistic label style"""
    return me.Style(
        color=GOVERNMENT_COLORS["text_secondary"], 
        font_weight="500"
    )

# Trust Section Styles
def trust_section_style():
    """Trust section container"""
    return me.Style(
        padding=me.Padding.symmetric(horizontal=24, vertical=60), 
        background=GOVERNMENT_COLORS["bg_accent"]
    )

def trust_container_style():
    """Trust content container"""
    return me.Style(
        max_width="800px", 
        margin=me.Margin.symmetric(horizontal="auto"), 
        text_align="center"
    )

def trust_badges_container_style():
    """Trust badges container"""
    return me.Style(
        display="flex",
        justify_content="center",
        gap=24,
        flex_wrap="wrap"
    )

def trust_badge_style():
    """Individual trust badge style"""
    return me.Style(
        background="rgba(0, 168, 132, 0.1)",
        color=GOVERNMENT_COLORS["secondary"],
        padding=me.Padding.symmetric(horizontal=16, vertical=8),
        border_radius="8px",
        font_size="0.875rem",
        font_weight="600"
    )

# CTA Section Styles
def cta_section_style():
    """CTA section container"""
    return me.Style(
        padding=me.Padding.symmetric(horizontal=24, vertical=80), 
        background=GOVERNMENT_COLORS["primary"]
    )

def cta_container_style():
    """CTA content container"""
    return me.Style(
        max_width="600px",
        margin=me.Margin.symmetric(horizontal="auto"),
        text_align="center"
    )

def cta_headline_style():
    """CTA headline style"""
    return me.Style(
        margin=me.Margin(bottom=16),
        color="white",
        font_weight="600"
    )

def cta_text_style():
    """CTA description text style"""
    return me.Style(
        margin=me.Margin(bottom=32),
        color="rgba(255, 255, 255, 0.9)",
        font_size="1.1rem",
        line_height="1.6"
    )

def cta_button_style():
    """CTA button style"""
    return me.Style(
        background="white",
        color=GOVERNMENT_COLORS["primary"],
        padding=me.Padding.symmetric(horizontal=32, vertical=16),
        border_radius="8px",
        font_weight="600",
        font_size="1.1rem",
        cursor="pointer",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)"
    )

# Footer Styles
def footer_section_style():
    """Footer section container"""
    return me.Style(
        padding=me.Padding.symmetric(horizontal=24, vertical=40),
        background=GOVERNMENT_COLORS["black"]
    )

def footer_container_style():
    """Footer content container"""
    return me.Style(
        max_width="1200px",
        margin=me.Margin.symmetric(horizontal="auto"),
        text_align="center"
    )

def footer_title_style():
    """Footer title style"""
    return me.Style(
        color="rgba(255, 255, 255, 0.9)",
        font_weight="600",
        margin=me.Margin(bottom=16)
    )

def footer_text_style():
    """Footer description text style"""
    return me.Style(
        color="rgba(255, 255, 255, 0.7)",
        font_size="0.9rem"
    )
