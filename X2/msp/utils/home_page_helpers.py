# msp/utils/home_page_helpers.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me

def navigate_to_chat(e: me.ClickEvent):
    """Navigate to chat page"""
    me.navigate("/chat")

def navigate_to_report(e: me.ClickEvent):
    """Navigate to report page"""
    me.navigate("/report")

def is_mobile_viewport():
    """Check if current viewport is mobile sized"""
    return me.viewport_size().width < 768

def format_statistic(value: str, unit: str = "") -> str:
    """Format statistic values with units"""
    return f"{value}{unit}"

def get_responsive_padding(mobile_padding: int = 16, desktop_padding: int = 24) -> me.Padding:
    """Get responsive padding based on viewport"""
    if is_mobile_viewport():
        return me.Padding.all(mobile_padding)
    return me.Padding.all(desktop_padding)

def get_responsive_margin(mobile_margin: int = 16, desktop_margin: int = 24) -> me.Margin:
    """Get responsive margin based on viewport"""
    if is_mobile_viewport():
        return me.Margin.all(mobile_margin)
    return me.Margin.all(desktop_margin)
