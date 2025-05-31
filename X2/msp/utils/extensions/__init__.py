# msp/utils/extensions/__init__.py

"""
Mesop Extensions for Enhanced Functionality

This package contains extensions that enhance Mesop's native capabilities,
providing additional features like smooth scrolling, advanced animations, etc.
"""

# Import web component-based functionality
from .webcomponents import (
    smooth_scroll_web_component,
    smooth_scroll_to_element,
    whatsapp_smooth_scroll,
    instant_scroll_to_element,
    async_smooth_scroll,
    SmoothScrollAPI,
    SmoothScrollState,
    handle_scroll_start,
    handle_scroll_progress,
    handle_scroll_complete,
    scroll_api,
    get_webcomponent_security_policy  # This function is now properly defined
)

__all__ = [
    "smooth_scroll_web_component",
    "smooth_scroll_to_element",
    "whatsapp_smooth_scroll",
    "instant_scroll_to_element",
    "async_smooth_scroll",
    "SmoothScrollAPI",
    "SmoothScrollState",
    "handle_scroll_start",
    "handle_scroll_progress",
    "handle_scroll_complete",
    "scroll_api",
    "get_webcomponent_security_policy"
]
