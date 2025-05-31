# msp/utils/extensions/__init__.py

"""
Mesop Extensions for Enhanced Functionality

This package contains extensions that enhance Mesop's native capabilities,
providing additional features like smooth scrolling, advanced animations, etc.
"""

# Make smooth scroll functions easily importable
from .smooth_scroll import (
    smooth_scroll_to_element,
    instant_scroll_to_element,
    whatsapp_smooth_scroll,
    enable_css_smooth_scroll,
    smooth_scroll,
    SmoothScrollExtension,
    ScrollAnimationState,
    scroll_progress_indicator
)

__all__ = [
    "smooth_scroll_to_element",
    "instant_scroll_to_element",
    "whatsapp_smooth_scroll",
    "enable_css_smooth_scroll",
    "smooth_scroll",
    "SmoothScrollExtension",
    "ScrollAnimationState",
    "scroll_progress_indicator"
]
