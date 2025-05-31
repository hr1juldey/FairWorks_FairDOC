# msp/utils/extensions/webcomponents.py

import asyncio
from typing import Any, Callable, Literal, Optional, Generator
import mesop as me
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()

# Type definitions
ScrollBehavior = Literal["auto", "smooth", "instant"]
ScrollBlock = Literal["start", "center", "end", "nearest"]
ScrollInline = Literal["start", "center", "end", "nearest"]
ScrollEasing = Literal["linear", "ease-in", "ease-out", "ease-in-out"]

@me.web_component(path="./smooth_scroll_component.js")
def smooth_scroll_web_component(
    *,
    target_key: str = "",
    behavior: ScrollBehavior = "smooth",
    block: ScrollBlock = "end",
    inline: ScrollInline = "nearest",
    duration: int = 300,
    easing: ScrollEasing = "ease-out",
    show_progress: bool = True,
    trigger_event: str = "",
    on_scroll_start: Optional[Callable[[me.WebEvent], Any]] = None,
    on_scroll_progress: Optional[Callable[[me.WebEvent], Any]] = None,
    on_scroll_complete: Optional[Callable[[me.WebEvent], Any]] = None,
    key: Optional[str] = None,
):
    """
    Advanced smooth scroll web component with React-like behaviors.
    Includes built-in security policy overrides for CDN resources.
    """
    return me.insert_web_component(
        name="smooth-scroll-component",
        key=key or f"smooth_scroll_{target_key}",
        events={
            "scrollStart": on_scroll_start,
            "scrollProgress": on_scroll_progress,
            "scrollComplete": on_scroll_complete,
        },
        properties={
            "targetKey": target_key,
            "behavior": behavior,
            "block": block,
            "inline": inline,
            "duration": duration,
            "easing": easing,
            "showProgress": show_progress,
            "trigger": trigger_event,
        },
    )

# State class for scroll management
@me.stateclass
class SmoothScrollState:
    """State for tracking smooth scroll operations."""
    current_target: str = ""
    is_scrolling: bool = False
    scroll_progress: float = 0.0
    last_scroll_duration: float = 0.0
    scroll_queue: list[str] = None

# High-level API functions
class SmoothScrollAPI:
    """High-level API for smooth scrolling with async/sync support."""
    
    @staticmethod
    def scroll_to_element(
        target_key: str,
        behavior: ScrollBehavior = "smooth",
        duration: int = 300,
        show_progress: bool = True
    ) -> Generator[None, None, None]:
        """Sync version: Smooth scroll to element with generator for Mesop compatibility."""
        state = me.state(SmoothScrollState)
        state.current_target = target_key
        state.is_scrolling = True
        yield
        
        estimated_duration = duration / 1000.0
        steps = max(10, duration // 50)
        step_duration = estimated_duration / steps
        
        for i in range(steps):
            progress = (i + 1) / steps
            state.scroll_progress = progress
            yield
            import time
            time.sleep(step_duration)
        
        state.is_scrolling = False
        state.scroll_progress = 1.0
        yield

    @staticmethod
    async def async_scroll_to_element(
        target_key: str,
        behavior: ScrollBehavior = "smooth",
        duration: int = 300
    ) -> None:
        """Async version: For use with me.effects or direct async calls."""
        await asyncio.sleep(duration / 1000.0)

    @staticmethod
    def whatsapp_scroll(target_key: str) -> Generator[None, None, None]:
        """WhatsApp-style quick smooth scroll."""
        yield from SmoothScrollAPI.scroll_to_element(
            target_key=target_key,
            behavior="smooth",
            duration=250,
            show_progress=False
        )

    @staticmethod
    def instant_scroll(target_key: str) -> None:
        """Instant scroll (no generator needed)."""
        state = me.state(SmoothScrollState)
        state.current_target = target_key

# Event handlers for scroll feedback
def handle_scroll_start(event: me.WebEvent):
    """Handle scroll start event from web component."""
    state = me.state(SmoothScrollState)
    state.is_scrolling = True
    state.current_target = event.value.get("targetKey", "")
    state.scroll_progress = 0.0

def handle_scroll_progress(event: me.WebEvent):
    """Handle scroll progress event from web component."""
    state = me.state(SmoothScrollState)
    state.scroll_progress = event.value.get("progress", 0.0)

def handle_scroll_complete(event: me.WebEvent):
    """Handle scroll completion event from web component."""
    state = me.state(SmoothScrollState)
    state.is_scrolling = False
    state.scroll_progress = 1.0
    state.last_scroll_duration = event.value.get("totalDuration", 0.0)

# Global scroll manager instance
scroll_api = SmoothScrollAPI()

# Convenience functions for easy importing
def smooth_scroll_to_element(target_key: str, **kwargs) -> Generator[None, None, None]:
    """Convenience function for smooth scrolling."""
    yield from scroll_api.scroll_to_element(target_key, **kwargs)

def whatsapp_smooth_scroll(target_key: str) -> Generator[None, None, None]:
    """WhatsApp-style smooth scroll."""
    yield from scroll_api.whatsapp_scroll(target_key)

def instant_scroll_to_element(target_key: str) -> None:
    """Instant scroll to element."""
    scroll_api.instant_scroll(target_key)

async def async_smooth_scroll(target_key: str, **kwargs) -> None:
    """Async smooth scroll for use with me.effects."""
    await scroll_api.async_scroll_to_element(target_key, **kwargs)

# Security policy override for CDN resources
def get_webcomponent_security_policy():
    """
    Returns a SecurityPolicy that allows jsdelivr CDN resources.
    Use this in your @me.page decorator.
    """
    return me.SecurityPolicy(
        allowed_script_srcs=[
            "https://cdn.jsdelivr.net",  # For Lit library and other CDN resources
            "https://*.jsdelivr.net",    # Wildcard for subdomains
        ],
        # Add other common CDN domains as needed
        allowed_connect_srcs=[
            "https://cdn.jsdelivr.net",
            "https://*.jsdelivr.net",
        ]
    )
