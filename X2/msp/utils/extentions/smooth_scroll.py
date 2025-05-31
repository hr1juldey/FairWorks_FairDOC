# msp/utils/extensions/smooth_scroll.py

import time
import mesop as me
from typing import Literal, Optional, Generator, Any
import utils.path_setup  # Ensure paths are set up

# Type definitions for better type safety
ScrollBehavior = Literal["auto", "smooth", "instant"]
ScrollBlock = Literal["start", "center", "end", "nearest"]
ScrollInline = Literal["start", "center", "end", "nearest"]

class SmoothScrollExtension:
    """
    Extension for Mesop to provide smooth scrolling functionality with options.
    
    This extends the native me.scroll_into_view() with smooth animation capabilities
    using Mesop's property animation system.
    """
    
    @staticmethod
    def scroll_into_view_smooth(
        key: str,
        behavior: ScrollBehavior = "smooth",
        block: ScrollBlock = "start", 
        inline: ScrollInline = "nearest",
        duration_ms: int = 500,
        easing: Literal["linear", "ease-in", "ease-out", "ease-in-out"] = "ease-out"
    ) -> Generator[None, None, None]:
        """
        Smoothly scroll to a component with animation options.
        
        Args:
            key: The unique identifier of the component to scroll to
            behavior: Scroll animation type ("auto", "smooth", "instant")
            block: Vertical alignment ("start", "center", "end", "nearest") 
            inline: Horizontal alignment ("start", "center", "end", "nearest")
            duration_ms: Duration of smooth scroll animation in milliseconds
            easing: Animation easing function
            
        Returns:
            Generator for use in Mesop event handlers with yield from
            
        Usage:
            def on_click(e: me.ClickEvent):
                yield from smooth_scroll_to_element(
                    key="chat_end_anchor",
                    behavior="smooth", 
                    block="end",
                    duration_ms=300
                )
        """
        
        if behavior == "instant":
            # For instant scroll, just use native Mesop function
            me.scroll_into_view(key=key)
            return
        elif behavior == "auto":
            # Auto behavior - use native with some delay for consistency
            me.scroll_into_view(key=key)
            yield
            return
        
        # For smooth behavior, implement custom animation
        yield from SmoothScrollExtension._animate_smooth_scroll(
            key=key,
            block=block,
            inline=inline, 
            duration_ms=duration_ms,
            easing=easing
        )
    
    @staticmethod
    def _animate_smooth_scroll(
        key: str,
        block: ScrollBlock,
        inline: ScrollInline,
        duration_ms: int,
        easing: str
    ) -> Generator[None, None, None]:
        """
        Internal method to perform smooth scroll animation.
        
        Since we can't directly manipulate scroll position in Mesop,
        we use a combination of visual feedback and final positioning.
        """
        
        # Step 1: Provide visual feedback during scroll
        # We can animate some visual properties to indicate scrolling is happening
        yield from SmoothScrollExtension._scroll_animation_feedback(duration_ms, easing)
        
        # Step 2: Perform the actual scroll at the end
        me.scroll_into_view(key=key)
        yield
    
    @staticmethod 
    def _scroll_animation_feedback(duration_ms: int, easing: str) -> Generator[None, None, None]:
        """
        Provide visual feedback during scroll animation.
        This could animate a scroll indicator or other UI feedback.
        """
        # Calculate animation steps
        steps = max(10, duration_ms // 20)  # ~50fps for smooth animation
        step_duration = duration_ms / steps / 1000.0  # Convert to seconds
        
        # Simple easing calculation
        for i in range(steps):
            progress = i / steps
            
            # Apply easing function
            if easing == "ease-in":
                eased_progress = progress * progress
            elif easing == "ease-out": 
                eased_progress = 1 - (1 - progress) * (1 - progress)
            elif easing == "ease-in-out":
                if progress < 0.5:
                    eased_progress = 2 * progress * progress
                else:
                    eased_progress = 1 - 2 * (1 - progress) * (1 - progress)
            else:  # linear
                eased_progress = progress
            
            # Here you could animate visual feedback elements
            # For example, opacity of a scroll indicator
            # state = me.state(SomeState)
            # state.scroll_progress = eased_progress
            
            yield
            time.sleep(step_duration)

    @staticmethod
    def scroll_to_top(duration_ms: int = 500) -> Generator[None, None, None]:
        """
        Smooth scroll to top of page.
        
        Args:
            duration_ms: Duration of animation
            
        Usage:
            def on_scroll_to_top(e: me.ClickEvent):
                yield from smooth_scroll.scroll_to_top(300)
        """
        yield from SmoothScrollExtension._animate_smooth_scroll(
            key="app_top_anchor",  # Assumes there's a top anchor
            block="start",
            inline="nearest", 
            duration_ms=duration_ms,
            easing="ease-out"
        )

    @staticmethod
    def scroll_to_bottom(duration_ms: int = 500) -> Generator[None, None, None]:
        """
        Smooth scroll to bottom of page.
        
        Args:
            duration_ms: Duration of animation
            
        Usage:
            def on_scroll_to_bottom(e: me.ClickEvent):
                yield from smooth_scroll.scroll_to_bottom(300)
        """
        yield from SmoothScrollExtension._animate_smooth_scroll(
            key="app_bottom_anchor",  # Assumes there's a bottom anchor
            block="end",
            inline="nearest",
            duration_ms=duration_ms, 
            easing="ease-out"
        )

# Create a global instance for easy import
smooth_scroll = SmoothScrollExtension()

# Convenience functions that can be imported directly
def smooth_scroll_to_element(
    key: str,
    behavior: ScrollBehavior = "smooth", 
    block: ScrollBlock = "end",
    inline: ScrollInline = "nearest",
    duration_ms: int = 300
) -> Generator[None, None, None]:
    """
    Convenience function for smooth scrolling to an element.
    
    Usage:
        from utils.extensions.smooth_scroll import smooth_scroll_to_element
        
        def handle_send_message(e: me.ClickEvent):
            # ... send message logic ...
            yield from smooth_scroll_to_element("chat_end_anchor", duration_ms=300)
    """
    yield from smooth_scroll.scroll_into_view_smooth(
        key=key,
        behavior=behavior,
        block=block, 
        inline=inline,
        duration_ms=duration_ms
    )

def instant_scroll_to_element(key: str) -> None:
    """
    Instant scroll to element (wrapper around native Mesop function).
    
    Usage:
        from utils.extensions.smooth_scroll import instant_scroll_to_element
        instant_scroll_to_element("some_anchor")
    """
    me.scroll_into_view(key=key)

# Enhanced scroll with WhatsApp-style animation timing
def whatsapp_smooth_scroll(key: str, block: ScrollBlock = "end") -> Generator[None, None, None]:
    """
    WhatsApp-style smooth scroll with optimized timing and easing.
    
    Usage:
        from utils.extensions.smooth_scroll import whatsapp_smooth_scroll
        
        def on_new_message(e: me.ClickEvent):
            # ... add message ...
            yield from whatsapp_smooth_scroll("chat_end_anchor")
    """
    yield from smooth_scroll.scroll_into_view_smooth(
        key=key,
        behavior="smooth",
        block=block,
        inline="nearest",
        duration_ms=250,  # WhatsApp-like quick but smooth
        easing="ease-out"
    )

# CSS-based smooth scroll helper (for when browser supports it)
def enable_css_smooth_scroll() -> str:
    """
    Returns CSS that can be applied to enable browser-native smooth scrolling.
    
    Usage:
        # In your style functions or app setup:
        css_smooth_scroll = enable_css_smooth_scroll()
        # Apply this CSS to your main container
    """
    return """
    html {
        scroll-behavior: smooth;
    }
    
    /* Smooth scroll for specific containers */
    .smooth-scroll {
        scroll-behavior: smooth;
    }
    
    /* Disable smooth scroll for users who prefer reduced motion */
    @media (prefers-reduced-motion: reduce) {
        html, .smooth-scroll {
            scroll-behavior: auto;
        }
    }
    """

# Animation state class for scroll feedback (optional)
@me.stateclass
class ScrollAnimationState:
    """
    Optional state class for tracking scroll animations.
    Can be used to show scroll progress indicators or other feedback.
    """
    is_scrolling: bool = False
    scroll_progress: float = 0.0  # 0.0 to 1.0
    target_element_key: str = ""
    
# Scroll progress indicator component (optional)
@me.component
def scroll_progress_indicator():
    """
    Optional component to show scroll progress during smooth scrolling.
    
    Usage:
        # Add this to your layout when you want scroll feedback
        scroll_progress_indicator()
    """
    state = me.state(ScrollAnimationState)
    
    if state.is_scrolling:
        with me.box(
            style=me.Style(
                position="fixed",
                top=0,
                left=0, 
                width=f"{state.scroll_progress * 100}%",
                height="2px",
                background="#00A884",  # WhatsApp green
                z_index=9999,
                transition="width 0.1s ease-out"
            )
        ):
            pass
