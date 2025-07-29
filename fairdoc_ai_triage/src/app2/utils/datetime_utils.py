"""
DateTime Utilities for Fairdoc AI V2

Drop-in replacements for deprecated datetime methods
with timezone-aware implementations.

Single responsibility: DateTime utility functions
File: src/app2/utils/datetime_utils.py
"""

from datetime import datetime, timezone
from typing import str as StrType


def utcnow() -> datetime:
    """
    Return timezone-aware UTC datetime
    
    This function replaces the deprecated datetime.utcnow() method
    with the recommended timezone-aware approach using UTC timezone.
    
    Returns:
        datetime: Current UTC time with timezone information
        
    Example:
        >>> from src.app2.utils.datetime_utils import utcnow
        >>> now = utcnow()
        >>> now.tzinfo  # Will be timezone.utc
        datetime.timezone.utc
    """
    return datetime.now(timezone.utc)


def utcnow_iso() -> StrType:
    """
    Return timezone-aware UTC datetime as ISO string
    
    Common pattern used throughout the API for JSON responses.
    Combines utcnow() with .isoformat() for consistent formatting.
    
    Returns:
        str: ISO formatted UTC datetime string
        
    Example:
        >>> utcnow_iso()
        '2025-01-29T21:30:45.123456+00:00'
    """
    return utcnow().isoformat()


def utcnow_timestamp() -> float:
    """
    Return timezone-aware UTC datetime as timestamp
    
    Useful for performance measurements and database storage.
    
    Returns:
        float: UTC timestamp (seconds since epoch)
        
    Example:
        >>> utcnow_timestamp()
        1738187445.123456
    """
    return utcnow().timestamp()


# Backward compatibility alias (for migration period)
# TODO: Remove after all imports are updated
get_utc_now = utcnow
