# msp/utils/transcript_helpers.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import re
from typing import Dict, List, Any, Tuple

def group_phrases_by_category(flagged_phrases: List[Dict]) -> Dict[str, List[Dict]]:
    """Group flagged phrases by clinical category"""
    categories = {}
    for phrase in flagged_phrases:
        category = phrase.get("category", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(phrase)
    return categories

def check_message_contains_flags(content: str, flagged_phrases: List[str]) -> bool:
    """Check if message content contains any flagged phrases"""
    return any(phrase.lower() in content.lower() for phrase in flagged_phrases)

def get_role_style_config(role: str) -> Dict[str, str]:
    """Get styling configuration for different user roles"""
    role_styles = {
        "user": {"icon": "👤", "bg": "white", "name": "Patient"},
        "assistant": {"icon": "🤖", "bg": "#E3F2FD", "name": "AI Assistant"},  # Using bg_accent value
        "system": {"icon": "⚙️", "bg": "#F5F7FA", "name": "System"}  # Using light_grey value
    }
    return role_styles.get(role, {"icon": "💬", "bg": "white", "name": "Unknown"})

def get_severity_color_mapping() -> Dict[str, str]:
    """Get color mapping for different severity levels"""
    return {
        "high": "#F44336",      # error color
        "medium": "#FFC107",    # warning color  
        "low": "#4CAF50"        # success color
    }

def get_category_icon_mapping() -> Dict[str, str]:
    """Get icon mapping for different clinical categories"""
    return {
        "cardiac": "❤️",
        "respiratory": "🫁", 
        "neurological": "🧠",
        "psychiatric": "🧘‍♀️",
        "sepsis": "🦠",
        "trauma": "🩹"
    }

def highlight_flagged_phrases(content: str, flagged_phrases: List[str]) -> str:
    """Highlight flagged phrases in content with case-insensitive matching"""
    highlighted_content = content
    
    for phrase in flagged_phrases:
        if phrase.lower() in content.lower():
            # Find the actual phrase in the content (preserving case)
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            highlighted_content = pattern.sub(f"🚨 {phrase} 🚨", highlighted_content)
    
    return highlighted_content

def extract_flagged_phrase_list(flagged_phrases: List[Dict]) -> List[str]:
    """Extract just the phrase strings from flagged phrase objects"""
    return [phrase["phrase"] for phrase in flagged_phrases]

def format_risk_level_display(risk_level: str) -> str:
    """Format risk level for display (replace underscores with spaces and title case)"""
    return risk_level.replace("_", " ").title()

def validate_message_data(message: Dict[str, Any]) -> Tuple[str, str, str]:
    """Validate and extract message data, returning safe defaults"""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")
    return role, content, timestamp

def validate_quote_data(quote: Dict[str, Any]) -> Dict[str, str]:
    """Validate and extract quote data with safe defaults"""
    return {
        "quote": quote.get("quote", ""),
        "flagged_phrase": quote.get("flagged_phrase", ""),
        "clinical_category": quote.get("clinical_category", "Unknown"),
        "severity": quote.get("severity", "unknown"),
        "inference": quote.get("inference", "")
    }

def calculate_citation_number(citation_map: Dict[str, int], flagged_phrase: str) -> int:
    """Calculate citation number for a flagged phrase, with fallback"""
    return citation_map.get(flagged_phrase, 1)

def create_conversation_statistics(chat_history: List[Dict], urgency_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Create conversation statistics for overview display"""
    from utils.report_helpers import calculate_chat_duration
    
    return [
        ("Total Messages", str(len(chat_history))),
        ("Duration", calculate_chat_duration(chat_history)),
        ("Clinical Flags", str(len(urgency_data.get("flagged_phrases", [])))),
        ("Risk Level", format_risk_level_display(urgency_data.get("risk_level", "Unknown")))
    ]

def format_category_assessment_text(category: str, phrases_count: int) -> str:
    """Format category assessment description text"""
    return f"Identified {phrases_count} relevant clinical indicators in this {category}:"

def get_default_icon_for_category(category: str) -> str:
    """Get default icon for unknown categories"""
    category_icons = get_category_icon_mapping()
    return category_icons.get(category, "⚕️")
