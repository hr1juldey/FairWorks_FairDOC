# msp/utils/report_helpers.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import json
import base64
from typing import Dict, List, Any
from datetime import datetime

def format_risk_level(risk_level: str) -> str:
    """Format risk level for display"""
    return risk_level.replace("_", " ").title()

def get_risk_badge_style(risk_color: str) -> str:
    """Get CSS style for risk badge"""
    return f"background: {risk_color}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 600;"

def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp for display"""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        return iso_timestamp

def generate_report_id() -> str:
    """Generate unique report ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"RPT_{timestamp}"

def prepare_download_data(report_data: Dict) -> str:
    """Prepare report data for download as JSON"""
    download_data = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "format": "Fairdoc_NHS_Report"
        },
        "report_content": report_data
    }
    return json.dumps(download_data, indent=2)

def extract_key_quotes(chat_messages: List[Dict], flagged_phrases: List[Dict]) -> List[Dict]:
    """Extract key quotes with clinical significance"""
    
    quotes = []
    
    for phrase_data in flagged_phrases:
        phrase = phrase_data["phrase"]
        category = phrase_data["category"]
        # FIXED: Use the message_text if available, otherwise search through chat messages
        message_text = phrase_data.get("message", "")
        
        if message_text:
            # Use the pre-existing message text from flagged phrase data
            quotes.append({
                "quote": message_text,
                "flagged_phrase": phrase,
                "clinical_category": category,
                "severity": phrase_data.get("severity", "medium"),
                "timestamp": phrase_data.get("timestamp", ""),
                "inference": generate_clinical_inference(phrase, category)
            })
        else:
            # Fallback: Find the full message containing this phrase in chat history
            for msg in chat_messages:
                if msg.get("role") == "user" and phrase in msg.get("content", "").lower():
                    quotes.append({
                        "quote": msg["content"],
                        "flagged_phrase": phrase,
                        "clinical_category": category,
                        "severity": phrase_data.get("severity", "medium"),
                        "timestamp": msg.get("timestamp", ""),
                        "inference": generate_clinical_inference(phrase, category)
                    })
                    break
    
    return quotes

def generate_clinical_inference(phrase: str, category: str) -> str:
    """Generate clinical inference for flagged phrases"""
    
    inferences = {
        "cardiac": f"'{phrase}' indicates possible cardiac emergency requiring immediate assessment",
        "respiratory": f"'{phrase}' suggests respiratory distress requiring urgent evaluation", 
        "neurological": f"'{phrase}' may indicate neurological compromise requiring immediate attention",
        "sepsis": f"'{phrase}' could suggest systemic infection requiring urgent medical care",
        "trauma": f"'{phrase}' indicates potential trauma requiring emergency assessment",
        "psychiatric": f"'{phrase}' suggests mental health crisis requiring immediate support"
    }
    
    return inferences.get(category, f"'{phrase}' requires clinical assessment")

def calculate_chat_duration(chat_messages: List[Dict]) -> str:
    """Calculate total chat duration"""
    
    if len(chat_messages) < 2:
        return "< 1 minute"
    
    try:
        first_msg = chat_messages[0]
        last_msg = chat_messages[-1]
        
        # Assuming timestamps are available
        start_time = datetime.fromisoformat(first_msg.get("timestamp", ""))
        end_time = datetime.fromisoformat(last_msg.get("timestamp", ""))
        
        duration = end_time - start_time
        minutes = duration.total_seconds() / 60
        
        if minutes < 1:
            return "< 1 minute"
        elif minutes < 60:
            return f"{int(minutes)} minutes"
        else:
            hours = minutes / 60
            return f"{hours:.1f} hours"
            
    except Exception:
        return "Duration unknown"

def create_citation_map(quotes: List[Dict]) -> Dict[str, int]:
    """Create citation mapping for quotes (like Perplexity)"""
    
    citation_map = {}
    for i, quote in enumerate(quotes, 1):
        key = quote["flagged_phrase"]
        citation_map[key] = i
    
    return citation_map

def validate_report_data(report_data: Dict) -> bool:
    """Validate report data structure"""
    
    required_fields = ["report_metadata", "report_content"]
    return all(field in report_data for field in required_fields)

def sanitize_clinical_text(text: str) -> str:
    """Sanitize clinical text for safe display"""
    
    if not text:
        return ""
    
    # Remove potential HTML/script tags for security
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Limit length to prevent UI issues
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()

def get_severity_color(severity: str) -> str:
    """Get color code for severity level"""
    
    severity_colors = {
        "high": "#F44336",      # Red
        "medium": "#FF9800",    # Orange  
        "low": "#4CAF50",       # Green
        "critical": "#D32F2F",  # Dark red
        "info": "#2196F3"       # Blue
    }
    
    return severity_colors.get(severity.lower(), "#757575")  # Default grey
