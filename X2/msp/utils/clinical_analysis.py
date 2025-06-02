# msp/utils/clinical_analysis.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# NHS Triage Risk Levels
NHS_RISK_LEVELS = {
    "IMMEDIATE": {"score": 1.0, "color": "#F44336", "action": "999 Emergency"},
    "URGENT": {"score": 0.8, "color": "#FF9800", "action": "111 Urgent"},
    "STANDARD": {"score": 0.6, "color": "#FFC107", "action": "111 Standard"},
    "ROUTINE": {"score": 0.4, "color": "#4CAF50", "action": "GP Routine"},
    "SELF_CARE": {"score": 0.2, "color": "#8BC34A", "action": "Self Care"}
}

# Clinical Keywords for Flagging
CLINICAL_RED_FLAGS = {
    "cardiac": ["chest pain", "crushing pain", "heart attack", "cardiac arrest", "no pulse"],
    "respiratory": ["can't breathe", "breathless", "wheezing", "silent chest", "cyanosis"],
    "neurological": ["stroke", "facial drooping", "slurred speech", "weakness", "seizure"],
    "sepsis": ["fever", "confusion", "rapid pulse", "low blood pressure"],
    "trauma": ["bleeding", "fracture", "unconscious", "head injury"],
    "psychiatric": ["suicide", "self harm", "depression", "crisis"]
}

NICE_PROTOCOL_MAPPING = {
    "chest_pain": "NICE CG95",
    "breathlessness": "NICE CG191", 
    "abdominal_pain": "NICE CG141",
    "headache": "NICE CG150",
    "back_pain": "NICE NG59",
    "mental_health": "NICE CG90",
    "stroke": "NICE CG68",
    "cardiac_arrest": "NICE CG167",
    "anaphylaxis": "NICE CG134"
}



def calculate_urgency_score(chat_messages: List[Dict], patient_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Calculate urgency score based on chat content and patient data"""
    
    score = 0.0
    flagged_phrases = []
    risk_factors = []
    
    # Analyze chat messages for red flags
    for message in chat_messages:
        if message.get("role") == "user":
            content = message.get("content", "").lower()
            
            for category, keywords in CLINICAL_RED_FLAGS.items():
                for keyword in keywords:
                    if keyword in content:
                        score += 0.2
                        flagged_phrases.append({
                            "phrase": keyword,
                            "category": category,
                            "severity": "high" if score > 0.6 else "medium",
                            "message": content[:100] + "..." if len(content) > 100 else content,
                            "timestamp": message.get("timestamp", datetime.now().isoformat())
                        })
    
    # Apply patient data modifiers
    if patient_data:
        age_value = patient_data.get("age")
        conditions = patient_data.get("conditions", [])
        
        # FIXED: Proper None handling for age comparison
        if age_value is not None:
            try:
                age = int(age_value)
                # FIXED: Now safe to compare since age is guaranteed to be an int
                if age > 65 or age < 5:
                    score += 0.1
                    risk_factors.append("Age-related risk factor")
            except (ValueError, TypeError):
                # Skip age processing if invalid
                pass
        
        # Process chronic conditions safely
        high_risk_conditions = ["diabetes", "copd", "heart disease", "cancer"]
        for condition in conditions:
            if any(risk_cond in str(condition).lower() for risk_cond in high_risk_conditions):
                score += 0.15
                risk_factors.append(f"High-risk condition: {condition}")
    
    # Determine risk level
    risk_level = "SELF_CARE"
    for level, data in NHS_RISK_LEVELS.items():
        if score >= data["score"]:
            risk_level = level
            break
    
    return {
        "urgency_score": min(score, 1.0),
        "risk_level": risk_level,
        "risk_color": NHS_RISK_LEVELS[risk_level]["color"],
        "recommended_action": NHS_RISK_LEVELS[risk_level]["action"],
        "flagged_phrases": flagged_phrases,
        "risk_factors": risk_factors,
        "analysis_timestamp": datetime.now().isoformat()
    }

def extract_clinical_entities(text: str) -> List[Dict[str, Any]]:
    """Extract clinical entities from text using pattern matching"""
    
    entities = []
    
    # Pain scores (0-10)
    pain_pattern = r'pain.*?(\d+(?:/10|out of 10)?)'
    pain_matches = re.finditer(pain_pattern, text.lower())
    for match in pain_matches:
        entities.append({
            "type": "pain_score",
            "value": match.group(1),
            "context": match.group(0),
            "position": match.span()
        })
    
    # Time indicators
    time_pattern = r'(since|for|about|around)\s+(\d+)\s+(minutes?|hours?|days?|weeks?)'
    time_matches = re.finditer(time_pattern, text.lower())
    for match in time_matches:
        entities.append({
            "type": "duration",
            "value": f"{match.group(2)} {match.group(3)}",
            "context": match.group(0),
            "position": match.span()
        })
    
    return entities

def match_nice_protocol(symptoms: List[str]) -> Optional[str]:
    """Match symptoms to appropriate NICE protocol"""
    
    symptom_text = " ".join(symptoms).lower()
    
    for condition, protocol in NICE_PROTOCOL_MAPPING.items():
        condition_keywords = condition.replace("_", " ")
        if condition_keywords in symptom_text:
            return protocol
    
    return None

def generate_clinical_summary(chat_messages: List[Dict], urgency_data: Dict) -> Dict[str, Any]:
    """Generate clinical summary from chat and urgency analysis"""
    
    # Extract key symptoms
    symptoms = []
    timeline = []
    
    for message in chat_messages:
        if message.get("role") == "user":
            content = message["content"]
            entities = extract_clinical_entities(content)
            
            for entity in entities:
                if entity["type"] == "pain_score":
                    symptoms.append(f"Pain severity: {entity['value']}")
                elif entity["type"] == "duration":
                    timeline.append(f"Duration: {entity['value']}")
    
    # Suggested NICE protocol
    nice_protocol = match_nice_protocol(symptoms)
    
    return {
        "primary_symptoms": symptoms[:5],  # Top 5 symptoms
        "timeline": timeline,
        "nice_protocol": nice_protocol,
        "urgency_level": urgency_data["risk_level"],
        "recommendation": urgency_data["recommended_action"],
        "summary_generated": datetime.now().isoformat()
    }
