# File: Fairdoc\frontend\msp\mock_backend\nhs_data.py
"""
NHS NICE Protocol Questions and Mock Data
Simulates real NHS 111 triage protocols
"""

# NHS NICE Protocol Questions for Chest Pain
NHS_CHEST_PAIN_QUESTIONS = [
    {
        "id": 1,
        "category": "demographics",
        "question": "What is your age?",
        "type": "number",
        "required": True,
        "red_flag": False
    },
    {
        "id": 2,
        "category": "demographics",
        "question": "What is your gender?",
        "type": "choice",
        "options": ["Male", "Female", "Other", "Prefer not to say"],
        "required": True,
        "red_flag": False
    },
    {
        "id": 3,
        "category": "chief_complaint",
        "question": "Can you describe your chest pain? How severe is it on a scale of 1-10?",
        "type": "text_with_scale",
        "required": True,
        "red_flag": True,
        "red_flag_threshold": 8
    },
    {
        "id": 4,
        "category": "symptoms",
        "question": "Are you experiencing shortness of breath or difficulty breathing?",
        "type": "yes_no",
        "required": True,
        "red_flag": True
    },
    {
        "id": 5,
        "category": "symptoms",
        "question": "Are you coughing up blood or pink frothy sputum?",
        "type": "yes_no",
        "required": True,
        "red_flag": True
    },
    {
        "id": 6,
        "category": "symptoms",
        "question": "Do you have pain spreading to your jaw, neck, or arms?",
        "type": "yes_no",
        "required": True,
        "red_flag": True
    },
    {
        "id": 7,
        "category": "history",
        "question": "Do you have a history of heart problems, high blood pressure, or diabetes?",
        "type": "text",
        "required": False,
        "red_flag": False
    },
    {
        "id": 8,
        "category": "medication",
        "question": "Are you currently taking any medications?",
        "type": "text",
        "required": False,
        "red_flag": False
    },
    {
        "id": 9,
        "category": "recent_activity",
        "question": "When did this pain start? What were you doing when it began?",
        "type": "text",
        "required": True,
        "red_flag": False
    },
    {
        "id": 10,
        "category": "file_upload",
        "question": "Do you have any recent medical reports, ECGs, or chest X-rays you'd like to upload? (Optional)",
        "type": "file_upload",
        "required": False,
        "red_flag": False,
        "accepted_types": ["pdf", "jpg", "png", "dcm"]
    }
]

# Mock emergency responses
EMERGENCY_RESPONSES = {
    "999": [
        "Call 999 immediately - Suspected heart attack",
        "Call 999 immediately - Severe chest pain with breathing difficulties",
        "Call 999 immediately - Signs of pulmonary embolism"
    ],
    "111": [
        "Contact NHS 111 for urgent assessment",
        "See a doctor within 2 hours",
        "Visit A&E within 4 hours"
    ],
    "gp": [
        "Book appointment with GP within 48 hours",
        "Monitor symptoms and contact GP if worsening",
        "Consider pharmacy consultation first"
    ]
}

# Mock AI analysis responses
AI_ANALYSIS_TEMPLATES = {
    "low_risk": {
        "urgency": 0.2,
        "importance": 0.3,
        "recommendation": "GP consultation recommended",
        "reasoning": "Mild symptoms with no immediate red flags"
    },
    "medium_risk": {
        "urgency": 0.6,
        "importance": 0.7,
        "recommendation": "NHS 111 urgent assessment",
        "reasoning": "Concerning symptoms requiring prompt medical evaluation"
    },
    "high_risk": {
        "urgency": 0.9,
        "importance": 0.95,
        "recommendation": "999 emergency response",
        "reasoning": "Critical symptoms suggesting life-threatening condition"
    }
}

def calculate_risk_score(answers: dict) -> dict:
    """Calculate risk score based on answers"""
    risk_factors = 0
    red_flags = 0
    
    # Check for red flag answers
    if answers.get("pain_severity", 0) >= 8:
        red_flags += 1
        risk_factors += 0
    if answers.get("shortness_of_breath") == "yes":
        red_flags += 1
        risk_factors += 1
    if answers.get("coughing_blood") == "yes":
        red_flags += 2
        risk_factors += 4
    if answers.get("pain_spreading") == "yes":
        red_flags += 1
        risk_factors += 2
    
    # Calculate scores
    if red_flags >= 2:
        return AI_ANALYSIS_TEMPLATES["high_risk"]
    elif red_flags == 1:
        return AI_ANALYSIS_TEMPLATES["medium_risk"]
    else:
        return AI_ANALYSIS_TEMPLATES["low_risk"]
