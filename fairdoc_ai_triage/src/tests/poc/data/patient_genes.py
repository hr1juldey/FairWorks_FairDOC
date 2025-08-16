"""
Patient Gene Expression Data Bank

Contains 20-factor patient profiles and epistatic interaction matrices
for Indian patient behavior modeling. No line limit for data files.
"""

# 20-Factor Gene Expression Profiles
PATIENT_GENE_EXPRESSIONS = {
    "urban_educated_male": {
        "age": 35,
        "gender": "male", 
        "education_level": "graduate",
        "occupation": "service",
        "marital_status": "married",
        "income_quintile": 4,
        "caste_category": "General",
        "residence_type": "urban",
        "city_tier": "tier1",
        "health_insurance": 1,
        "state_region": "west",
        "ethnicity": "marathi",
        "distance_to_healthcare": 5.2,
        "smartphone_access": 1,
        "internet_literacy": 3,
        "tobacco_use": "never",
        "alcohol_consumption": "occasional", 
        "physical_activity": 2,
        "time_of_day": "evening",
        "season": "monsoon"
    },
    
    "rural_traditional_female": {
        "age": 45,
        "gender": "female",
        "education_level": "primary", 
        "occupation": "homemaker",
        "marital_status": "married",
        "income_quintile": 2,
        "caste_category": "OBC",
        "residence_type": "rural",
        "city_tier": "tier3",
        "health_insurance": 0,
        "state_region": "north",
        "ethnicity": "hindi",
        "distance_to_healthcare": 25.0,
        "smartphone_access": 0,
        "internet_literacy": 0,
        "tobacco_use": "never",
        "alcohol_consumption": "never",
        "physical_activity": 3,
        "time_of_day": "morning", 
        "season": "winter"
    },
    
    "tech_professional_bengali": {
        "age": 28,
        "gender": "male",
        "education_level": "postgraduate",
        "occupation": "professional",
        "marital_status": "single", 
        "income_quintile": 5,
        "caste_category": "General",
        "residence_type": "urban",
        "city_tier": "metro",
        "health_insurance": 1,
        "state_region": "east", 
        "ethnicity": "bengali",
        "distance_to_healthcare": 2.1,
        "smartphone_access": 1,
        "internet_literacy": 3,
        "tobacco_use": "former",
        "alcohol_consumption": "occasional",
        "physical_activity": 1,
        "time_of_day": "night",
        "season": "summer"
    }
}

# Epistatic Interaction Matrix (γᵢⱼ coefficients)
EPISTATIC_INTERACTIONS = {
    ("education_level", "income_quintile"): 0.25,
    ("smartphone_access", "internet_literacy"): 0.30,
    ("state_region", "ethnicity"): 0.20,
    ("tobacco_use", "alcohol_consumption"): 0.15,
    ("age", "gender"): 0.18,
    ("residence_type", "city_tier"): 0.22,
    ("health_insurance", "income_quintile"): 0.35,
    ("caste_category", "occupation"): 0.12,
    ("distance_to_healthcare", "smartphone_access"): -0.25,
    ("time_of_day", "season"): 0.10
}

# Main Effect Coefficients (βᵢ values)
MAIN_EFFECT_COEFFICIENTS = {
    "age": 0.15,
    "gender": 0.12, 
    "education_level": 0.28,
    "income_quintile": 0.22,
    "smartphone_access": 0.18,
    "state_region": 0.14,
    "health_insurance": 0.20,
    "distance_to_healthcare": -0.16,
    "ethnicity": 0.13,
    "city_tier": 0.17
}

# Regional Behavior Patterns from Markdown
REGIONAL_PATTERNS = {
    "north": {
        "family_consultation_weight": 0.8,
        "average_delay_minutes": 45,
        "language_preference": "hindi",
        "authority_respect": 0.9,
        "decision_making": "hierarchical"
    },
    "south": {
        "family_consultation_weight": 0.6, 
        "average_delay_minutes": 30,
        "language_preference": "english",
        "analytical_approach": 0.8,
        "tech_adoption": 0.9
    },
    "west": {
        "family_consultation_weight": 0.5,
        "average_delay_minutes": 20, 
        "language_preference": "english",
        "efficiency_focus": 0.9,
        "business_mindset": 0.8
    },
    "east": {
        "family_consultation_weight": 0.7,
        "average_delay_minutes": 35,
        "language_preference": "bengali",
        "intellectual_discourse": 0.8,
        "detailed_discussion": 0.9
    },
    "northeast": {
        "family_consultation_weight": 0.75,
        "average_delay_minutes": 40,
        "language_preference": "english",
        "community_consensus": 0.9,
        "tribal_protocols": 0.7
    }
}

# Wealth Psychology Classifications
WEALTH_PSYCHOLOGY_CATEGORIES = {
    "survival_rich": {
        "percentage": 15,
        "delay_coefficient": -0.4,
        "government_hospital_preference": 0.6,
        "price_sensitivity": -0.5,
        "family_consultation_threshold": 5000
    },
    "aspirational_poor": {
        "percentage": 25,
        "branded_hospital_preference": 0.7,
        "emi_decisions": 0.6,
        "planning_delay": -0.3,
        "social_validation": 0.4
    },
    "hidden_wealth": {
        "percentage": 8,
        "decision_delay": -0.2,
        "traditional_first": 0.5,
        "cash_preference": 0.8,
        "wealth_protection": 0.6
    },
    "generational_wealth": {
        "percentage": 5,
        "specialist_access": 0.9,
        "proactive_health": 0.7,
        "multiple_opinions": 0.6,
        "hybrid_payment": 0.4
    }
}

# Disease Risk Multipliers by Region (RR values from markdown)
DISEASE_RISK_MULTIPLIERS = {
    "north": {
        "cardiovascular_rr": 1.4,
        "stroke_rr": 1.3,
        "diabetes_rr": 1.2,
        "infectious_rr": 1.1
    },
    "south": {
        "diabetes_rr": 1.5,
        "hypertension_rr": 1.4,
        "metabolic_rr": 1.5,
        "kidney_rr": 1.2
    },
    "west": {
        "lifestyle_rr": 1.3,
        "stress_cvd_rr": 1.4,
        "obesity_rr": 1.3,
        "infectious_rr": 0.9
    },
    "east": {
        "cardiovascular_rr": 0.9,
        "infectious_rr": 1.2,
        "nutritional_rr": 1.3,
        "stroke_rr": 1.1
    },
    "northeast": {
        "infectious_rr": 1.4,
        "cardiovascular_rr": 0.8,
        "access_challenges": 0.6
    }
}
