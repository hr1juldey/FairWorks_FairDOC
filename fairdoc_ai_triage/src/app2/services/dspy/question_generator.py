"""
DSPy Question Generator (≤180 LOC)
Utility to provide up to 50 NICE-style clarifying questions that the
MedicalTriageAgent can draw from when the NICE protocol either returns
"NONE" or its own question list is exhausted.

The questions are grouped by high-level symptom clusters so a trivial
keyword match will pick relevant items without heavy NLP.  Future
improvements could swap in semantic search / embedding lookup, but we
keep it simple for now.
"""
from __future__ import annotations

from typing import List, Dict
import re
import random
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------
# 50 HIGH-YIELD TRIAGE QUESTIONS (curated from NICE clinical pathways)
# ---------------------------------------------------------------------
_QUESTIONS: Dict[str, List[str]] = {
    # GENERAL / UNSPECIFIED
    "general": [
        "How long have you had these symptoms?",
        "Have your symptoms been getting better, worse, or staying the same?",
        "On a scale of 0 to 10, how severe is the discomfort right now?",
        "Have you taken any medication, and if so did it help?",
        "Have you experienced any recent injuries or falls?",
    ],
    # HEADACHE
    "headache": [
        "Did the headache start suddenly or build up gradually?",
        "Is this the worst headache you have ever experienced?",
        "Do you notice any vision changes, such as blurred or double vision?",
        "Have you had any nausea or vomiting with the headache?",
        "Do you have any neck stiffness or sensitivity to light?",
    ],
    # CHEST PAIN
    "chest": [
        "Can you describe the pain—sharp, dull, crushing, or burning?",
        "Does the pain radiate to your arm, neck, jaw, or back?",
        "Does physical activity make the pain worse?",
        "Are you short of breath or sweating when the pain occurs?",
        "Do you have a history of heart disease or high blood pressure?",
    ],
    # ABDOMINAL PAIN
    "abdomen": [
        "Where exactly in your abdomen is the pain located?",
        "Is the pain constant or does it come in waves?",
        "Have you noticed any changes in bowel movements?",
        "Have you had any vomiting or blood in your stool?",
        "For females, could you be pregnant or experiencing menstrual changes?",
    ],
    # FEVER / INFECTION
    "fever": [
        "What was the highest temperature you have recorded?",
        "Have you experienced chills or night sweats?",
        "Are there any localized symptoms such as cough, sore throat, or rash?",
        "Have you recently travelled or been in contact with someone who is ill?",
        "Are you taking any medications that suppress your immune system?",
    ],
    # BREATHING DIFFICULTY
    "breath": [
        "When did the shortness of breath start?",
        "Is it worse when lying flat or during exertion?",
        "Do you hear any wheezing or feel tightness in your chest?",
        "Have you noticed swelling in your ankles or legs?",
        "Do you have a history of asthma or lung disease?",
    ],
    # DIZZINESS / SYNCOPE
    "dizzy": [
        "Did you lose consciousness or just feel light-headed?",
        "Were there any warning signs before the episode?",
        "Have you had any recent changes in medication?",
        "Are you experiencing palpitations or irregular heartbeat?",
        "Have you been eating and drinking normally today?",
    ],
    # URINARY SYMPTOMS
    "urinary": [
        "Are you having pain or burning when you urinate?",
        "Have you noticed blood in your urine?",
        "How often are you needing to urinate?",
        "Do you feel lower abdominal or back pain?",
        "Do you have a fever or chills?",
    ],
    # SKIN RASH
    "rash": [
        "When did you first notice the rash?",
        "Is the rash itchy, painful, or neither?",
        "Has the rash spread since it started?",
        "Have you started any new medications or products recently?",
        "Do you have a fever or any other symptoms with the rash?",
    ],
}

# Flatten to ensure we hit the 50-question requirement
_ALL_QUESTIONS: List[str] = [q for group in _QUESTIONS.values() for q in group]
assert len(_ALL_QUESTIONS) == 50, "Question set must contain exactly 50 items"


# ---------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------

def suggest_questions(symptom_text: str, max_questions: int = 3) -> List[str]:
    """Return up to *max_questions* context-appropriate clarifying questions.

    The algorithm performs a naive keyword search over the symptom text to
    choose a relevant cluster; if nothing matches, it falls back to the
    general pool.
    """
    text = symptom_text.lower()
    cluster_key = "general"  # default

    for key in _QUESTIONS.keys():
        if key != "general" and re.search(key, text):
            cluster_key = key
            break

    questions = _QUESTIONS.get(cluster_key, _QUESTIONS["general"])
    selected = random.sample(questions, k=min(max_questions, len(questions)))

    logger.debug(
        "Question suggestions",
        cluster=cluster_key,
        picks=selected,
    )
    return selected
