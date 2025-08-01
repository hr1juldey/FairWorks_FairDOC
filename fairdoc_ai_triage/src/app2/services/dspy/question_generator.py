"""
DSPy Medical Question Generator with Examples and Thinking
Uses DSPy examples and reasoning for NICE-compliant medical questions
Enhanced with DeepSeek-R1 thinking for emergency protocols
"""

import dspy
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class MedicalQuestionSignature(dspy.Signature):
    """Generate NICE-compliant medical questions with reasoning"""
    symptoms: str = dspy.InputField(desc="Patient's current symptoms")
    conversation_history: str = dspy.InputField(desc="Previous conversation context")
    nice_protocols: str = dspy.InputField(desc="Relevant NICE emergency protocols")
    emergency_indicators: str = dspy.InputField(desc="Red flag symptoms detected")
    
    # Thinking outputs for DeepSeek-R1
    medical_reasoning: str = dspy.OutputField(desc="Clinical reasoning for question priority")
    emergency_assessment: str = dspy.OutputField(desc="Emergency risk assessment reasoning")
    
    # Question outputs
    priority_questions: str = dspy.OutputField(desc="High-priority questions (| separated)")
    emergency_questions: str = dspy.OutputField(desc="Emergency-specific questions (| separated)")

class QuestionPrioritizationSignature(dspy.Signature):
    """Prioritize questions based on medical urgency"""
    available_questions: str = dspy.InputField(desc="Pool of questions (| separated)")
    symptoms: str = dspy.InputField(desc="Patient symptoms")
    urgency_level: str = dspy.InputField(desc="Emergency urgency: low|medium|high|critical")
    
    selected_questions: str = dspy.OutputField(desc="Prioritized questions (| separated)")
    prioritization_reasoning: str = dspy.OutputField(desc="Medical prioritization logic")

class MedicalQuestionModule(dspy.Module):
    """DSPy module for NICE-compliant medical question generation"""
    
    def __init__(self):
        super().__init__()
        # Use ChainOfThoughtWithHint for guided medical reasoning
        self.question_generator = dspy.ChainOfThoughtWithHint(MedicalQuestionSignature)
        self.question_prioritizer = dspy.ChainOfThought(QuestionPrioritizationSignature)
        
        # Initialize with DSPy examples for few-shot learning
        self.medical_examples = self._create_medical_examples()
        self._setup_few_shot_optimizer()
    
    def _create_medical_examples(self) -> List[dspy.Example]:
        """Create DSPy training examples for medical question generation"""
        examples = [
            dspy.Example(
                symptoms="severe chest pain radiating to left arm",
                conversation_history="Initial complaint: chest discomfort",
                nice_protocols="CG95: Chest Pain - Emergency criteria: crushing pain >20min with radiation",
                emergency_indicators="chest_pain_radiation, severe_intensity",
                medical_reasoning="Patient presents with classic cardiac emergency symptoms requiring immediate assessment",
                emergency_assessment="High risk for acute coronary syndrome - requires emergency questions",
                priority_questions="Are you sweating or feeling nauseous? | Do you have difficulty breathing? | Is this the worst chest pain you've ever experienced?",
                emergency_questions="Call 999 immediately - Do you have crushing chest pain? | Are you experiencing severe shortness of breath?"
            ).with_inputs('symptoms', 'conversation_history', 'nice_protocols', 'emergency_indicators'),
            
            dspy.Example(
                symptoms="sudden severe headache worst ever experienced",
                conversation_history="Patient reports sudden onset headache",
                nice_protocols="NG127: Headache - Emergency: thunderclap headache, neck stiffness",
                emergency_indicators="thunderclap_headache, sudden_onset",
                medical_reasoning="Thunderclap headache pattern suggests possible subarachnoid hemorrhage",
                emergency_assessment="Critical emergency requiring immediate neurological assessment",
                priority_questions="Did the headache start suddenly like a thunderclap? | Do you have neck stiffness? | Any vision changes?",
                emergency_questions="This requires emergency assessment - Are you experiencing neck stiffness? | Any loss of consciousness?"
            ).with_inputs('symptoms', 'conversation_history', 'nice_protocols', 'emergency_indicators'),
            
            dspy.Example(
                symptoms="mild tension headache, stress-related",
                conversation_history="Patient reports work stress, similar headaches before",
                nice_protocols="NG127: Headache - Routine assessment for tension-type",
                emergency_indicators="none_detected",
                medical_reasoning="Pattern consistent with tension-type headache, no red flags present",
                emergency_assessment="Low risk scenario - routine assessment questions appropriate",
                priority_questions="How long have you had this headache? | On a scale of 1-10, how severe is the pain? | Have you taken any pain relief?",
                emergency_questions=""
            ).with_inputs('symptoms', 'conversation_history', 'nice_protocols', 'emergency_indicators'),
            
            dspy.Example(
                symptoms="shortness of breath, ankle swelling",
                conversation_history="Progressive breathlessness over 2 weeks",
                nice_protocols="NG106: Heart Failure - Assessment of fluid retention and dyspnea",
                emergency_indicators="progressive_dyspnea, fluid_retention",
                medical_reasoning="Combination suggests possible heart failure requiring systematic assessment",
                emergency_assessment="Moderate priority - cardiac assessment needed",
                priority_questions="Is breathing worse when lying flat? | Any chest tightness? | Previous heart problems?",
                emergency_questions="Are you struggling to breathe at rest? | Any severe chest pain?"
            ).with_inputs('symptoms', 'conversation_history', 'nice_protocols', 'emergency_indicators'),
            
            dspy.Example(
                symptoms="abdominal pain, right lower quadrant",
                conversation_history="Pain started around navel, moved to right side",
                nice_protocols="CG141: Appendicitis - McBurney's point tenderness, migration pattern",
                emergency_indicators="pain_migration, mcburney_point",
                medical_reasoning="Classic appendicitis presentation with pain migration pattern",
                emergency_assessment="High priority surgical emergency assessment required",
                priority_questions="Did the pain start around your belly button? | Any nausea or vomiting? | Pain worse with movement?",
                emergency_questions="Rate pain severity 1-10? | Any fever with the pain? | Unable to walk due to pain?"
            ).with_inputs('symptoms', 'conversation_history', 'nice_protocols', 'emergency_indicators')
        ]
        return examples
    
    def _setup_few_shot_optimizer(self):
        """Setup few-shot learning with medical examples"""
        try:
            from dspy import BootstrapFewShot
            # Use bootstrap few-shot for medical question optimization
            self.optimizer = BootstrapFewShot(
                metric=self._medical_question_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=5
            )
            logger.info("✅ Few-shot optimizer configured with medical examples")
        except Exception as e:
            logger.warning("⚠️ Few-shot optimizer setup failed", error=str(e))
            self.optimizer = None
    
    def forward(self, symptoms, conversation_context="", nice_protocols="", max_questions=3):
        # Detect emergency indicators from symptoms
        emergency_indicators = self._detect_emergency_indicators(symptoms, nice_protocols)
        urgency_level = self._assess_urgency_level(emergency_indicators)
        
        # Generate questions with medical reasoning (thinking enabled)
        generation_result = self.question_generator(
            symptoms=symptoms,
            conversation_history=conversation_context,
            nice_protocols=nice_protocols,
            emergency_indicators=", ".join(emergency_indicators),
            hint=f"Apply NICE emergency protocols for {urgency_level} priority assessment"
        )
        
        # Combine generated questions with curated emergency questions
        all_questions = self._combine_with_emergency_bank(
            priority_qs=generation_result.priority_questions,
            emergency_qs=generation_result.emergency_questions,
            symptoms=symptoms,
            urgency=urgency_level
        )
        
        # Prioritize questions based on medical urgency
        prioritization_result = self.question_prioritizer(
            available_questions=" | ".join(all_questions),
            symptoms=symptoms,
            urgency_level=urgency_level
        )
        
        final_questions = prioritization_result.selected_questions.split(" | ")[:max_questions]
        
        return dspy.Prediction(
            questions=final_questions,
            medical_reasoning=generation_result.medical_reasoning,
            emergency_assessment=generation_result.emergency_assessment,
            urgency_level=urgency_level,
            emergency_indicators=emergency_indicators
        )
    
    def _detect_emergency_indicators(self, symptoms: str, protocols: str) -> List[str]:
        """Detect emergency red flags in symptoms"""
        symptoms_lower = symptoms.lower()
        emergency_patterns = {
            'crushing_chest_pain': ['crushing', 'squeezing', 'elephant on chest', 'heavy chest'],
            'chest_pain_radiation': ['radiating', 'spreading', 'arm pain', 'jaw pain', 'shoulder pain', 'back pain'],
            'thunderclap_headache': ['thunderclap', 'worst headache ever', 'sudden severe'],
            'severe_breathlessness': ['can\'t breathe', 'gasping', 'severe shortness', 'air hunger', 'laboured breathing'],
            'loss_consciousness': ['fainted', 'passed out', 'lost consciousness', 'unconscious', 'syncopal episode'],
            'severe_bleeding': ['heavy bleeding', 'blood loss', 'hemorrhage', 'uncontrollable bleeding', 'saturating dressing'],
            'pain_migration': ['pain moved', 'started around navel', 'migrated', 'belly button', 'moved to right side'],
            'neck_stiffness': ['neck stiff', 'can\'t bend neck', 'meningeal', 'pain bending neck'],
            'heart_failure_triad': ['shortness of breath', 'ankle swelling', 'fatigue', 'fluid retention', 'swollen legs', 'weight gain'],
            'appendicitis_classic': ['belly button', 'right side', 'mcburney', 'rebound tenderness', 'pain in lower right abdomen'],
            'stroke_symptoms_FAST': ['face droop', 'arm weakness', 'slurred speech', 'facial drooping', 'one side weak'],
            'sepsis_signs': ['high temperature', 'fever', 'shivering', 'chills', 'fast breathing', 'fast heart rate', 'confusion'],
            'anaphylaxis': ['swelling face', 'swollen tongue', 'wheezing', 'hives', 'rash', 'difficulty swallowing'],
            'pulmonary_embolism_signs': ['sudden chest pain', 'coughing blood', 'low oxygen', 'breathless', 'sharp pain on breathing'],
            'ectopic_pregnancy': ['shoulder tip pain', 'one-sided tummy pain', 'vaginal bleeding', 'spotting'],
            'diabetic_ketoacidosis': ['fruity breath', 'high blood sugar', 'frequent urination', 'vomiting', 'abdominal pain'],
            'severe_allergic_reaction': ['itchy skin', 'swollen throat', 'hives', 'breathing difficulty after sting'],
            'sudden_vision_loss': ['lost vision suddenly', 'blurry vision in one eye', 'double vision', 'seeing black spots'],
            'hypoglycaemic_coma': ['confusion', 'pale', 'sweaty', 'shaking', 'seizure', 'low blood sugar', 'lost consciousness'],
            'hyperglycaemic_hyperosmolar_state': ['severe dehydration', 'very high blood sugar', 'no ketones', 'coma'],
            'adrenal_crisis': ['severe weakness', 'hypotension', 'vomiting', 'abdominal pain', 'fever'],
            'thyroid_storm': ['high fever', 'tachycardia', 'palpitations', 'confusion', 'agitation'],
            'acute_angle_closure_glaucoma': ['severe eye pain', 'red eye', 'blurred vision', 'seeing halos'],
            'retinal_detachment': ['flashes of light', 'floaters', 'curtain coming down', 'lost peripheral vision'],
            'cauda_equina_syndrome': ['saddle anaesthesia', 'urinary retention', 'incontinence', 'back pain', 'numbness inner thighs'],
            'septic_arthritis': ['hot joint', 'swollen joint', 'painful joint', 'fever', 'can\'t move joint'],
            'compartment_syndrome': ['severe leg pain', 'pain out of proportion', 'tense muscle', 'numbness', 'tingling'],
            'non_accidental_injury': ['suspicious bruising', 'multiple fractures', 'injury inconsistent with story', 'cigarette burns'],
            'febrile_seizure': ['seizure with fever', 'shaking', 'high temperature', 'loss of consciousness'],
            'croup': ['barking cough', 'inspiratory stridor', 'hoarse voice', 'breathing difficulty'],
            'meningitis_rash': ['non-blanching rash', 'fever', 'neck stiffness', 'headache'],
            'necrotizing_fasciitis': ['severe pain', 'redness', 'swelling', 'blisters', 'skin death'],
            'gastrointestinal_bleeding': ['vomiting blood', 'coffee ground vomit', 'black tarry stools', 'melena'],
            'bowel_obstruction': ['severe stomach pain', 'colicky pain', 'bloated abdomen', 'can\'t pass gas'],
            'pancreatitis': ['severe abdominal pain', 'radiating to back', 'worse after eating', 'nausea', 'vomiting'],
            'ruptured_aortic_aneurysm': ['sudden severe abdominal pain', 'back pain', 'fainting', 'pulsating lump'],
            'acute_cholecystitis': ['severe upper right abdominal pain', 'radiating to shoulder', 'fever', 'nausea'],
            'acute_pyelonephritis': ['loin pain', 'back pain', 'fever', 'shivering', 'painful urination'],
            'renal_colic': ['severe loin pain', 'groin pain', 'agony', 'can\'t get comfortable'],
            'testicular_torsion': ['sudden severe testicular pain', 'swollen testicle', 'nausea', 'vomiting'],
            'postpartum_hemorrhage': ['heavy vaginal bleeding after birth', 'saturating pads', 'clots', 'dizziness'],
            'pre-eclampsia_eclampsia': ['headache', 'blurred vision', 'swelling face/hands', 'high blood pressure', 'seizure'],
            'placental_abruption': ['constant abdominal pain', 'dark red vaginal bleeding', 'firm uterus'],
            'uterine_rupture': ['sudden severe abdominal pain during labour', 'loss of contractions', 'fetal distress'],
            'cord_prolapse': ['feeling of something coming out', 'visible cord in vagina', 'fetal distress'],
            'burns_severe': ['blisters', 'redness', 'pain', 'loss of sensation', 'charred skin'],
            'head_trauma': ['confusion', 'headache', 'vomiting', 'loss of consciousness', 'amnesia'],
            'spinal_cord_injury': ['neck pain', 'back pain', 'numbness', 'paralysis', 'lost sensation'],
            'drug_overdose_opioid': ['pinpoint pupils', 'slow breathing', 'unresponsive', 'blue lips'],
            'drug_overdose_stimulant': ['agitation', 'fast heart rate', 'high temperature', 'seizure', 'chest pain'],
            'poisoning_general': ['vomiting', 'abdominal pain', 'confusion', 'drowsiness', 'unusual smell'],
            'carbon_monoxide_poisoning': ['headache', 'dizziness', 'nausea', 'cherry-red skin', 'multiple people affected'],
            'heat_stroke': ['high body temperature', 'hot dry skin', 'confusion', 'loss of consciousness'],
            'hypothermia': ['shivering', 'confusion', 'cold skin', 'slow breathing', 'unconsciousness'],
            'acute_psychosis': ['hallucinations', 'delusions', 'disorganized thoughts', 'agitation', 'incoherent speech'],
            'suicidal_ideation': ['talking about suicide', 'giving away possessions', 'feeling hopeless', 'making a plan'],
            'catatonia': ['immobility', 'mutism', 'waxy flexibility', 'staring', 'echolalia'],
            'acute_gout': ['sudden severe joint pain', 'red joint', 'swollen joint', 'big toe'],
            'septic_shock': ['low blood pressure', 'high fever', 'cold extremities', 'confusion', 'organ failure'],
            'toxic_shock_syndrome': ['high fever', 'rash', 'low blood pressure', 'vomiting', 'diarrhea'],
            'febrile_neutropenia': ['fever', 'low white cell count', 'chills', 'sore throat'],
            'hemolytic_uremic_syndrome': ['bloody diarrhea', 'low platelet count', 'kidney failure', 'tiredness'],
            'hypertensive_emergency': ['severe headache', 'blurred vision', 'chest pain', 'blood pressure >180/120'],
            'cardiogenic_shock': ['low blood pressure', 'fast heart rate', 'cold clammy skin', 'shortness of breath'],
            'dissecting_aortic_aneurysm': ['sudden tearing chest pain', 'radiating to back', 'different blood pressures in arms'],
            'pulmonary_oedema': ['severe breathlessness', 'coughing pink frothy sputum', 'sweating', 'anxiety'],
            'acute_urinary_retention': ['severe lower abdominal pain', 'unable to urinate', 'distended bladder'],
            'volvulus': ['sudden severe abdominal pain', 'vomiting', 'bloating', 'bloody stool'],
            'intussusception': ['abdominal pain', 'vomiting', 'jelly-like stool with blood'],
            'pneumothorax': ['sudden sharp chest pain', 'shortness of breath', 'collapsed lung', 'decreased breath sounds'],
            'tension_pneumothorax': ['sudden sharp chest pain', 'tracheal deviation', 'low blood pressure', 'neck vein distension'],
            'status_epilepticus': ['seizure lasting more than 5 minutes', 'multiple seizures without recovery', 'unconsciousness'],
            'acute_vertigo': ['sudden severe dizziness', 'spinning sensation', 'nausea', 'vomiting'],
            'transient_ischaemic_attack': ['temporary stroke symptoms', 'slurred speech', 'arm weakness', 'facial droop'],
            'deep_vein_thrombosis': ['painful leg swelling', 'redness', 'warmth', 'calf tenderness'],
            'acute_kidney_injury': ['decreased urination', 'swelling legs', 'tiredness', 'confusion']
        }
        
        detected = []
        for indicator, patterns in emergency_patterns.items():
            if any(pattern in symptoms_lower for pattern in patterns):
                detected.append(indicator)
        
        return detected
    
    def _assess_urgency_level(self, emergency_indicators: List[str]) -> str:
        """Assess medical urgency based on emergency indicators"""
        if not emergency_indicators:
            return "low"
        
        critical_indicators = ['crushing_chest_pain', 'thunderclap_headache', 'loss_consciousness', 'severe_bleeding']
        high_indicators = ['chest_pain_radiation', 'severe_breathlessness', 'pain_migration', 'appendicitis_classic']
        medium_indicators = ['heart_failure_triad', 'neck_stiffness']

        if any(ind in critical_indicators for ind in emergency_indicators):
            return "critical"
        elif any(ind in high_indicators for ind in emergency_indicators):
            return "high"
        elif any(ind in medium_indicators for ind in emergency_indicators):
            return "medium"
        else:
            return "low"

    
    def _combine_with_emergency_bank(self, priority_qs: str, emergency_qs: str, symptoms: str, urgency: str) -> List[str]:
        """Combine generated questions with emergency question bank"""
        questions = []
        
        # Add generated priority questions
        if priority_qs:
            questions.extend([q.strip() for q in priority_qs.split("|") if q.strip()])
        
        # Add emergency questions for high/critical urgency
        if urgency in ["high", "critical"] and emergency_qs:
            questions.extend([q.strip() for q in emergency_qs.split("|") if q.strip()])
        
        # Add curated questions based on symptom patterns
        curated = self._get_curated_emergency_questions(symptoms, urgency)
        questions.extend(curated)
        
        return list(dict.fromkeys(questions))  # Remove duplicates while preserving order
    
    def _get_curated_emergency_questions(self, symptoms: str, urgency: str) -> List[str]:
        """Get curated emergency questions from NICE protocols"""
        emergency_bank = {
            "critical": [
                "Are you experiencing severe difficulty breathing right now?",
                "Do you have crushing chest pain that won't go away?",
                "Have you lost consciousness or are you feeling faint?",
                "Is this the worst pain you have ever experienced?"
            ],
            "high": [
                "On a scale of 1-10, how severe is your pain right now?",
                "Are you experiencing any sweating, nausea, or shortness of breath?",
                "Has the pain changed or worsened since it started?",
                "Do you have a history of heart disease or similar episodes?"
            ],
            "medium": [
                "How long have you been experiencing these symptoms?",
                "Have you taken any medication and did it help?",
                "Are there any activities that make the symptoms worse?",
                "Have you experienced anything like this before?"
            ]
        }
        
        return emergency_bank.get(urgency, emergency_bank["medium"])[:2]
    
    def _medical_question_metric(self, example, pred, trace=None) -> float:
        """Custom metric for evaluating medical question quality"""
        if not pred or not hasattr(pred, 'questions'):
            return 0.0
        
        # Score based on medical relevance and NICE compliance
        score = 0.0
        questions = pred.questions if isinstance(pred.questions, list) else [pred.questions]
        
        for question in questions:
            if any(term in question.lower() for term in ['pain', 'severity', 'duration', 'location']):
                score += 0.3
            if any(term in question.lower() for term in ['emergency', 'severe', 'breathing', 'chest']):
                score += 0.4
            if len(question.split()) > 5:  # Prefer detailed questions
                score += 0.1
        
        return min(1.0, score)

class QuestionProgram(dspy.Module):
    """DSPy program orchestrating medical question generation with examples"""
    
    def __init__(self):
        super().__init__()
        self.question_module = MedicalQuestionModule()
    
    def forward(self, symptoms, conversation_context="", nice_protocols="", max_questions=3):
        return self.question_module(
            symptoms=symptoms,
            conversation_context=conversation_context,
            nice_protocols=nice_protocols,
            max_questions=max_questions
        )

class MedicalQuestionGenerator:
    """Production DSPy medical question generator with thinking enabled"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        self.model_name = model_name
        self._configure_dspy_with_thinking()
        
        # Initialize DSPy program with examples
        self.question_program = QuestionProgram()
        
        logger.info("❓ Medical Question Generator with thinking initialized", model=model_name)
    
    def _configure_dspy_with_thinking(self):
        """Configure DSPy with DeepSeek-R1 thinking ENABLED for medical reasoning"""
        try:
            lm = dspy.LM(
                f'ollama_chat/{self.model_name}',
                api_base='http://localhost:11434',
                api_key='',
                stream=False
            )
            dspy.configure(lm=lm)
            logger.info("✅ DSPy configured with thinking ENABLED for medical reasoning")
        except Exception as e:
            logger.error("❌ Failed to configure DSPy with thinking", error=str(e))
            raise
    
    def suggest_questions(
        self,
        symptom_text: str,
        conversation_context: str = "",
        nice_protocols: str = "",
        max_questions: int = 3
    ) -> Dict[str, Any]:
        """Generate NICE-compliant medical questions with reasoning"""
        try:
            result = self.question_program(
                symptoms=symptom_text,
                conversation_context=conversation_context,
                nice_protocols=nice_protocols,
                max_questions=max_questions
            )
            
            logger.info(
                "❓ Medical questions generated with reasoning",
                symptom_keywords=symptom_text[:50],
                questions_count=len(result.questions),
                urgency_level=getattr(result, 'urgency_level', 'unknown'),
                emergency_indicators=len(getattr(result, 'emergency_indicators', []))
            )
            
            return {
                "questions": result.questions,
                "medical_reasoning": getattr(result, 'medical_reasoning', ''),
                "emergency_assessment": getattr(result, 'emergency_assessment', ''),
                "urgency_level": getattr(result, 'urgency_level', 'low'),
                "emergency_indicators": getattr(result, 'emergency_indicators', []),
                "thinking_enabled": True
            }
            
        except Exception as e:
            logger.error("❌ Question generation error", error=str(e))
            return self._emergency_fallback_questions(symptom_text, max_questions)
    
    def _emergency_fallback_questions(self, symptom_text: str, max_questions: int) -> Dict[str, Any]:
        """Emergency fallback with basic NICE-compliant questions"""
        emergency_questions = [
            "On a scale of 1-10, how severe is your discomfort right now?",
            "How long have you been experiencing these symptoms?",
            "Are you having any difficulty breathing or chest pain?",
            "Have you taken any medication and did it help?",
            "Is this similar to anything you've experienced before?"
        ]
        
        return {
            "questions": emergency_questions[:max_questions],
            "medical_reasoning": "Fallback mode - using NICE-compliant safety questions",
            "emergency_assessment": "Unable to assess - using conservative approach",
            "urgency_level": "medium",  # Conservative fallback
            "emergency_indicators": [],
            "thinking_enabled": False
        }

# Singleton with DSPy examples and thinking enabled
question_generator = MedicalQuestionGenerator()

# Legacy compatibility maintained
def suggest_questions(symptom_text: str, max_questions: int = 3) -> List[str]:
    """Legacy function with enhanced DSPy backend"""
    result = question_generator.suggest_questions(symptom_text, max_questions=max_questions)
    return result["questions"]
