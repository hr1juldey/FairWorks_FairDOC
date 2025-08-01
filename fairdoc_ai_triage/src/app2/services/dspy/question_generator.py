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
            'crushing_chest_pain': ['crushing', 'squeezing', 'elephant on chest'],
            'chest_pain_radiation': ['radiating', 'spreading', 'arm pain', 'jaw pain'],
            'thunderclap_headache': ['thunderclap', 'worst headache ever', 'sudden severe'],
            'severe_breathlessness': ['can\'t breathe', 'gasping', 'severe shortness'],
            'loss_consciousness': ['fainted', 'passed out', 'lost consciousness'],
            'severe_bleeding': ['heavy bleeding', 'blood loss', 'hemorrhage'],
            'pain_migration': ['pain moved', 'started around navel', 'migrated'],
            'neck_stiffness': ['neck stiff', 'can\'t bend neck', 'meningeal']
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
        high_indicators = ['chest_pain_radiation', 'severe_breathlessness', 'pain_migration']
        
        if any(ind in critical_indicators for ind in emergency_indicators):
            return "critical"
        elif any(ind in high_indicators for ind in emergency_indicators):
            return "high"
        else:
            return "medium"
    
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
