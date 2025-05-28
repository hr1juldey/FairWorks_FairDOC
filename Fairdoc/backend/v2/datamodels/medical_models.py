"""
NHS-compliant medical models for telemedicine diagnosis and patient management.
Based on NICE guidelines, clinical pathways, and NHS digital health standards.
Supports AI-assisted diagnosis with human clinician oversight.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID, uuid4
from pydantic import Field, field_validator
from enum import Enum
import re

from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity
)

# ============================================================================
# NICE-COMPLIANT MEDICAL ENUMS
# ============================================================================

class NICEPathwayCategory(str, Enum):
    """NICE clinical pathway categories for telemedicine."""
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    DERMATOLOGY = "dermatology"
    MENTAL_HEALTH = "mental_health"
    GASTROINTESTINAL = "gastrointestinal"
    MUSCULOSKELETAL = "musculoskeletal"
    ENDOCRINE = "endocrine"
    NEUROLOGICAL = "neurological"
    GENITOURINARY = "genitourinary"
    INFECTIOUS_DISEASE = "infectious_disease"
    PAIN_MANAGEMENT = "pain_management"

class TelemedicineSuitability(str, Enum):
    """Suitability for telemedicine diagnosis per NICE guidelines."""
    HIGHLY_SUITABLE = "highly_suitable"       # Can be fully assessed remotely
    MODERATELY_SUITABLE = "moderately_suitable"  # Remote assessment with limitations
    LIMITED_SUITABILITY = "limited_suitability"  # Requires some physical examination
    NOT_SUITABLE = "not_suitable"            # Requires in-person assessment
    EMERGENCY_ONLY = "emergency_only"        # Emergency triage only

class SymptomSeverity(str, Enum):
    """Standardized symptom severity scale."""
    NONE = "none"           # 0
    MILD = "mild"           # 1-3
    MODERATE = "moderate"   # 4-6
    SEVERE = "severe"       # 7-8
    VERY_SEVERE = "very_severe"  # 9-10

class ClinicalDecisionStatus(str, Enum):
    """Status of clinical decision making process."""
    AI_ASSESSMENT_PENDING = "ai_assessment_pending"
    AI_ASSESSMENT_COMPLETE = "ai_assessment_complete"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_REVIEW_IN_PROGRESS = "human_review_in_progress"
    CLINICAL_DECISION_MADE = "clinical_decision_made"
    TREATMENT_INITIATED = "treatment_initiated"
    FOLLOW_UP_SCHEDULED = "follow_up_scheduled"

class InterventionType(str, Enum):
    """Types of primary interventions per NICE guidelines."""
    SELF_CARE = "self_care"
    PHARMACOLOGICAL = "pharmacological"
    NON_PHARMACOLOGICAL = "non_pharmacological"
    LIFESTYLE_MODIFICATION = "lifestyle_modification"
    PSYCHOLOGICAL_INTERVENTION = "psychological_intervention"
    URGENT_REFERRAL = "urgent_referral"
    ROUTINE_REFERRAL = "routine_referral"
    EMERGENCY_SERVICES = "emergency_services"
    FOLLOW_UP_MONITORING = "follow_up_monitoring"

class EscalationTrigger(str, Enum):
    """Triggers for escalation to human clinician."""
    RED_FLAG_SYMPTOMS = "red_flag_symptoms"
    COMPLEX_PRESENTATION = "complex_presentation"
    MULTIPLE_COMORBIDITIES = "multiple_comorbidities"
    HIGH_RISK_PATIENT = "high_risk_patient"
    AI_CONFIDENCE_LOW = "ai_confidence_low"
    PATIENT_REQUEST = "patient_request"
    CLINICAL_UNCERTAINTY = "clinical_uncertainty"
    SAFEGUARDING_CONCERN = "safeguarding_concern"

# ============================================================================
# MEDICAL CODING MODELS
# ============================================================================

class SNOMEDCode(TimestampMixin, ValidationMixin):
    """SNOMED CT codes for standardized medical terminology."""
    concept_id: str = Field(..., pattern=r"^\d{6,18}$", description="SNOMED CT concept ID")
    preferred_term: str = Field(..., max_length=255, description="Preferred clinical term")
    semantic_tag: str = Field(..., description="Semantic category")
    
    # Additional metadata
    is_active: bool = Field(default=True)
    module_id: str = Field(default="999000031000000106", description="UK Edition module")
    
    @field_validator('concept_id')
    @classmethod
    def validate_snomed_format(cls, v: str) -> str:
        """Validate SNOMED CT concept ID format."""
        if not v.isdigit() or len(v) < 6:
            raise ValueError('SNOMED CT concept ID must be 6-18 digits')
        return v

class ICD10Code(TimestampMixin, ValidationMixin):
    """ICD-10 codes for diagnosis classification."""
    code: str = Field(..., pattern=r"^[A-Z]\d{2}(\.[0-9X]{1,3})?$", description="ICD-10 code")
    description: str = Field(..., max_length=255, description="Condition description")
    category: str = Field(..., description="ICD-10 category")
    
    # NICE pathway mapping
    nice_pathway: Optional[NICEPathwayCategory] = None
    telemedicine_suitable: TelemedicineSuitability = Field(default=TelemedicineSuitability.MODERATELY_SUITABLE)
    
    @field_validator('code')
    @classmethod
    def validate_icd10_format(cls, v: str) -> str:
        """Validate ICD-10 code format."""
        pattern = re.compile(r"^[A-Z]\d{2}(\.[0-9X]{1,3})?$")
        if not pattern.match(v):
            raise ValueError('Invalid ICD-10 code format')
        return v.upper()

class NICEGuideline(TimestampMixin, ValidationMixin):
    """NICE clinical guidelines and quality standards."""
    guideline_id: str = Field(..., description="NICE guideline identifier (e.g., NG194)")
    title: str = Field(..., max_length=500)
    pathway_category: NICEPathwayCategory
    
    # Guideline content
    recommendations: List[str] = Field(default_factory=list)
    quality_statements: List[str] = Field(default_factory=list)
    red_flag_indicators: List[str] = Field(default_factory=list)
    
    # Digital health considerations
    telemedicine_guidance: Optional[str] = Field(None, max_length=1000)
    ai_considerations: Optional[str] = Field(None, max_length=1000)
    
    # Validity and updates
    published_date: datetime
    last_updated: datetime
    next_review_date: Optional[datetime] = None
    
    @field_validator('guideline_id')
    @classmethod
    def validate_nice_id(cls, v: str) -> str:
        """Validate NICE guideline ID format."""
        pattern = re.compile(r"^(NG|CG|QS|DG|IPG|MTG|TA)\d+$")
        if not pattern.match(v):
            raise ValueError('Invalid NICE guideline ID format')
        return v.upper()

# ============================================================================
# SYMPTOM MODELS
# ============================================================================

class SymptomDescriptor(TimestampMixin, ValidationMixin):
    """Standardized symptom description and classification."""
    symptom_id: UUID = Field(default_factory=uuid4)
    snomed_code: SNOMEDCode
    
    # Symptom characteristics
    symptom_name: str = Field(..., max_length=100)
    body_system: NICEPathwayCategory
    description: str = Field(..., max_length=500)
    
    # Assessment parameters
    severity_scale: Dict[str, str] = Field(
        default_factory=lambda: {
            "mild": "1-3: Minimal impact on daily activities",
            "moderate": "4-6: Some impact on daily activities",
            "severe": "7-8: Significant impact on daily activities",
            "very_severe": "9-10: Unable to perform daily activities"
        }
    )
    
    # Telemedicine assessment
    remotely_assessable: bool = Field(default=True)
    requires_visual_inspection: bool = Field(default=False)
    requires_physical_examination: bool = Field(default=False)
    
    # Clinical significance
    red_flag_potential: bool = Field(default=False)
    emergency_indicator: bool = Field(default=False)
    nice_guidance_available: bool = Field(default=False)
    
    # Associated questions for AI assessment
    assessment_questions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)

class PatientSymptom(BaseEntity, ValidationMixin, MetadataMixin):
    """Individual patient's reported symptom with assessment."""
    
    # Patient and symptom identification
    patient_id: UUID
    symptom_descriptor_id: UUID
    snomed_code: str = Field(..., description="SNOMED CT code for symptom")
    
    # Patient demographics for bias monitoring
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    
    # Symptom presentation
    severity: SymptomSeverity
    severity_score: int = Field(..., ge=0, le=10, description="Patient-rated severity 0-10")
    onset_date: datetime = Field(..., description="When symptom first appeared")
    duration: timedelta = Field(..., description="How long symptom has been present")
    
    # Symptom characteristics
    location: Optional[str] = Field(None, max_length=100, description="Anatomical location")
    character: Optional[str] = Field(None, max_length=100, description="Pain/symptom character")
    radiation: Optional[str] = Field(None, max_length=100, description="Does symptom spread")
    alleviating_factors: List[str] = Field(default_factory=list)
    exacerbating_factors: List[str] = Field(default_factory=list)
    associated_symptoms: List[str] = Field(default_factory=list)
    
    # Time patterns
    pattern: Optional[str] = Field(None, description="constant, intermittent, episodic")
    time_of_day_variation: Optional[str] = None
    seasonal_variation: Optional[str] = None
    
    # Impact assessment
    functional_impact: SymptomSeverity = Field(default=SymptomSeverity.MILD)
    sleep_impact: SymptomSeverity = Field(default=SymptomSeverity.NONE)
    work_impact: SymptomSeverity = Field(default=SymptomSeverity.NONE)
    quality_of_life_impact: int = Field(default=0, ge=0, le=10)
    
    # Clinical assessment
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    red_flag_identified: bool = Field(default=False)
    red_flag_details: List[str] = Field(default_factory=list)
    
    # AI assessment
    ai_assessed: bool = Field(default=False)
    ai_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_suggested_conditions: List[str] = Field(default_factory=list)
    
    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v: timedelta, info) -> timedelta:
        """Validate symptom duration is reasonable."""
        if v.total_seconds() < 0:
            raise ValueError('Duration cannot be negative')
        
        # Check if duration is consistent with onset
        if hasattr(info, 'data') and 'onset_date' in info.data:
            onset = info.data['onset_date']
            if isinstance(onset, datetime):
                expected_duration = datetime.now() - onset
                if abs((v - expected_duration).total_seconds()) > 86400:  # 1 day tolerance
                    raise ValueError('Duration inconsistent with onset date')
        
        return v
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score for this symptom."""
        base_score = self.severity_score / 10.0
        
        # Adjust for red flags
        if self.red_flag_identified:
            base_score = min(1.0, base_score + 0.3)
        
        # Adjust for functional impact
        impact_multiplier = {
            SymptomSeverity.NONE: 1.0,
            SymptomSeverity.MILD: 1.1,
            SymptomSeverity.MODERATE: 1.3,
            SymptomSeverity.SEVERE: 1.6,
            SymptomSeverity.VERY_SEVERE: 2.0
        }
        
        base_score *= impact_multiplier.get(self.functional_impact, 1.0)
        
        return min(1.0, base_score)

# ============================================================================
# DIAGNOSIS MODELS
# ============================================================================

class DifferentialDiagnosis(TimestampMixin, ValidationMixin):
    """Potential diagnosis with evidence and probability."""
    diagnosis_id: UUID = Field(default_factory=uuid4)
    
    # Medical coding
    icd10_code: str = Field(..., description="Primary ICD-10 code")
    snomed_codes: List[str] = Field(default_factory=list, description="Related SNOMED CT codes")
    
    # Diagnosis details
    condition_name: str = Field(..., max_length=200)
    description: str = Field(..., max_length=1000)
    pathway_category: NICEPathwayCategory
    
    # Clinical evidence
    supporting_symptoms: List[str] = Field(default_factory=list)
    contradicting_symptoms: List[str] = Field(default_factory=list)
    required_investigations: List[str] = Field(default_factory=list)
    
    # Probability and confidence
    probability_score: float = Field(..., ge=0.0, le=1.0, description="Likelihood of this diagnosis")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    evidence_strength: Literal["weak", "moderate", "strong"] = "moderate"
    
    # NICE compliance
    nice_guideline_ref: Optional[str] = None
    evidence_based: bool = Field(default=True)
    
    # Telemedicine considerations
    telemedicine_diagnosable: TelemedicineSuitability
    requires_physical_exam: bool = Field(default=False)
    requires_investigations: bool = Field(default=False)
    
    # Risk stratification
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    
    def update_probability(self, new_evidence: Dict[str, Any]):
        """Update diagnosis probability based on new evidence."""
        # Bayesian-style probability update
        prior = self.probability_score
        
        # Evidence weighting
        evidence_weight = len(new_evidence.get('supporting', [])) * 0.1
        contradicting_weight = len(new_evidence.get('contradicting', [])) * 0.15
        
        # Update probability
        posterior = prior + evidence_weight - contradicting_weight
        self.probability_score = max(0.0, min(1.0, posterior))
        
        self.update_timestamp()

class ClinicalAssessment(BaseEntity, ValidationMixin, MetadataMixin):
    """Comprehensive clinical assessment combining symptoms and differential diagnoses."""
    
    # Patient identification
    patient_id: UUID
    assessment_id: UUID = Field(default_factory=uuid4)
    
    # Patient demographics for bias monitoring
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    
    # Assessment context
    assessment_type: Literal["initial", "follow_up", "urgent", "routine"] = "initial"
    presentation_mode: Literal["telemedicine", "face_to_face", "telephone", "digital_triage"] = "telemedicine"
    
    # Clinical presentation
    chief_complaint: str = Field(..., max_length=500, description="Primary presenting complaint")
    history_of_present_illness: str = Field(..., max_length=2000)
    symptoms: List[UUID] = Field(default_factory=list, description="References to PatientSymptom IDs")
    
    # Medical history
    past_medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: Dict[str, Any] = Field(default_factory=dict)
    
    # Assessment findings
    differential_diagnoses: List[UUID] = Field(default_factory=list, description="DifferentialDiagnosis IDs")
    most_likely_diagnosis: Optional[UUID] = None
    
    # Clinical decision making
    decision_status: ClinicalDecisionStatus = Field(default=ClinicalDecisionStatus.AI_ASSESSMENT_PENDING)
    escalation_triggers: List[EscalationTrigger] = Field(default_factory=list)
    escalation_required: bool = Field(default=False)
    escalation_urgency: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    
    # AI assessment
    ai_processing_started: Optional[datetime] = None
    ai_processing_completed: Optional[datetime] = None
    ai_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_recommendations: List[str] = Field(default_factory=list)
    
    # Human clinician review
    clinician_review_required: bool = Field(default=False)
    clinician_id: Optional[UUID] = None
    clinician_notes: Optional[str] = Field(None, max_length=2000)
    clinician_agreement_with_ai: Optional[bool] = None
    
    # Assessment outcomes
    primary_diagnosis: Optional[str] = None
    secondary_diagnoses: List[str] = Field(default_factory=list)
    diagnostic_certainty: Literal["definitive", "probable", "possible", "uncertain"] = "uncertain"
    
    # Safety netting
    red_flags_assessed: bool = Field(default=False)
    safety_net_advice: List[str] = Field(default_factory=list)
    follow_up_required: bool = Field(default=False)
    follow_up_timeframe: Optional[timedelta] = None
    
    def add_escalation_trigger(self, trigger: EscalationTrigger, details: str = ""):
        """Add escalation trigger and update status."""
        self.escalation_triggers.append(trigger)
        self.escalation_required = True
        
        # Determine urgency based on trigger type
        urgent_triggers = [
            EscalationTrigger.RED_FLAG_SYMPTOMS,
            EscalationTrigger.HIGH_RISK_PATIENT,
            EscalationTrigger.SAFEGUARDING_CONCERN
        ]
        
        if trigger in urgent_triggers:
            self.escalation_urgency = UrgencyLevel.URGENT
        
        self.add_metadata(f"escalation_{trigger.value}", {
            "trigger": trigger.value,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.update_timestamp()
    
    def calculate_assessment_complexity(self) -> float:
        """Calculate complexity score for resource allocation."""
        complexity_score = 0.0
        
        # Number of symptoms
        complexity_score += len(self.symptoms) * 0.1
        
        # Number of differential diagnoses
        complexity_score += len(self.differential_diagnoses) * 0.15
        
        # Comorbidities
        complexity_score += len(self.past_medical_history) * 0.05
        
        # Red flags or escalation triggers
        complexity_score += len(self.escalation_triggers) * 0.2
        
        # AI confidence (lower confidence = higher complexity)
        if self.ai_confidence_score:
            complexity_score += (1.0 - self.ai_confidence_score) * 0.3
        
        return min(1.0, complexity_score)

# ============================================================================
# INTERVENTION AND TREATMENT MODELS
# ============================================================================

class PrimaryIntervention(TimestampMixin, ValidationMixin):
    """Primary intervention recommendations based on NICE guidelines."""
    intervention_id: UUID = Field(default_factory=uuid4)
    
    # Intervention classification
    intervention_type: InterventionType
    nice_guideline_ref: Optional[str] = None
    evidence_level: Literal["high", "moderate", "low", "very_low"] = "moderate"
    
    # Intervention details
    intervention_name: str = Field(..., max_length=200)
    description: str = Field(..., max_length=1000)
    instructions: List[str] = Field(default_factory=list)
    
    # Suitability criteria
    suitable_for_telemedicine: bool = Field(default=True)
    age_restrictions: Optional[str] = None
    contraindications: List[str] = Field(default_factory=list)
    precautions: List[str] = Field(default_factory=list)
    
    # Expected outcomes
    expected_benefit: str = Field(..., max_length=500)
    time_to_benefit: Optional[timedelta] = None
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Monitoring requirements
    requires_monitoring: bool = Field(default=False)
    monitoring_frequency: Optional[timedelta] = None
    monitoring_parameters: List[str] = Field(default_factory=list)
    
    # Safety considerations
    side_effects: List[str] = Field(default_factory=list)
    warning_signs: List[str] = Field(default_factory=list)
    when_to_seek_help: List[str] = Field(default_factory=list)

class TreatmentPlan(BaseEntity, ValidationMixin, MetadataMixin):
    """Comprehensive treatment plan with interventions and monitoring."""
    
    # Plan identification
    patient_id: UUID
    assessment_id: UUID
    plan_id: UUID = Field(default_factory=uuid4)
    
    # Plan overview
    primary_diagnosis: str = Field(..., max_length=200)
    treatment_goals: List[str] = Field(default_factory=list)
    target_outcomes: List[str] = Field(default_factory=list)
    
    # Interventions
    primary_interventions: List[UUID] = Field(default_factory=list, description="PrimaryIntervention IDs")
    immediate_actions: List[str] = Field(default_factory=list)
    ongoing_management: List[str] = Field(default_factory=list)
    
    # Monitoring and follow-up
    follow_up_required: bool = Field(default=False)
    follow_up_timeframe: Optional[timedelta] = None
    monitoring_plan: Dict[str, Any] = Field(default_factory=dict)
    
    # Safety netting
    red_flag_symptoms: List[str] = Field(default_factory=list)
    when_to_seek_urgent_care: List[str] = Field(default_factory=list)
    emergency_contact_info: Optional[str] = None
    
    # Plan approval and execution
    approved_by_clinician: bool = Field(default=False)
    approving_clinician_id: Optional[UUID] = None
    plan_started: Optional[datetime] = None
    plan_completed: Optional[datetime] = None
    
    # Patient engagement
    patient_consent: bool = Field(default=False)
    patient_understanding_confirmed: bool = Field(default=False)
    written_information_provided: bool = Field(default=False)
    
    def activate_plan(self, clinician_id: UUID):
        """Activate treatment plan with clinician approval."""
        self.approved_by_clinician = True
        self.approving_clinician_id = clinician_id
        self.plan_started = datetime.utcnow()
        self.update_timestamp()

# ============================================================================
# CONDITION ROUTING MODELS
# ============================================================================

class SpecialtyRouting(TimestampMixin, ValidationMixin):
    """Routing logic for specialist referrals based on conditions."""
    
    # Condition identification
    condition_codes: List[str] = Field(..., description="ICD-10 or SNOMED codes")
    condition_category: NICEPathwayCategory
    
    # Specialist requirements
    primary_specialty: str = Field(..., max_length=100, description="Primary specialty required")
    secondary_specialties: List[str] = Field(default_factory=list)
    subspecialty_required: Optional[str] = None
    
    # Urgency and priority
    default_urgency: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    escalation_criteria: List[str] = Field(default_factory=list)
    
    # Telemedicine suitability
    telemedicine_first_contact: bool = Field(default=True)
    telemedicine_follow_up: bool = Field(default=True)
    requires_face_to_face: bool = Field(default=False)
    
    # Resource requirements
    estimated_consultation_time: timedelta = Field(default=timedelta(minutes=20))
    requires_investigations: bool = Field(default=False)
    requires_multidisciplinary_team: bool = Field(default=False)

class PatientRoutingDecision(BaseEntity, ValidationMixin, MetadataMixin):
    """Individual patient routing decision with audit trail."""
    
    # Patient and assessment context
    patient_id: UUID
    assessment_id: UUID
    routing_id: UUID = Field(default_factory=uuid4)
    
    # Patient demographics for resource planning
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    
    # Routing decision
    recommended_pathway: NICEPathwayCategory
    specialty_required: str = Field(..., max_length=100)
    urgency_level: UrgencyLevel
    risk_level: RiskLevel
    
    # Decision rationale
    routing_criteria_met: List[str] = Field(default_factory=list)
    escalation_triggers: List[EscalationTrigger] = Field(default_factory=list)
    ai_recommendation: Optional[str] = Field(None, max_length=500)
    clinician_override: bool = Field(default=False)
    override_reason: Optional[str] = None
    
    # Resource allocation
    estimated_wait_time: Optional[timedelta] = None
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution tracking
    referral_initiated: bool = Field(default=False)
    specialist_assigned: Optional[UUID] = None
    appointment_scheduled: Optional[datetime] = None
    
    # Outcomes
    patient_seen: bool = Field(default=False)
    routing_accuracy: Optional[bool] = None  # Was the routing decision correct?
    patient_satisfaction: Optional[int] = Field(None, ge=1, le=5)
    
    def calculate_priority_score(self) -> float:
        """Calculate priority score for queue management."""
        base_score = 0.5
        
        # Urgency multiplier
        urgency_weights = {
            UrgencyLevel.ROUTINE: 1.0,
            UrgencyLevel.URGENT: 1.5,
            UrgencyLevel.EMERGENT: 2.0,
            UrgencyLevel.IMMEDIATE: 3.0
        }
        
        # Risk multiplier
        risk_weights = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MODERATE: 1.3,
            RiskLevel.HIGH: 1.6,
            RiskLevel.CRITICAL: 2.0
        }
        
        score = base_score * urgency_weights.get(self.urgency_level, 1.0)
        score *= risk_weights.get(self.risk_level, 1.0)
        
        # Age adjustment (older patients get slight priority)
        if self.patient_age and self.patient_age > 65:
            score *= 1.1
        
        # Escalation triggers
        score += len(self.escalation_triggers) * 0.1
        
        self.priority_score = min(1.0, score)
        return self.priority_score

# ============================================================================
# CONDITION-SPECIFIC MODELS
# ============================================================================

class ChestPainAssessment(BaseEntity, ValidationMixin):
    """Specialized assessment model for chest pain presentations."""
    
    patient_id: UUID
    assessment_id: UUID
    
    # Chest pain characteristics
    pain_location: Literal["central", "left_sided", "right_sided", "epigastric", "back"] = "central"
    pain_character: Literal["sharp", "dull", "crushing", "burning", "stabbing", "aching"] = "aching"
    pain_severity: int = Field(..., ge=0, le=10)
    pain_duration: timedelta
    pain_radiation: bool = Field(default=False)
    radiation_sites: List[str] = Field(default_factory=list)
    
    # Associated symptoms
    shortness_of_breath: bool = Field(default=False)
    nausea_vomiting: bool = Field(default=False)
    sweating: bool = Field(default=False)
    dizziness: bool = Field(default=False)
    palpitations: bool = Field(default=False)
    
    # Risk factors
    cardiovascular_risk_factors: List[str] = Field(default_factory=list)
    previous_cardiac_history: bool = Field(default=False)
    family_history_cardiac: bool = Field(default=False)
    
    # Clinical scores
    heart_score: Optional[int] = Field(None, ge=0, le=10, description="HEART score for ACS risk")
    well_score: Optional[int] = Field(None, ge=0, le=12, description="Wells score for PE risk")
    
    # NICE pathway compliance
    nice_cg95_compliant: bool = Field(default=True, description="NICE CG95 Chest Pain compliance")
    troponin_required: bool = Field(default=False)
    ecg_required: bool = Field(default=True)
    
    def calculate_heart_score(self) -> int:
        """Calculate HEART score for acute coronary syndrome risk."""
        score = 0
        
        # History (placeholder - would need more detailed history)
        if len(self.cardiovascular_risk_factors) >= 3:
            score += 2
        elif len(self.cardiovascular_risk_factors) >= 1:
            score += 1
        
        # ECG (would need ECG interpretation)
        score += 1  # Placeholder
        
        # Age
        if hasattr(self, 'patient_age'):
            if self.patient_age >= 65:
                score += 2
            elif self.patient_age >= 45:
                score += 1
        
        # Risk factors
        if self.previous_cardiac_history:
            score += 2
        elif self.family_history_cardiac:
            score += 1
        
        # Troponin (would need lab values)
        score += 1  # Placeholder
        
        self.heart_score = min(10, score)
        return self.heart_score

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class SymptomAssessmentResponse(BaseResponse):
    """Response for symptom assessment requests."""
    symptoms: List[PatientSymptom]
    overall_risk_level: RiskLevel
    urgency_level: UrgencyLevel
    recommended_pathway: NICEPathwayCategory
    ai_confidence: float
    escalation_required: bool

class DiagnosisResponse(BaseResponse):
    """Response for diagnosis requests."""
    assessment: ClinicalAssessment
    differential_diagnoses: List[DifferentialDiagnosis]
    recommended_investigations: List[str]
    treatment_plan: Optional[TreatmentPlan]
    routing_decision: PatientRoutingDecision

class SpecialistRoutingResponse(BaseResponse):
    """Response for specialist routing requests."""
    routing_decision: PatientRoutingDecision
    available_specialists: List[Dict[str, Any]]
    estimated_wait_time: timedelta
    alternative_pathways: List[str]

# ============================================================================
# NICE PATHWAY CONFIGURATIONS
# ============================================================================


NICE_PATHWAY_CONFIGS = {
    NICEPathwayCategory.CARDIOVASCULAR: {
        "primary_guidelines": ["NG194", "CG95", "NG185"],
        "telemedicine_suitable": True,
        "red_flags": [
            "Chest pain with radiation",
            "Severe breathlessness",
            "Syncope",
            "Severe hypertension"
        ],
        "specialist_routing": "Cardiology",
        "max_ai_confidence_threshold": 0.7
    },
    
    NICEPathwayCategory.RESPIRATORY: {
        "primary_guidelines": ["NG117", "QS25", "NG80"],
        "telemedicine_suitable": True,
        "red_flags": [
            "Severe breathlessness at rest",
            "Cyanosis",
            "Stridor",
            "Haemoptysis"
        ],
        "specialist_routing": "Respiratory Medicine",
        "max_ai_confidence_threshold": 0.8
    },
    
    NICEPathwayCategory.DERMATOLOGY: {
        "primary_guidelines": ["NG12", "CG153", "NG142"],
        "telemedicine_suitable": True,
        "red_flags": [
            "Rapidly changing lesion",
            "Ulceration",
            "Irregular borders",
            "Multiple colors"
        ],
        "specialist_routing": "Dermatology",
        "max_ai_confidence_threshold": 0.9
    },
    
    NICEPathwayCategory.MENTAL_HEALTH: {
        "primary_guidelines": ["NG222", "CG136", "NG185"],
        "telemedicine_suitable": True,
        "red_flags": [
            "Suicidal ideation",
            "Psychosis",
            "Self-harm",
            "Severe agitation"
        ],
        "specialist_routing": "Psychiatry",
        "max_ai_confidence_threshold": 0.6
    }
}

# Default symptom-to-pathway mapping
SYMPTOM_PATHWAY_MAPPING = {
    "chest pain": NICEPathwayCategory.CARDIOVASCULAR,
    "shortness of breath": NICEPathwayCategory.RESPIRATORY,
    "skin lesion": NICEPathwayCategory.DERMATOLOGY,
    "depression": NICEPathwayCategory.MENTAL_HEALTH,
    "abdominal pain": NICEPathwayCategory.GASTROINTESTINAL,
    "back pain": NICEPathwayCategory.MUSCULOSKELETAL,
    "headache": NICEPathwayCategory.NEUROLOGICAL,
    "fatigue": NICEPathwayCategory.ENDOCRINE,
    "urinary symptoms": NICEPathwayCategory.GENITOURINARY,
    "fever": NICEPathwayCategory.INFECTIOUS_DISEASE
}

# NICE red flag symptoms requiring immediate escalation
NICE_RED_FLAGS = {
    "cardiovascular": [
        "Central chest pain with radiation to arm/jaw",
        "Chest pain with severe breathlessness",
        "Chest pain with loss of consciousness",
        "Signs of heart failure"
    ],
    "respiratory": [
        "Acute severe breathlessness",
        "Stridor",
        "Cyanosis",
        "Massive haemoptysis"
    ],
    "neurological": [
        "Sudden severe headache",
        "Focal neurological deficit",
        "Altered consciousness",
        "Signs of meningism"
    ],
    "gastrointestinal": [
        "Severe abdominal pain with rigidity",
        "Significant GI bleeding",
        "Signs of bowel obstruction",
        "Severe dehydration"
    ]
}
