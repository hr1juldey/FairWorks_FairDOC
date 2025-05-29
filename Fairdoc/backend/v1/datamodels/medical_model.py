"""
Medical Data Models - NICE/NIH Compliant - Pydantic v2
General purpose medical triage models with ML scoring integration
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
from enum import IntEnum, Enum
import uuid

# =============================================================================
# CORE ENUMS WITH ML SCORING - NICE/NIH Standards
# =============================================================================

class SeverityLevel(IntEnum):
    """Pain/symptom severity levels with integer scores for ML evaluation"""
    NONE = 0           # No pain/symptoms
    MINIMAL = 1        # 1-2 on scale
    MILD = 3           # 3-4 on scale
    MODERATE = 5       # 5-6 on scale
    SEVERE = 7         # 7-8 on scale
    VERY_SEVERE = 9    # 9-10 on scale
    
    @classmethod
    def from_scale_score(cls, score: int) -> 'SeverityLevel':
        """Convert 0-10 pain scale to severity level"""
        if score == 0:
            return cls.NONE
        elif 1 <= score <= 2:
            return cls.MINIMAL
        elif 3 <= score <= 4:
            return cls.MILD
        elif 5 <= score <= 6:
            return cls.MODERATE
        elif 7 <= score <= 8:
            return cls.SEVERE
        else:  # 9-10
            return cls.VERY_SEVERE

class UrgencyCategory(IntEnum):
    """NHS urgency categories with integer scores for triage algorithms"""
    SELF_CARE = 1          # Self-management advice
    ROUTINE = 2            # GP appointment within 48 hours
    LESS_URGENT = 3        # Care needed within 6 hours
    URGENT = 4             # Urgent care needed within 1 hour
    EMERGENCY = 5          # Life threatening - call 999
    
    def get_timeframe_hours(self) -> Optional[float]:
        """Get recommended timeframe in hours"""
        timeframes = {
            self.SELF_CARE: None,
            self.ROUTINE: 48.0,
            self.LESS_URGENT: 6.0,
            self.URGENT: 1.0,
            self.EMERGENCY: 0.0  # Immediate
        }
        return timeframes[self]

class ImportanceLevel(IntEnum):
    """Clinical importance levels for coordinate mapping"""
    LOW = 1            # Minor conditions, self-limiting
    MODERATE_LOW = 2   # Some clinical significance
    MODERATE = 3       # Clinically important
    MODERATE_HIGH = 4  # Significant clinical impact
    HIGH = 5           # Major clinical significance

class TriageDisposition(IntEnum):
    """Final triage disposition with priority scores"""
    SELF_CARE = 1             # Self-care with advice
    PHARMACY = 2              # Pharmacy consultation
    GP_ROUTINE = 3            # GP routine appointment
    GP_EMERGENCY = 4          # GP emergency appointment
    URGENT_CARE = 5           # Urgent Care Center
    ED_URGENT = 6             # Emergency Department - urgent
    ED_IMMEDIATE = 7          # Emergency Department - immediate
    MENTAL_HEALTH = 8         # Mental health crisis team
    DENTAL = 9                # Dental emergency

class GenderType(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class QuestionType(str, Enum):
    """NICE protocol question types"""
    TEXT = "text"
    NUMBER = "number"
    MULTIPLE_CHOICE = "multiple_choice"
    MULTIPLE_SELECT = "multiple_select"
    YES_NO = "yes_no"
    SCALE_1_10 = "scale_1_10"
    FILE_UPLOAD = "file_upload"
    DATE = "date"
    TIME = "time"

# =============================================================================
# COORDINATE MAPPING SYSTEM FOR ML TRIAGE
# =============================================================================

class TriageCoordinate(BaseModel):
    """X-Y coordinate system for triage decision making"""
    model_config = ConfigDict(frozen=True)
    
    urgency: float = Field(..., ge=-1.0, le=1.0, description="Urgency score: -1 (not urgent) to +1 (very urgent)")
    importance: float = Field(..., ge=-1.0, le=1.0, description="Importance score: -1 (not important) to +1 (very important)")
    
    def get_quadrant(self) -> str:
        """Determine triage quadrant"""
        if self.urgency >= 0 and self.importance >= 0:
            return "urgent_important"      # High priority
        elif self.urgency >= 0 and self.importance < 0:
            return "urgent_not_important"  # Fast track
        elif self.urgency < 0 and self.importance >= 0:
            return "not_urgent_important"  # Planned care
        else:
            return "not_urgent_not_important"  # Self-care
    
    def get_priority_score(self) -> float:
        """Calculate overall priority score (0-1)"""
        # Weighted combination: urgency slightly more important than importance
        return (0.6 * (self.urgency + 1) / 2) + (0.4 * (self.importance + 1) / 2)
    
    def to_urgency_category(self) -> UrgencyCategory:
        """Convert coordinate to NHS urgency category"""
        priority = self.get_priority_score()
        if priority >= 0.8:
            return UrgencyCategory.EMERGENCY
        elif priority >= 0.6:
            return UrgencyCategory.URGENT
        elif priority >= 0.4:
            return UrgencyCategory.LESS_URGENT
        elif priority >= 0.2:
            return UrgencyCategory.ROUTINE
        else:
            return UrgencyCategory.SELF_CARE

class RiskScoreWeights(BaseModel):
    """Configurable weights for risk scoring algorithm"""
    model_config = ConfigDict(frozen=True)
    
    age_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    gender_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    symptom_severity_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    red_flags_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    medical_history_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    vital_signs_weight: float = Field(default=0.10, ge=0.0, le=1.0)

# =============================================================================
# BASE MEDICAL MODELS - Generic for all conditions
# =============================================================================

class Demographics(BaseModel):
    """Patient demographics following NHS standards"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    age: int = Field(..., ge=0, le=150, description="Patient age in years")
    gender: GenderType = Field(..., description="Patient gender")
    pregnancy_status: Optional[bool] = Field(default=None, description="If female, pregnancy status")
    ethnicity: Optional[str] = Field(default=None, max_length=100, description="Patient ethnicity (optional)")
    postcode_sector: Optional[str] = Field(default=None, max_length=10, description="Postcode sector for location")
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not 0 <= v <= 150:
            raise ValueError('Age must be between 0 and 150')
        return v
    
    def get_age_risk_score(self) -> float:
        """Calculate age-based risk score (0-1)"""
        if self.age < 18:
            return 0.1
        elif self.age < 45:
            return 0.2
        elif self.age < 65:
            return 0.4
        elif self.age < 75:
            return 0.6
        else:
            return 0.8

class VitalSigns(BaseModel):
    """Standard vital signs measurements with normal ranges"""
    model_config = ConfigDict(validate_default=True)
    
    systolic_bp: Optional[int] = Field(default=None, ge=50, le=300, description="Systolic blood pressure mmHg")
    diastolic_bp: Optional[int] = Field(default=None, ge=30, le=200, description="Diastolic blood pressure mmHg")
    heart_rate: Optional[int] = Field(default=None, ge=30, le=300, description="Heart rate beats per minute")
    respiratory_rate: Optional[int] = Field(default=None, ge=8, le=60, description="Respiratory rate per minute")
    temperature: Optional[float] = Field(default=None, ge=32.0, le=45.0, description="Temperature in Celsius")
    oxygen_saturation: Optional[int] = Field(default=None, ge=70, le=100, description="Oxygen saturation percentage")
    blood_glucose: Optional[float] = Field(default=None, ge=2.0, le=30.0, description="Blood glucose mmol/L")
    peak_flow: Optional[int] = Field(default=None, ge=50, le=800, description="Peak expiratory flow L/min")
    
    def calculate_vital_signs_score(self) -> float:
        """Calculate abnormal vital signs score (0-1)"""
        score = 0.0
        count = 0
        
        # Blood pressure scoring
        if self.systolic_bp is not None and self.diastolic_bp is not None:
            if self.systolic_bp > 180 or self.diastolic_bp > 110:
                score += 0.8  # Hypertensive crisis
            elif self.systolic_bp > 140 or self.diastolic_bp > 90:
                score += 0.4  # High BP
            elif self.systolic_bp < 90:
                score += 0.6  # Hypotension
            count += 1
        
        # Heart rate scoring
        if self.heart_rate is not None:
            if self.heart_rate > 120 or self.heart_rate < 50:
                score += 0.6
            elif self.heart_rate > 100 or self.heart_rate < 60:
                score += 0.3
            count += 1
        
        # Temperature scoring
        if self.temperature is not None:
            if self.temperature > 38.5 or self.temperature < 35.0:
                score += 0.7
            elif self.temperature > 37.5 or self.temperature < 36.0:
                score += 0.3
            count += 1
        
        # Oxygen saturation scoring
        if self.oxygen_saturation is not None:
            if self.oxygen_saturation < 90:
                score += 0.9
            elif self.oxygen_saturation < 95:
                score += 0.5
            count += 1
        
        return min(score / max(count, 1), 1.0)

class MedicalHistory(BaseModel):
    """Comprehensive medical history with risk scoring"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    conditions: List[str] = Field(default_factory=list, description="Known medical conditions")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    surgeries: List[str] = Field(default_factory=list, description="Previous surgeries")
    family_history: List[str] = Field(default_factory=list, description="Relevant family history")
    smoking_status: Optional[str] = Field(default=None, description="Never/Former/Current smoker")
    alcohol_units_weekly: Optional[int] = Field(default=None, ge=0, le=200, description="Alcohol units per week")
    
    @field_validator('conditions', 'medications', 'allergies', 'surgeries', 'family_history')
    @classmethod
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        return [item.strip().lower() for item in v if item.strip()]
    
    def calculate_risk_score(self) -> float:
        """Calculate medical history risk score (0-1)"""
        score = 0.0
        
        # High-risk conditions
        high_risk_conditions = [
            "diabetes", "heart disease", "stroke", "cancer", "kidney disease",
            "liver disease", "copd", "asthma", "hypertension", "atrial fibrillation"
        ]
        
        for condition in self.conditions:
            if any(risk in condition.lower() for risk in high_risk_conditions):
                score += 0.2
        
        # Smoking risk
        if self.smoking_status == "Current":
            score += 0.3
        elif self.smoking_status == "Former":
            score += 0.1
        
        # Alcohol risk
        if self.alcohol_units_weekly and self.alcohol_units_weekly > 14:
            score += 0.2
        
        return min(score, 1.0)

class UploadedFile(BaseModel):
    """File attachment model with metadata"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique file identifier")
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    file_type: str = Field(..., description="MIME type")
    file_category: str = Field(..., description="medical_image/lab_report/prescription/other")
    minio_url: str = Field(..., description="MinIO storage URL")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    upload_timestamp: datetime = Field(default_factory=datetime.now, description="Upload time")
    description: Optional[str] = Field(default=None, max_length=500, description="User description of file")
    ai_analysis: Optional[Dict[str, Any]] = Field(default=None, description="AI analysis results")

# =============================================================================
# NICE PROTOCOL MODELS
# =============================================================================

class NICEProtocolQuestion(BaseModel):
    """NICE clinical protocol question"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    question_id: int = Field(..., description="Unique question identifier")
    category: str = Field(..., description="Question category")
    question_text: str = Field(..., min_length=1, max_length=1000, description="Question text")
    question_type: QuestionType = Field(..., description="Type of question input")
    options: Optional[List[str]] = Field(default=None, description="Available options for choice questions")
    validation_rules: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules")
    is_required: bool = Field(default=True, description="Whether question is mandatory")
    is_red_flag: bool = Field(default=False, description="Whether question identifies red flag symptoms")
    order_index: int = Field(..., ge=0, description="Question order in protocol")
    condition_specific: Optional[str] = Field(default=None, description="Specific condition this applies to")
    next_question_logic: Optional[Dict[str, Any]] = Field(default=None, description="Conditional question logic")

class PatientResponse(BaseModel):
    """Patient response to a protocol question with scoring"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique response ID")
    question_id: int = Field(..., description="Question being answered")
    response_text: str = Field(..., min_length=1, description="Patient's text response")
    response_value: Optional[Union[str, int, float, bool, List[str]]] = Field(default=None, description="Structured response value")
    confidence_level: Optional[float] = Field(default=None, ge=0, le=1, description="Patient's confidence in answer")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    response_time_seconds: Optional[int] = Field(default=None, ge=0, description="Time taken to respond")

# =============================================================================
# CHEST PAIN SPECIFIC MODELS - Disease Specialization with ML Scoring
# =============================================================================

class ChestPainCharacteristics(BaseModel):
    """Detailed chest pain assessment with ML scoring"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    location: str = Field(..., description="Pain location")
    radiation_areas: List[str] = Field(default_factory=list, description="Areas where pain radiates")
    character: str = Field(..., description="Pain character")
    severity_score: int = Field(..., ge=0, le=10, description="Pain severity 0-10 scale")
    onset_description: str = Field(..., description="How pain started")
    duration_description: str = Field(..., description="How long pain has lasted")
    timing_pattern: str = Field(..., description="Pattern (constant/intermittent/waves)")
    
    # Red flag indicators with boolean scoring
    is_crushing: bool = Field(default=False, description="Crushing chest pain (red flag)")
    is_radiating_left_arm: bool = Field(default=False, description="Radiates to left arm (red flag)")
    is_radiating_jaw: bool = Field(default=False, description="Radiates to jaw/neck (red flag)")
    is_exertional: bool = Field(default=False, description="Brought on by exertion")
    relieved_by_rest: bool = Field(default=False, description="Relieved by rest")
    relieved_by_gtn: bool = Field(default=False, description="Relieved by GTN spray")
    
    triggers: List[str] = Field(default_factory=list, description="What triggers the pain")
    relief_factors: List[str] = Field(default_factory=list, description="What relieves the pain")
    associated_activities: List[str] = Field(default_factory=list, description="Activities when pain occurs")
    
    def get_severity_level(self) -> SeverityLevel:
        """Convert severity score to enum"""
        return SeverityLevel.from_scale_score(self.severity_score)
    
    def calculate_cardiac_risk_score(self) -> float:
        """Calculate cardiac risk score based on pain characteristics (0-1)"""
        score = 0.0
        
        # Pain character scoring
        if "crushing" in self.character.lower() or "pressure" in self.character.lower():
            score += 0.3
        elif "sharp" in self.character.lower() or "stabbing" in self.character.lower():
            score += 0.1
        
        # Radiation scoring (highest risk)
        if self.is_radiating_left_arm:
            score += 0.4
        if self.is_radiating_jaw:
            score += 0.3
        
        # Pain characteristics
        if self.is_exertional:
            score += 0.2
        if self.relieved_by_rest:
            score += 0.2
        if self.relieved_by_gtn:
            score += 0.3
        
        # Severity scoring
        if self.severity_score >= 7:
            score += 0.2
        
        return min(score, 1.0)

class ChestPainRedFlags(BaseModel):
    """Red flag symptoms with binary scoring"""
    model_config = ConfigDict(validate_default=True)
    
    acute_onset_severe_pain: bool = Field(default=False)
    tearing_interscapular_pain: bool = Field(default=False)
    syncope_with_chest_pain: bool = Field(default=False)
    hypotension: bool = Field(default=False)
    new_murmur: bool = Field(default=False)
    unequal_arm_bp: bool = Field(default=False)
    absent_pulse: bool = Field(default=False)
    
    def count_red_flags(self) -> int:
        """Count number of red flags present"""
        flags = [
            self.acute_onset_severe_pain, self.tearing_interscapular_pain,
            self.syncope_with_chest_pain, self.hypotension, self.new_murmur,
            self.unequal_arm_bp, self.absent_pulse
        ]
        return sum(flags)
    
    def get_red_flag_score(self) -> float:
        """Get red flag urgency score (0-1)"""
        count = self.count_red_flags()
        if count >= 3:
            return 1.0
        elif count >= 2:
            return 0.8
        elif count >= 1:
            return 0.6
        else:
            return 0.0

class ChestPainAssociatedSymptoms(BaseModel):
    """Associated symptoms with scoring weights"""
    model_config = ConfigDict(validate_default=True)
    
    shortness_of_breath: bool = Field(default=False)
    cough: bool = Field(default=False)
    wheeze: bool = Field(default=False)
    sweating: bool = Field(default=False)
    nausea: bool = Field(default=False)
    vomiting: bool = Field(default=False)
    dizziness: bool = Field(default=False)
    palpitations: bool = Field(default=False)
    fatigue: bool = Field(default=False)
    syncope: bool = Field(default=False)
    fever: bool = Field(default=False)
    leg_swelling: bool = Field(default=False)
    
    def calculate_symptom_score(self) -> float:
        """Calculate weighted symptom score for urgency (0-1)"""
        # Symptom weights based on clinical significance
        weights = {
            'shortness_of_breath': 0.3,
            'sweating': 0.25,
            'nausea': 0.2,
            'syncope': 0.4,
            'dizziness': 0.15,
            'palpitations': 0.2,
            'fatigue': 0.1,
            'vomiting': 0.15,
            'fever': 0.1,
            'cough': 0.05,
            'wheeze': 0.1,
            'leg_swelling': 0.1
        }
        
        score = 0.0
        for symptom, weight in weights.items():
            if getattr(self, symptom, False):
                score += weight
        
        return min(score, 1.0)

# =============================================================================
# AI ASSESSMENT WITH COORDINATE MAPPING
# =============================================================================

class RiskFactorAssessment(BaseModel):
    """Risk factor scoring with numerical values"""
    model_config = ConfigDict(validate_default=True)
    
    age_risk_score: float = Field(..., ge=0, le=1, description="Age-related risk (0-1)")
    gender_risk_score: float = Field(..., ge=0, le=1, description="Gender-related risk (0-1)")
    smoking_risk_score: float = Field(default=0.0, ge=0, le=1, description="Smoking-related risk (0-1)")
    diabetes_risk_score: float = Field(default=0.0, ge=0, le=1, description="Diabetes-related risk (0-1)")
    hypertension_risk_score: float = Field(default=0.0, ge=0, le=1, description="Hypertension risk (0-1)")
    family_history_risk_score: float = Field(default=0.0, ge=0, le=1, description="Family history risk (0-1)")
    overall_cardiovascular_risk: float = Field(..., ge=0, le=1, description="Combined CV risk (0-1)")

class AIAssessment(BaseModel):
    """AI-generated medical assessment with coordinate mapping"""
    model_config = ConfigDict(validate_default=True)
    
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core triage coordinates
    triage_coordinate: TriageCoordinate = Field(..., description="X-Y triage coordinates")
    
    # Component scores
    symptom_severity_score: float = Field(..., ge=0, le=1, description="Symptom severity (0-1)")
    red_flags_score: float = Field(..., ge=0, le=1, description="Red flags severity (0-1)")
    vital_signs_score: float = Field(default=0.0, ge=0, le=1, description="Vital signs abnormality (0-1)")
    risk_factors: RiskFactorAssessment = Field(..., description="Risk factor analysis")
    
    # Clinical assessment
    differential_diagnoses: List[str] = Field(..., min_items=1, description="Possible conditions ranked by likelihood")
    red_flags_identified: List[str] = Field(default_factory=list, description="Red flag symptoms identified")
    
    # Disposition
    recommended_disposition: TriageDisposition = Field(..., description="Recommended care pathway")
    urgency_category: UrgencyCategory = Field(..., description="NHS urgency category")
    recommended_timeframe: str = Field(..., description="Timeframe for seeking care")
    
    # Clinical reasoning
    clinical_reasoning: str = Field(..., min_length=1, description="AI reasoning for assessment")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    uncertainty_factors: List[str] = Field(default_factory=list, description="Uncertainty factors")
    
    # Safety netting
    safety_netting_advice: List[str] = Field(..., min_items=1, description="Safety netting instructions")
    return_if_worse_advice: List[str] = Field(..., min_items=1, description="When to return for reassessment")
    
    # Metadata
    assessment_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = Field(default=None, ge=0)
    model_version: str = Field(..., description="AI model version used")
    
    def calculate_overall_priority_score(self, weights: Optional[RiskScoreWeights] = None) -> float:
        """Calculate weighted overall priority score (0-1)"""
        if weights is None:
            weights = RiskScoreWeights()
        
        score = (
            weights.symptom_severity_weight * self.symptom_severity_score +
            weights.red_flags_weight * self.red_flags_score +
            weights.vital_signs_weight * self.vital_signs_score +
            weights.medical_history_weight * self.risk_factors.overall_cardiovascular_risk
        )
        
        return min(score, 1.0)

# =============================================================================
# COMPREHENSIVE CASE REPORT
# =============================================================================

class CaseReport(BaseModel):
    """Comprehensive medical case report with ML scoring"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    # Case identification
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None)
    
    # Demographics and basic info
    demographics: Demographics = Field(...)
    
    # Chief complaint
    chief_complaint: str = Field(..., min_length=1, max_length=1000)
    presenting_complaint_category: str = Field(...)
    
    # Clinical data
    vital_signs: Optional[VitalSigns] = Field(default=None)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    
    # Disease-specific assessments
    chest_pain_assessment: Optional[ChestPainCharacteristics] = Field(default=None)
    chest_pain_red_flags: Optional[ChestPainRedFlags] = Field(default=None)
    associated_symptoms: Optional[ChestPainAssociatedSymptoms] = Field(default=None)
    
    # Data and responses
    patient_responses: List[PatientResponse] = Field(default_factory=list)
    uploaded_files: List[UploadedFile] = Field(default_factory=list)
    
    # AI analysis
    ai_assessment: Optional[AIAssessment] = Field(default=None)
    
    # Case management
    status: str = Field(default="created")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Quality metrics
    data_quality_score: Optional[float] = Field(default=None, ge=0, le=1)
    protocol_compliance: Optional[bool] = Field(default=None)
    clinical_review_required: bool = Field(default=False)
    
    # Report generation
    pdf_report_url: Optional[str] = Field(default=None)
    
    def calculate_composite_scores(self) -> Tuple[float, float, float]:
        """Calculate composite urgency, importance, and priority scores"""
        urgency_score = 0.0
        importance_score = 0.0
        
        # Age-based scoring
        urgency_score += self.demographics.get_age_risk_score() * 0.2
        importance_score += self.demographics.get_age_risk_score() * 0.3
        
        # Chest pain specific scoring
        if self.chest_pain_assessment:
            cardiac_risk = self.chest_pain_assessment.calculate_cardiac_risk_score()
            urgency_score += cardiac_risk * 0.4
            importance_score += cardiac_risk * 0.4
        
        # Red flags scoring
        if self.chest_pain_red_flags:
            red_flag_score = self.chest_pain_red_flags.get_red_flag_score()
            urgency_score += red_flag_score * 0.5
            importance_score += red_flag_score * 0.3
        
        # Associated symptoms scoring
        if self.associated_symptoms:
            symptom_score = self.associated_symptoms.calculate_symptom_score()
            urgency_score += symptom_score * 0.3
            importance_score += symptom_score * 0.2
        
        # Vital signs scoring
        if self.vital_signs:
            vital_score = self.vital_signs.calculate_vital_signs_score()
            urgency_score += vital_score * 0.4
            importance_score += vital_score * 0.2
        
        # Medical history scoring
        history_score = self.medical_history.calculate_risk_score()
        urgency_score += history_score * 0.2
        importance_score += history_score * 0.3
        
        # Normalize scores to [-1, 1] range
        urgency_normalized = min(max((urgency_score * 2) - 1, -1), 1)
        importance_normalized = min(max((importance_score * 2) - 1, -1), 1)
        
        # Calculate priority score (0-1)
        priority_score = (urgency_score + importance_score) / 2
        
        return urgency_normalized, importance_normalized, priority_score

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_triage_coordinate(urgency: float, importance: float) -> TriageCoordinate:
    """Create triage coordinate with validation"""
    return TriageCoordinate(urgency=urgency, importance=importance)

def calculate_cardiovascular_risk_score(demographics: Demographics, medical_history: MedicalHistory) -> float:
    """Calculate comprehensive cardiovascular risk score"""
    risk_score = 0.0
    
    # Age risk (using demographics method)
    risk_score += demographics.get_age_risk_score() * 0.3
    
    # Medical history risk
    risk_score += medical_history.calculate_risk_score() * 0.7
    
    return min(risk_score, 1.0)

# =============================================================================
# EXPORT MODELS
# =============================================================================


__all__ = [
    # Enums with ML scoring
    "SeverityLevel", "UrgencyCategory", "ImportanceLevel", "TriageDisposition", "GenderType", "QuestionType",
    
    # Coordinate system
    "TriageCoordinate", "RiskScoreWeights",
    
    # Base models
    "Demographics", "VitalSigns", "MedicalHistory", "UploadedFile",
    
    # Protocol models
    "NICEProtocolQuestion", "PatientResponse",
    
    # Disease-specific models
    "ChestPainCharacteristics", "ChestPainRedFlags", "ChestPainAssociatedSymptoms",
    
    # Assessment models
    "RiskFactorAssessment", "AIAssessment",
    
    # Main case model
    "CaseReport",
    
    # Utility functions
    "create_triage_coordinate", "calculate_cardiovascular_risk_score"
]
