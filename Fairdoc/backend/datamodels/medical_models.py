from pydantic import BaseModel, Field, validator, ConfigDict, EmailStr
from typing import Optional, List, Dict, Union, Literal, Any
from datetime import datetime, date, time
from enum import Enum
import re
from uuid import UUID, uuid4

# NHS-specific validation patterns
NHS_NUMBER_PATTERN = r"^\d{10}$"
SNOMED_CODE_PATTERN = r"^\d+$"
ICD10_CODE_PATTERN = r"^[A-Z]\d{2}(\.\d{1,2})?$"
POSTCODE_PATTERN = r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$"

class NHSBaseModel(BaseModel):
    """Base model for all NHS EHR data with common configurations"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid',  # Security: prevent additional fields
        frozen=False,  # Allow updates for CRUD operations
        validate_default=True
    )

# Enums for controlled vocabularies
class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"

class EthnicityEnum(str, Enum):
    WHITE_BRITISH = "A"
    WHITE_IRISH = "B"
    WHITE_OTHER = "C"
    MIXED_WHITE_BLACK_CARIBBEAN = "D"
    MIXED_WHITE_BLACK_AFRICAN = "E"
    MIXED_WHITE_ASIAN = "F"
    MIXED_OTHER = "G"
    ASIAN_INDIAN = "H"
    ASIAN_PAKISTANI = "J"
    ASIAN_BANGLADESHI = "K"
    ASIAN_OTHER = "L"
    BLACK_CARIBBEAN = "M"
    BLACK_AFRICAN = "N"
    BLACK_OTHER = "P"
    CHINESE = "R"
    OTHER = "S"
    NOT_STATED = "Z"

class ConsultationTypeEnum(str, Enum):
    FACE_TO_FACE = "face_to_face"
    TELEPHONE = "telephone"
    VIDEO = "video"
    EMAIL = "email"
    HOME_VISIT = "home_visit"
    EMERGENCY = "emergency"

class PriorityEnum(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    CRITICAL = "critical"

# Core Patient Model
class PatientModel(NHSBaseModel):
    """NHS Patient record model following NHS Digital standards"""
    
    # Core identifiers
    patient_id: UUID = Field(default_factory=uuid4, description="Internal patient identifier")
    nhs_number: str = Field(..., pattern=NHS_NUMBER_PATTERN, description="10-digit NHS number")
    
    # Demographics (NHS Digital requirements)
    title: Optional[str] = Field(None, max_length=10)
    first_name: str = Field(..., min_length=1, max_length=100)
    middle_names: Optional[str] = Field(None, max_length=200)
    surname: str = Field(..., min_length=1, max_length=100)
    
    # NHS-specific demographic fields
    date_of_birth: date = Field(..., description="Patient date of birth")
    date_of_death: Optional[date] = Field(None, description="Date of death if applicable")
    gender: GenderEnum = Field(..., description="Patient gender")
    ethnicity: Optional[EthnicityEnum] = Field(None, description="NHS ethnicity code")
    
    # Contact information
    address_line_1: str = Field(..., min_length=1, max_length=100)
    address_line_2: Optional[str] = Field(None, max_length=100)
    city: str = Field(..., min_length=1, max_length=50)
    postcode: str = Field(..., pattern=POSTCODE_PATTERN)
    
    # Communication
    phone_primary: Optional[str] = Field(None, pattern=r"^(\+44|0)[0-9\s-]{10,15}$")
    phone_mobile: Optional[str] = Field(None, pattern=r"^(\+44|0)[0-9\s-]{10,15}$")
    email: Optional[EmailStr] = Field(None)
    preferred_language: str = Field(default="English", max_length=50)
    
    # Registration details
    gp_practice_code: str = Field(..., pattern=r"^[A-Z0-9]{6}$", description="6-character GP practice code")
    registration_date: date = Field(default_factory=date.today)
    registration_status: Literal["active", "inactive", "deceased", "transferred"] = "active"
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(..., description="Healthcare professional who created record")
    
    @validator('date_of_birth')
    def validate_birth_date(cls, v):
        if v >= date.today():
            raise ValueError('Date of birth must be in the past')
        if v < date(1900, 1, 1):
            raise ValueError('Date of birth cannot be before 1900')
        return v
    
    @validator('date_of_death')
    def validate_death_date(cls, v, values):
        if v is not None:
            if 'date_of_birth' in values and v <= values['date_of_birth']:
                raise ValueError('Date of death must be after date of birth')
            if v > date.today():
                raise ValueError('Date of death cannot be in the future')
        return v
    
    @validator('nhs_number')
    def validate_nhs_number_checksum(cls, v):
        """Validate NHS number using the modulus 11 algorithm"""
        if len(v) != 10 or not v.isdigit():
            raise ValueError('NHS number must be 10 digits')
        
        # NHS number validation algorithm
        check_digit = int(v[9])
        total = sum(int(v[i]) * (10 - i) for i in range(9))
        remainder = total % 11
        
        if remainder < 2:
            expected_check = 0
        else:
            expected_check = 11 - remainder
        
        if check_digit != expected_check:
            raise ValueError('Invalid NHS number checksum')
        
        return v

# Clinical Coding Models
class SNOMEDConcept(NHSBaseModel):
    """SNOMED CT concept for clinical coding"""
    concept_id: str = Field(..., pattern=SNOMED_CODE_PATTERN)
    display: str = Field(..., min_length=1, max_length=500)
    system: str = Field(default="http://snomed.info/sct")

class ICD10Code(NHSBaseModel):
    """ICD-10 coding for diagnoses"""
    code: str = Field(..., pattern=ICD10_CODE_PATTERN)
    display: str = Field(..., min_length=1, max_length=200)
    system: str = Field(default="http://hl7.org/fhir/sid/icd-10")

# Clinical Observation Model
class ClinicalObservation(NHSBaseModel):
    """Clinical observation following NHS standards"""
    
    observation_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID = Field(..., description="Reference to patient")
    
    # Clinical coding
    snomed_concept: SNOMEDConcept = Field(..., description="SNOMED CT concept")
    
    # Observation details
    value_quantity: Optional[float] = Field(None, description="Numerical value")
    value_unit: Optional[str] = Field(None, max_length=50, description="Unit of measurement")
    value_string: Optional[str] = Field(None, max_length=1000, description="String value")
    value_boolean: Optional[bool] = Field(None, description="Boolean value")
    value_date: Optional[date] = Field(None, description="Date value")
    
    # Reference ranges
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    
    # Status and metadata
    status: Literal["preliminary", "final", "amended", "cancelled"] = "final"
    category: str = Field(..., max_length=100, description="Observation category")
    effective_datetime: datetime = Field(default_factory=datetime.now)
    
    # Clinical context
    body_site: Optional[SNOMEDConcept] = None
    method: Optional[SNOMEDConcept] = None
    
    # Audit trail
    recorded_by: str = Field(..., description="Healthcare professional")
    recorded_at: datetime = Field(default_factory=datetime.now)
    
    @validator('value_quantity')
    def validate_value_within_range(cls, v, values):
        if v is not None:
            if 'reference_range_low' in values and values['reference_range_low'] is not None:
                if v < values['reference_range_low'] * 0.1:  # Allow 90% below normal
                    raise ValueError('Value appears to be outside reasonable clinical range')
            if 'reference_range_high' in values and values['reference_range_high'] is not None:
                if v > values['reference_range_high'] * 10:  # Allow 10x above normal
                    raise ValueError('Value appears to be outside reasonable clinical range')
        return v

# Medication Model
class MedicationModel(NHSBaseModel):
    """NHS medication record model"""
    
    medication_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID = Field(..., description="Reference to patient")
    
    # Drug identification (dm+d codes - Dictionary of Medicines and Devices)
    dmd_code: str = Field(..., pattern=r"^\d+$", description="dm+d concept code")
    medication_name: str = Field(..., min_length=1, max_length=200)
    
    # Prescription details
    dose_quantity: float = Field(..., gt=0, description="Dose amount")
    dose_unit: str = Field(..., max_length=50, description="Dose unit (mg, ml, etc.)")
    frequency: str = Field(..., max_length=100, description="Dosing frequency")
    route: str = Field(..., max_length=50, description="Route of administration")
    
    # Prescription period
    start_date: date = Field(..., description="Start date of prescription")
    end_date: Optional[date] = Field(None, description="End date of prescription")
    quantity_prescribed: Optional[int] = Field(None, gt=0, description="Total quantity prescribed")
    
    # Clinical context
    indication: Optional[SNOMEDConcept] = Field(None, description="Reason for prescription")
    prescriber_name: str = Field(..., min_length=1, max_length=100)
    prescriber_gmc_number: Optional[str] = Field(None, pattern=r"^\d{7}$")
    
    # Safety
    allergies_checked: bool = Field(default=True, description="Allergies checked before prescribing")
    interactions_checked: bool = Field(default=True, description="Drug interactions checked")
    
    # Audit
    prescribed_at: datetime = Field(default_factory=datetime.now)
    last_review_date: Optional[date] = None
    
    @validator('end_date')
    def validate_end_after_start(cls, v, values):
        if v is not None and 'start_date' in values:
            if v <= values['start_date']:
                raise ValueError('End date must be after start date')
        return v

# Consultation Model
class ConsultationModel(NHSBaseModel):
    """NHS consultation record model"""
    
    consultation_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID = Field(..., description="Reference to patient")
    
    # Consultation details
    consultation_type: ConsultationTypeEnum = Field(..., description="Type of consultation")
    appointment_datetime: datetime = Field(..., description="Scheduled appointment time")
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    
    # Location and participants
    location: str = Field(..., max_length=200, description="Consultation location")
    consulting_practitioner: str = Field(..., min_length=1, max_length=100)
    practitioner_gmc_number: Optional[str] = Field(None, pattern=r"^\d{7}$")
    chaperone_present: bool = Field(default=False)
    chaperone_name: Optional[str] = Field(None, max_length=100)
    
    # Clinical content
    chief_complaint: str = Field(..., min_length=1, max_length=1000, description="Main presenting complaint")
    history_of_present_illness: Optional[str] = Field(None, max_length=5000)
    past_medical_history: Optional[str] = Field(None, max_length=2000)
    medications_reviewed: bool = Field(default=True)
    allergies_reviewed: bool = Field(default=True)
    
    # Assessment and plan
    clinical_assessment: Optional[str] = Field(None, max_length=2000)
    diagnosis_codes: List[Union[SNOMEDConcept, ICD10Code]] = Field(default_factory=list)
    management_plan: Optional[str] = Field(None, max_length=2000)
    
    # Follow-up
    follow_up_required: bool = Field(default=False)
    follow_up_timeframe: Optional[str] = Field(None, max_length=100)
    referrals_made: List[str] = Field(default_factory=list, description="Referrals to other services")
    
    # Information sharing
    information_given_to_patient: Optional[str] = Field(None, max_length=1000)
    patient_understanding_confirmed: bool = Field(default=True)
    
    # Refusals and preferences
    treatments_refused: List[str] = Field(default_factory=list)
    patient_preferences: Optional[str] = Field(None, max_length=1000)
    
    # Priority and status
    priority: PriorityEnum = Field(default=PriorityEnum.ROUTINE)
    consultation_status: Literal["scheduled", "in_progress", "completed", "cancelled", "no_show"] = "scheduled"
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('actual_end_time')
    def validate_end_after_start(cls, v, values):
        if v is not None and 'actual_start_time' in values and values['actual_start_time'] is not None:
            if v <= values['actual_start_time']:
                raise ValueError('End time must be after start time')
        return v

# Allergy and Adverse Reaction Model
class AllergyModel(NHSBaseModel):
    """NHS allergy and adverse reaction model"""
    
    allergy_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID = Field(..., description="Reference to patient")
    
    # Allergen details
    allergen: SNOMEDConcept = Field(..., description="Substance causing allergy/reaction")
    category: Literal["food", "medication", "environment", "biologic", "other"] = Field(..., description="Allergy category")
    
    # Clinical details
    reaction_type: Literal["allergy", "intolerance", "adverse_reaction"] = Field(..., description="Type of reaction")
    severity: Literal["mild", "moderate", "severe", "life_threatening"] = Field(..., description="Reaction severity")
    manifestation: List[SNOMEDConcept] = Field(..., min_items=1, description="Clinical manifestations")
    
    # Onset and timing
    onset_date: Optional[date] = Field(None, description="Date of first reaction")
    last_occurrence: Optional[date] = Field(None, description="Date of last reaction")
    
    # Clinical context
    verification_status: Literal["unconfirmed", "confirmed", "refuted", "entered_in_error"] = "unconfirmed"
    criticality: Literal["low", "high", "unable_to_assess"] = "unable_to_assess"
    
    # Additional details
    notes: Optional[str] = Field(None, max_length=1000, description="Additional clinical notes")
    
    # Audit
    recorded_by: str = Field(..., description="Healthcare professional")
    recorded_at: datetime = Field(default_factory=datetime.now)
    
    @validator('last_occurrence')
    def validate_last_occurrence(cls, v, values):
        if v is not None:
            if 'onset_date' in values and values['onset_date'] is not None:
                if v < values['onset_date']:
                    raise ValueError('Last occurrence cannot be before onset date')
            if v > date.today():
                raise ValueError('Last occurrence cannot be in the future')
        return v

# Complete EHR Record Container
class ElectronicHealthRecord(NHSBaseModel):
    """Complete NHS Electronic Health Record container"""
    
    record_id: UUID = Field(default_factory=uuid4, description="Unique EHR record identifier")
    patient: PatientModel = Field(..., description="Patient demographics and registration")
    
    # Clinical data collections
    observations: List[ClinicalObservation] = Field(default_factory=list, description="Clinical observations and measurements")
    medications: List[MedicationModel] = Field(default_factory=list, description="Current and historical medications")
    allergies: List[AllergyModel] = Field(default_factory=list, description="Known allergies and adverse reactions")
    consultations: List[ConsultationModel] = Field(default_factory=list, description="Consultation history")
    
    # Record metadata
    record_version: str = Field(default="1.0", description="EHR record version")
    last_updated: datetime = Field(default_factory=datetime.now)
    data_source: str = Field(..., description="Source system or practice")
    
    # Permissions and access
    access_level: Literal["full", "summary", "emergency_only", "restricted"] = "full"
    sharing_preferences: Dict[str, bool] = Field(
        default_factory=lambda: {
            "share_with_secondary_care": True,
            "share_with_social_care": False,
            "share_for_research": False,
            "share_for_planning": True
        }
    )
    
    @validator('observations', 'medications', 'allergies', 'consultations')
    def validate_patient_references(cls, v, values):
        """Ensure all records reference the same patient"""
        if 'patient' in values:
            patient_id = values['patient'].patient_id
            for item in v:
                if hasattr(item, 'patient_id') and item.patient_id != patient_id:
                    raise ValueError('All records must reference the same patient')
        return v

# CRUD Operation Models
class EHRCreateRequest(NHSBaseModel):
    """Request model for creating new EHR records"""
    patient: PatientModel
    initial_consultation: Optional[ConsultationModel] = None
    
class EHRUpdateRequest(NHSBaseModel):
    """Request model for updating EHR records"""
    record_id: UUID
    updates: Dict[str, Any] = Field(..., description="Fields to update")
    updated_by: str = Field(..., description="Healthcare professional making update")
    reason_for_update: str = Field(..., max_length=500)

class EHRSearchRequest(NHSBaseModel):
    """Request model for searching EHR records"""
    nhs_number: Optional[str] = Field(None, pattern=NHS_NUMBER_PATTERN)
    patient_id: Optional[UUID] = None
    surname: Optional[str] = Field(None, min_length=2, max_length=100)
    date_of_birth: Optional[date] = None
    postcode: Optional[str] = Field(None, pattern=POSTCODE_PATTERN)
    
    @validator('*')
    def at_least_one_search_criteria(cls, v, values):
        """Ensure at least one search criterion is provided"""
        if not any(values.values()) and v is None:
            raise ValueError('At least one search criterion must be provided')
        return v

# Error Response Model
class EHRErrorResponse(NHSBaseModel):
    """Standardized error response model"""
    error_code: str = Field(..., description="NHS-specific error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: UUID = Field(default_factory=uuid4, description="For error tracking")

# Success Response Model
class EHRSuccessResponse(NHSBaseModel):
    """Standardized success response model"""
    success: bool = Field(default=True)
    message: str = Field(..., description="Success message")
    data: Optional[Union[ElectronicHealthRecord, List[ElectronicHealthRecord], Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
