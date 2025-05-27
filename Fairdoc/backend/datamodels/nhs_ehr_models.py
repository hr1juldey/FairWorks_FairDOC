"""
NHS-specific Electronic Health Record models for Fairdoc Medical AI Backend.
Implements NHS Digital standards, FHIR R4 compliance, and GP Connect specifications.
Properly uses UUID and uuid4 for all identifier fields.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin, UUIDMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity
)

# ============================================================================
# NHS-SPECIFIC ENUMS
# ============================================================================

class NHSDataStandard(str, Enum):
    """NHS digital standards compliance."""
    FHIR_R4 = "fhir_r4"
    GP_CONNECT = "gp_connect"
    NHS_DIGITAL = "nhs_digital"
    SNOMED_CT = "snomed_ct"
    ICD10 = "icd10"
    OPCS4 = "opcs4"

class NHSOrganisationType(str, Enum):
    """NHS organisation types."""
    GP_PRACTICE = "gp_practice"
    HOSPITAL_TRUST = "hospital_trust"
    FOUNDATION_TRUST = "foundation_trust"
    CLINICAL_COMMISSIONING_GROUP = "ccg"
    INTEGRATED_CARE_SYSTEM = "ics"
    AMBULANCE_TRUST = "ambulance_trust"
    MENTAL_HEALTH_TRUST = "mental_health_trust"

class NHSRecordType(str, Enum):
    """Types of NHS health records."""
    GP_SUMMARY = "gp_summary"
    HOSPITAL_EPISODE = "hospital_episode"
    OUTPATIENT_APPOINTMENT = "outpatient_appointment"
    EMERGENCY_ATTENDANCE = "emergency_attendance"
    PRESCRIPTION = "prescription"
    PATHOLOGY_RESULT = "pathology_result"
    RADIOLOGY_REPORT = "radiology_report"
    DISCHARGE_SUMMARY = "discharge_summary"

class NHSSystemStatus(str, Enum):
    """NHS system integration status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SYNCHRONIZING = "synchronizing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

# ============================================================================
# NHS ORGANISATION MODELS
# ============================================================================

class NHSOrganisation(BaseEntity, ValidationMixin, MetadataMixin):
    """NHS organisation details for healthcare provider identification."""
    
    # UUID identifiers using uuid4
    organisation_uuid: UUID = Field(default_factory=uuid4, description="Internal UUID for organisation")
    
    # Official NHS identifiers
    ods_code: str = Field(..., pattern=r"^[A-Z0-9]{3,6}$", description="NHS ODS organisation code")
    organisation_name: str = Field(..., max_length=200, description="Official organisation name")
    organisation_type: NHSOrganisationType
    
    # Contact information
    address_line1: str = Field(..., max_length=100)
    address_line2: Optional[str] = Field(None, max_length=100)
    city: str = Field(..., max_length=50)
    county: Optional[str] = Field(None, max_length=50)
    postcode: str = Field(..., pattern=r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$")
    
    # Digital connectivity
    gp_connect_enabled: bool = Field(default=False)
    fhir_endpoint: Optional[str] = Field(None, description="FHIR R4 endpoint URL")
    last_sync: Optional[datetime] = None
    sync_status: NHSSystemStatus = Field(default=NHSSystemStatus.DISCONNECTED)
    
    # Service capabilities
    services_offered: List[str] = Field(default_factory=list)
    specialties: List[str] = Field(default_factory=list)
    emergency_services: bool = Field(default=False)
    
    # Patient population
    registered_patients: Optional[int] = Field(None, ge=0)
    catchment_population: Optional[int] = Field(None, ge=0)
    
    # Related UUIDs
    parent_organisation_id: Optional[UUID] = Field(None, description="UUID of parent organisation")
    child_organisation_ids: List[UUID] = Field(default_factory=list, description="UUIDs of child organisations")
    
    @field_validator('ods_code')
    @classmethod
    def validate_ods_code(cls, v: str) -> str:
        """Validate NHS ODS code format."""
        if not v or len(v) < 3 or len(v) > 6:
            raise ValueError('ODS code must be 3-6 characters')
        return v.upper()
    
    def update_sync_status(self, status: NHSSystemStatus):
        """Update synchronization status with timestamp."""
        self.sync_status = status
        if status == NHSSystemStatus.CONNECTED:
            self.last_sync = datetime.utcnow()
        self.update_timestamp()
    
    def add_child_organisation(self, child_organisation_id: UUID):
        """Add child organisation UUID."""
        if child_organisation_id not in self.child_organisation_ids:
            self.child_organisation_ids.append(child_organisation_id)
            self.update_timestamp()

# ============================================================================
# PATIENT HEALTH RECORD MODELS
# ============================================================================

class NHSPatientRecord(BaseEntity, ValidationMixin, MetadataMixin):
    """Comprehensive NHS patient health record with UUID identifiers."""
    
    # UUID identifiers using uuid4
    patient_uuid: UUID = Field(default_factory=uuid4, description="Internal UUID for patient")
    
    # NHS patient identification
    nhs_number: str = Field(..., pattern=r"^\d{10}$", description="10-digit NHS number")
    local_patient_id: str = Field(..., description="Local system patient identifier")
    
    # Demographics
    family_name: str = Field(..., max_length=100)
    given_names: List[str] = Field(..., min_items=1)
    date_of_birth: datetime
    gender: Gender
    ethnicity: Ethnicity
    
    # Address and contact
    current_address: Dict[str, str] = Field(..., description="Current patient address")
    previous_addresses: List[Dict[str, str]] = Field(default_factory=list)
    contact_numbers: List[str] = Field(default_factory=list)
    email_address: Optional[str] = None
    
    # NHS registration with UUID references
    registered_gp_practice_uuid: UUID = Field(..., description="UUID of registered GP practice")
    registered_gp_practice_ods: str = Field(..., description="ODS code of registered GP practice")
    registration_date: datetime
    registration_status: Literal["active", "inactive", "temporary"] = "active"
    
    # Emergency contacts with UUIDs
    emergency_contacts: List[Dict[str, Any]] = Field(default_factory=list)
    next_of_kin_uuid: Optional[UUID] = Field(None, description="UUID of next of kin if they're also a patient")
    next_of_kin: Optional[Dict[str, str]] = None
    
    # Medical identifiers with organisation UUIDs
    hospital_numbers: Dict[UUID, str] = Field(default_factory=dict, description="Hospital UUID to patient number mapping")
    
    # Care team UUIDs
    primary_care_team_uuids: List[UUID] = Field(default_factory=list, description="UUIDs of primary care team members")
    specialist_care_team_uuids: List[UUID] = Field(default_factory=list, description="UUIDs of specialist care team members")
    
    # Consent and preferences
    data_sharing_consent: bool = Field(default=False)
    research_consent: bool = Field(default=False)
    communication_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('nhs_number')
    @classmethod
    def validate_nhs_number(cls, v: str) -> str:
        """Validate NHS number with check digit."""
        if len(v) != 10 or not v.isdigit():
            raise ValueError('NHS number must be 10 digits')
        
        # NHS number checksum validation
        total = sum(int(digit) * (10 - i) for i, digit in enumerate(v[:9]))
        remainder = total % 11
        check_digit = 11 - remainder if remainder != 0 else 0
        
        if check_digit == 10 or int(v[9]) != check_digit:
            raise ValueError('Invalid NHS number checksum')
        
        return v
    
    def get_display_name(self) -> str:
        """Get patient display name."""
        return f"{' '.join(self.given_names)} {self.family_name}"
    
    def add_hospital_number(self, hospital_uuid: UUID, patient_number: str):
        """Add hospital-specific patient number using UUID."""
        self.hospital_numbers[hospital_uuid] = patient_number
        self.update_timestamp()
    
    def add_care_team_member(self, team_member_uuid: UUID, is_specialist: bool = False):
        """Add care team member UUID."""
        if is_specialist:
            if team_member_uuid not in self.specialist_care_team_uuids:
                self.specialist_care_team_uuids.append(team_member_uuid)
        else:
            if team_member_uuid not in self.primary_care_team_uuids:
                self.primary_care_team_uuids.append(team_member_uuid)
        self.update_timestamp()

class ClinicalCode(TimestampMixin, ValidationMixin):
    """Clinical coding for diagnoses, procedures, and medications with UUID tracking."""
    
    # UUID for tracking code usage
    code_usage_uuid: UUID = Field(default_factory=uuid4, description="UUID for tracking this code usage")
    
    # Coding system
    coding_system: NHSDataStandard
    code: str = Field(..., description="Clinical code")
    display_term: str = Field(..., description="Human-readable term")
    
    # SNOMED CT specific
    concept_id: Optional[str] = Field(None, description="SNOMED CT concept ID")
    description_id: Optional[str] = Field(None, description="SNOMED CT description ID")
    
    # ICD-10 specific
    icd10_code: Optional[str] = Field(None, pattern=r"^[A-Z]\d{2}(\.[0-9X]{1,3})?$")
    
    # Validity and versioning
    effective_date: datetime = Field(default_factory=datetime.utcnow)
    expiry_date: Optional[datetime] = None
    version: str = Field(default="current")
    
    # Coded by UUID reference
    coded_by_clinician_uuid: Optional[UUID] = Field(None, description="UUID of clinician who assigned this code")
    
    @field_validator('code')
    @classmethod
    def validate_code_format(cls, v: str, info) -> str:
        """Validate code format based on coding system."""
        if hasattr(info, 'data') and 'coding_system' in info.data:
            coding_system = info.data['coding_system']
            
            if coding_system == NHSDataStandard.SNOMED_CT:
                if not v.isdigit() or len(v) < 6:
                    raise ValueError('SNOMED CT codes must be 6+ digits')
            elif coding_system == NHSDataStandard.ICD10:
                import re
                if not re.match(r"^[A-Z]\d{2}(\.[0-9X]{1,3})?$", v):
                    raise ValueError('Invalid ICD-10 code format')
        
        return v

# ============================================================================
# CLINICAL RECORD MODELS
# ============================================================================

class ClinicalEncounter(BaseEntity, ValidationMixin, MetadataMixin):
    """Clinical encounter/episode record with comprehensive UUID references."""
    
    # Encounter UUIDs
    encounter_uuid: UUID = Field(default_factory=uuid4, description="Unique encounter UUID")
    patient_uuid: UUID = Field(..., description="Patient UUID")
    
    # Legacy identifiers
    encounter_id: str = Field(..., description="Legacy encounter identifier")
    patient_nhs_number: str = Field(..., pattern=r"^\d{10}$")
    
    # Encounter details
    encounter_type: NHSRecordType
    encounter_date: datetime
    encounter_duration: Optional[timedelta] = None
    
    # Healthcare provider UUIDs
    organisation_uuid: UUID = Field(..., description="Provider organisation UUID")
    organisation_ods_code: str = Field(..., description="Provider organisation ODS code")
    department_uuid: Optional[UUID] = Field(None, description="Department UUID")
    consultant_uuid: Optional[UUID] = Field(None, description="Consultant UUID")
    attending_clinicians_uuids: List[UUID] = Field(default_factory=list, description="All attending clinician UUIDs")
    
    # Clinical content
    chief_complaint: Optional[str] = Field(None, max_length=500)
    history_of_present_illness: Optional[str] = Field(None, max_length=2000)
    examination_findings: Optional[str] = Field(None, max_length=2000)
    
    # Diagnoses and procedures with UUIDs
    primary_diagnosis_uuid: Optional[UUID] = Field(None, description="Primary diagnosis code UUID")
    secondary_diagnoses_uuids: List[UUID] = Field(default_factory=list, description="Secondary diagnosis code UUIDs")
    procedures_performed_uuids: List[UUID] = Field(default_factory=list, description="Procedure code UUIDs")
    
    # Medications
    medications_prescribed: List[Dict[str, Any]] = Field(default_factory=list)
    medications_administered: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Referrals and follow-up with UUIDs
    referrals_made_uuids: List[UUID] = Field(default_factory=list, description="Referral UUIDs")
    follow_up_required: bool = Field(default=False)
    follow_up_date: Optional[datetime] = None
    follow_up_clinician_uuid: Optional[UUID] = Field(None, description="Follow-up clinician UUID")
    
    # Administrative
    admission_method: Optional[str] = None
    discharge_method: Optional[str] = None
    discharge_destination: Optional[str] = None
    
    # Clinical risk assessment
    clinical_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    
    # Related encounters
    parent_encounter_uuid: Optional[UUID] = Field(None, description="Parent encounter UUID if this is a follow-up")
    child_encounter_uuids: List[UUID] = Field(default_factory=list, description="Child encounter UUIDs")
    
    def add_diagnosis(self, diagnosis_code_uuid: UUID, is_primary: bool = False):
        """Add diagnosis UUID to encounter."""
        if is_primary:
            self.primary_diagnosis_uuid = diagnosis_code_uuid
        else:
            if diagnosis_code_uuid not in self.secondary_diagnoses_uuids:
                self.secondary_diagnoses_uuids.append(diagnosis_code_uuid)
        self.update_timestamp()
    
    def add_attending_clinician(self, clinician_uuid: UUID):
        """Add attending clinician UUID."""
        if clinician_uuid not in self.attending_clinicians_uuids:
            self.attending_clinicians_uuids.append(clinician_uuid)
            self.update_timestamp()
    
    def add_referral(self, referral_uuid: UUID):
        """Add referral UUID."""
        if referral_uuid not in self.referrals_made_uuids:
            self.referrals_made_uuids.append(referral_uuid)
            self.update_timestamp()

class PathologyResult(BaseEntity, ValidationMixin, MetadataMixin):
    """Pathology/laboratory test results with UUID references."""
    
    # Test UUIDs
    test_uuid: UUID = Field(default_factory=uuid4, description="Unique test UUID")
    patient_uuid: UUID = Field(..., description="Patient UUID")
    
    # Legacy identifiers
    test_id: str = Field(..., description="Legacy test identifier")
    patient_nhs_number: str = Field(..., pattern=r"^\d{10}$")
    
    # Organisation UUIDs
    requesting_organisation_uuid: UUID = Field(..., description="UUID of requesting organisation")
    performing_lab_uuid: UUID = Field(..., description="UUID of performing laboratory")
    requesting_clinician_uuid: Optional[UUID] = Field(None, description="UUID of requesting clinician")
    
    # Test details
    test_type: str = Field(..., description="Type of pathology test")
    test_code_uuid: UUID = Field(..., description="UUID of test code record")
    specimen_type: str = Field(..., description="Specimen type (blood, urine, etc.)")
    specimen_uuid: UUID = Field(default_factory=uuid4, description="Unique specimen UUID")
    
    # Timing
    sample_collected_datetime: datetime
    result_available_datetime: datetime
    reported_datetime: Optional[datetime] = None
    
    # Results
    result_value: Optional[str] = None
    result_unit: Optional[str] = None
    reference_range: Optional[str] = None
    result_status: Literal["preliminary", "final", "corrected", "cancelled"] = "final"
    
    # Clinical interpretation
    abnormal_flag: Optional[Literal["low", "high", "abnormal", "critical"]] = None
    clinical_comment: Optional[str] = Field(None, max_length=1000)
    interpreted_by_clinician_uuid: Optional[UUID] = Field(None, description="UUID of interpreting clinician")
    
    # Risk assessment
    result_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    requires_urgent_action: bool = Field(default=False)
    
    # Related tests
    related_test_uuids: List[UUID] = Field(default_factory=list, description="Related test UUIDs")
    
    def calculate_risk_level(self) -> RiskLevel:
        """Calculate risk level based on result."""
        if self.abnormal_flag == "critical":
            return RiskLevel.CRITICAL
        elif self.abnormal_flag in ["high", "low"]:
            return RiskLevel.MODERATE
        elif self.abnormal_flag == "abnormal":
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def add_related_test(self, related_test_uuid: UUID):
        """Add related test UUID."""
        if related_test_uuid not in self.related_test_uuids:
            self.related_test_uuids.append(related_test_uuid)
            self.update_timestamp()

class PrescriptionRecord(BaseEntity, ValidationMixin, MetadataMixin):
    """Electronic prescription record with UUID references."""
    
    # Prescription UUIDs
    prescription_uuid: UUID = Field(default_factory=uuid4, description="Unique prescription UUID")
    patient_uuid: UUID = Field(..., description="Patient UUID")
    
    # Legacy identifiers
    prescription_id: str = Field(..., description="Legacy prescription identifier")
    patient_nhs_number: str = Field(..., pattern=r"^\d{10}$")
    
    # Prescriber UUIDs
    prescriber_uuid: UUID = Field(..., description="Prescriber UUID")
    prescriber_code: str = Field(..., description="Prescriber professional code")
    prescribing_organisation_uuid: UUID = Field(..., description="Prescribing organisation UUID")
    
    # Medication details
    medication_code_uuid: UUID = Field(..., description="UUID of medication code record")
    medication_name: str = Field(..., max_length=200)
    form: str = Field(..., description="Medication form (tablet, capsule, etc.)")
    strength: str = Field(..., description="Medication strength")
    
    # Dosage instructions
    dosage_instruction: str = Field(..., max_length=500)
    quantity_prescribed: str = Field(..., description="Quantity to dispense")
    duration_of_treatment: Optional[timedelta] = None
    
    # Prescription dates
    prescription_date: datetime = Field(default_factory=datetime.utcnow)
    prescription_effective_date: datetime
    prescription_expiry_date: datetime
    
    # Dispensing with UUIDs
    dispensed: bool = Field(default=False)
    dispensed_date: Optional[datetime] = None
    dispensing_organisation_uuid: Optional[UUID] = Field(None, description="Dispensing organisation UUID")
    dispensing_pharmacist_uuid: Optional[UUID] = Field(None, description="Dispensing pharmacist UUID")
    
    # Clinical context
    indication: Optional[str] = Field(None, max_length=200)
    contraindications_checked: bool = Field(default=True)
    drug_interactions_checked: bool = Field(default=True)
    allergy_checked: bool = Field(default=True)
    
    # Repeat prescription
    is_repeat: bool = Field(default=False)
    repeat_number: Optional[int] = Field(None, ge=1)
    remaining_repeats: Optional[int] = Field(None, ge=0)
    original_prescription_uuid: Optional[UUID] = Field(None, description="Original prescription UUID for repeats")
    
    # Related prescriptions
    related_prescription_uuids: List[UUID] = Field(default_factory=list, description="Related prescription UUIDs")
    
    def mark_as_dispensed(self, dispensing_org_uuid: UUID, pharmacist_uuid: UUID):
        """Mark prescription as dispensed with UUID references."""
        self.dispensed = True
        self.dispensed_date = datetime.utcnow()
        self.dispensing_organisation_uuid = dispensing_org_uuid
        self.dispensing_pharmacist_uuid = pharmacist_uuid
        self.update_timestamp()
    
    def add_related_prescription(self, related_prescription_uuid: UUID):
        """Add related prescription UUID."""
        if related_prescription_uuid not in self.related_prescription_uuids:
            self.related_prescription_uuids.append(related_prescription_uuid)
            self.update_timestamp()

# ============================================================================
# EHR INTEGRATION MODELS
# ============================================================================

class FHIRResource(TimestampMixin, ValidationMixin):
    """FHIR R4 resource wrapper for NHS integration with UUID tracking."""
    
    # UUID identifiers
    fhir_resource_uuid: UUID = Field(default_factory=uuid4, description="Internal UUID for FHIR resource")
    patient_uuid: Optional[UUID] = Field(None, description="Associated patient UUID")
    organisation_uuid: Optional[UUID] = Field(None, description="Source organisation UUID")
    
    # FHIR resource identification
    resource_type: str = Field(..., description="FHIR resource type")
    resource_id: str = Field(..., description="FHIR resource ID")
    version_id: Optional[str] = None
    
    # Resource content
    fhir_content: Dict[str, Any] = Field(..., description="Complete FHIR resource JSON")
    
    # NHS-specific extensions
    nhs_extensions: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    source_system: str = Field(..., description="Source NHS system")
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    
    # Validation status
    fhir_valid: bool = Field(default=True)
    validation_errors: List[str] = Field(default_factory=list)
    
    # Related resources
    related_fhir_resource_uuids: List[UUID] = Field(default_factory=list, description="Related FHIR resource UUIDs")
    
    def validate_fhir_structure(self) -> bool:
        """Validate FHIR resource structure."""
        required_fields = ["resourceType", "id"]
        
        for field in required_fields:
            if field not in self.fhir_content:
                self.validation_errors.append(f"Missing required field: {field}")
                self.fhir_valid = False
        
        return self.fhir_valid
    
    def add_related_resource(self, related_resource_uuid: UUID):
        """Add related FHIR resource UUID."""
        if related_resource_uuid not in self.related_fhir_resource_uuids:
            self.related_fhir_resource_uuids.append(related_resource_uuid)
            self.update_timestamp()

class GPConnectRecord(BaseEntity, ValidationMixin, MetadataMixin):
    """GP Connect API record for primary care integration with UUID references."""
    
    # GP Connect UUIDs
    gp_connect_record_uuid: UUID = Field(default_factory=uuid4, description="Unique GP Connect record UUID")
    patient_uuid: UUID = Field(..., description="Patient UUID")
    practice_uuid: UUID = Field(..., description="GP practice UUID")
    
    # Legacy identifiers
    gp_connect_id: str = Field(..., description="GP Connect interaction ID")
    patient_nhs_number: str = Field(..., pattern=r"^\d{10}$")
    practice_ods_code: str = Field(..., description="GP practice ODS code")
    
    # Record type and content
    record_type: Literal["structured", "documents", "appointments"] = "structured"
    content: Dict[str, Any] = Field(..., description="GP Connect formatted content")
    
    # API interaction details
    api_version: str = Field(default="1.2.7", description="GP Connect API version")
    interaction_id: str = Field(..., description="Spine interaction ID")
    
    # Timestamps
    retrieved_datetime: datetime = Field(default_factory=datetime.utcnow)
    last_updated_in_source: datetime
    
    # Data quality
    completeness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    data_quality_issues: List[str] = Field(default_factory=list)
    
    # Security and audit
    access_granted_by_uuid: UUID = Field(..., description="UUID of person who granted access")
    access_purpose: str = Field(..., description="Purpose for accessing this record")
    retention_period: timedelta = Field(default=timedelta(days=2555))  # 7 years default
    
    # Related records
    related_gp_records_uuids: List[UUID] = Field(default_factory=list, description="Related GP Connect record UUIDs")
    
    def add_related_record(self, related_record_uuid: UUID):
        """Add related GP Connect record UUID."""
        if related_record_uuid not in self.related_gp_records_uuids:
            self.related_gp_records_uuids.append(related_record_uuid)
            self.update_timestamp()

# ============================================================================
# EHR SYNCHRONIZATION AND AUDIT
# ============================================================================

class EHRSyncLog(BaseEntity, ValidationMixin):
    """Audit log for EHR synchronization activities with UUID tracking."""
    
    # Sync UUIDs
    sync_log_uuid: UUID = Field(default_factory=uuid4, description="Unique sync log UUID")
    source_organisation_uuid: UUID = Field(..., description="Source NHS organisation UUID")
    target_system_uuid: UUID = Field(..., description="Target system UUID")
    
    # Sync operation details
    sync_operation_id: str = Field(..., description="Legacy sync operation identifier")
    source_system: str = Field(..., description="Source NHS system")
    target_system: str = Field(default="fairdoc_ai", description="Target system")
    
    # Sync scope
    sync_type: Literal["full", "incremental", "patient_specific"] = "incremental"
    patient_uuids_synced: List[UUID] = Field(default_factory=list, description="Patient UUIDs synced")
    record_uuids_synced: List[UUID] = Field(default_factory=list, description="Record UUIDs synced")
    patient_count: int = Field(default=0, ge=0)
    record_count: int = Field(default=0, ge=0)
    
    # Timing
    sync_started: datetime = Field(default_factory=datetime.utcnow)
    sync_completed: Optional[datetime] = None
    sync_duration: Optional[timedelta] = None
    
    # Results
    sync_status: Literal["running", "completed", "failed", "partial"] = "running"
    successful_records: int = Field(default=0, ge=0)
    failed_records: int = Field(default=0, ge=0)
    failed_record_uuids: List[UUID] = Field(default_factory=list, description="UUIDs of failed records")
    error_details: List[str] = Field(default_factory=list)
    
    # Data integrity
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    integrity_verified: bool = Field(default=False)
    
    # Related sync logs
    related_sync_log_uuids: List[UUID] = Field(default_factory=list, description="Related sync log UUIDs")
    
    def complete_sync(self, success_count: int, failure_count: int, failed_uuids: List[UUID] = None):
        """Mark synchronization as completed with UUID tracking."""
        self.sync_completed = datetime.utcnow()
        self.sync_duration = self.sync_completed - self.sync_started
        self.successful_records = success_count
        self.failed_records = failure_count
        
        if failed_uuids:
            self.failed_record_uuids = failed_uuids
        
        if failure_count == 0:
            self.sync_status = "completed"
        elif success_count > 0:
            self.sync_status = "partial"
        else:
            self.sync_status = "failed"
        
        self.update_timestamp()
    
    def add_related_sync_log(self, related_sync_uuid: UUID):
        """Add related sync log UUID."""
        if related_sync_uuid not in self.related_sync_log_uuids:
            self.related_sync_log_uuids.append(related_sync_uuid)
            self.update_timestamp()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class NHSPatientResponse(BaseResponse):
    """Response for NHS patient record requests with UUID references."""
    patient_record: NHSPatientRecord
    clinical_encounters: List[ClinicalEncounter]
    recent_pathology: List[PathologyResult]
    current_medications: List[PrescriptionRecord]
    data_freshness: datetime
    
    # Related UUIDs for navigation
    care_team_uuids: List[UUID] = Field(default_factory=list)
    related_patient_uuids: List[UUID] = Field(default_factory=list)

class EHRIntegrationResponse(BaseResponse):
    """Response for EHR integration status with UUID tracking."""
    connected_organisations: List[NHSOrganisation]
    sync_status: Dict[UUID, NHSSystemStatus] = Field(default_factory=dict)
    last_sync_times: Dict[UUID, datetime] = Field(default_factory=dict)
    data_coverage: Dict[UUID, float] = Field(default_factory=dict)
    
    # Integration metrics by organisation UUID
    organisation_metrics: Dict[UUID, Dict[str, Any]] = Field(default_factory=dict)

class ClinicalDataResponse(BaseResponse):
    """Response for clinical data queries with UUID references."""
    fhir_resources: List[FHIRResource]
    gp_connect_records: List[GPConnectRecord]
    total_records: int
    data_quality_score: float
    
    # Data lineage UUIDs
    source_organisation_uuids: List[UUID] = Field(default_factory=list)
    related_patient_uuids: List[UUID] = Field(default_factory=list)

# ============================================================================
# NHS SYSTEM INTEGRATION UTILITIES
# ============================================================================


NHS_ORGANISATION_TYPES = {
    "GP": NHSOrganisationType.GP_PRACTICE,
    "TR": NHSOrganisationType.HOSPITAL_TRUST,
    "FT": NHSOrganisationType.FOUNDATION_TRUST,
    "CC": NHSOrganisationType.CLINICAL_COMMISSIONING_GROUP,
    "AM": NHSOrganisationType.AMBULANCE_TRUST,
    "MH": NHSOrganisationType.MENTAL_HEALTH_TRUST
}

FHIR_R4_RESOURCE_TYPES = [
    "Patient", "Encounter", "Observation", "Condition", "Procedure",
    "MedicationRequest", "DiagnosticReport", "DocumentReference",
    "Appointment", "ServiceRequest", "AllergyIntolerance"
]

GP_CONNECT_INTERACTIONS = {
    "structured_record": "urn:nhs:names:services:gpconnect:fhir:operation:gpc.getstructuredrecord-1",
    "documents": "urn:nhs:names:services:gpconnect:fhir:rest:search:documentreference-1",
    "appointments": "urn:nhs:names:services:gpconnect:fhir:rest:search:appointment-1"
}

def validate_nhs_number(nhs_number: str) -> bool:
    """Validate NHS number with checksum verification."""
    if len(nhs_number) != 10 or not nhs_number.isdigit():
        return False
    
    total = sum(int(digit) * (10 - i) for i, digit in enumerate(nhs_number[:9]))
    remainder = total % 11
    check_digit = 11 - remainder if remainder != 0 else 0
    
    return check_digit != 10 and int(nhs_number[9]) == check_digit

def map_ods_to_organisation_type(ods_code: str) -> Optional[NHSOrganisationType]:
    """Map ODS code to organisation type."""
    if not ods_code or len(ods_code) < 3:
        return None
    
    prefix = ods_code[:2].upper()
    return NHS_ORGANISATION_TYPES.get(prefix)

def generate_fhir_patient_resource(patient_record: NHSPatientRecord) -> Dict[str, Any]:
    """Generate FHIR R4 Patient resource from NHS patient record with UUID references."""
    return {
        "resourceType": "Patient",
        "id": patient_record.nhs_number,
        "identifier": [
            {
                "use": "official",
                "system": "https://fhir.nhs.uk/Id/nhs-number",
                "value": patient_record.nhs_number
            },
            {
                "use": "secondary",
                "system": "https://fairdoc.ai/patient-uuid",
                "value": str(patient_record.patient_uuid)
            }
        ],
        "name": [{
            "use": "official",
            "family": patient_record.family_name,
            "given": patient_record.given_names
        }],
        "gender": patient_record.gender.value.lower(),
        "birthDate": patient_record.date_of_birth.strftime("%Y-%m-%d"),
        "address": [patient_record.current_address],
        "generalPractitioner": [{
            "identifier": {
                "system": "https://fhir.nhs.uk/Id/ods-organization-code",
                "value": patient_record.registered_gp_practice_ods
            },
            "extension": [{
                "url": "https://fairdoc.ai/organisation-uuid",
                "valueString": str(patient_record.registered_gp_practice_uuid)
            }]
        }],
        "extension": [
            {
                "url": "https://fhir.nhs.uk/StructureDefinition/Extension-UKCore-EthnicCategory",
                "valueCodeableConcept": {
                    "coding": [{
                        "system": "https://fhir.nhs.uk/CodeSystem/UKCore-EthnicCategory",
                        "code": patient_record.ethnicity.value
                    }]
                }
            },
            {
                "url": "https://fairdoc.ai/patient-uuid",
                "valueString": str(patient_record.patient_uuid)
            }
        ]
    }

def create_uuid_mapping_table(records: List[BaseEntity]) -> Dict[str, UUID]:
    """Create mapping table from legacy IDs to UUIDs for migration."""
    mapping = {}
    for record in records:
        if hasattr(record, 'id') and hasattr(record, 'uuid'):
            mapping[str(record.id)] = record.uuid
    return mapping

def resolve_uuid_references(record_dict: Dict[str, Any], uuid_mapping: Dict[str, UUID]) -> Dict[str, Any]:
    """Resolve legacy ID references to UUIDs using mapping table."""
    resolved_record = record_dict.copy()
    
    # Common fields that might need UUID resolution
    uuid_fields = [
        'patient_id', 'organisation_id', 'clinician_id', 'encounter_id',
        'prescription_id', 'test_id', 'referral_id'
    ]
    
    for field in uuid_fields:
        if field in resolved_record and resolved_record[field] in uuid_mapping:
            uuid_field = field.replace('_id', '_uuid')
            resolved_record[uuid_field] = uuid_mapping[resolved_record[field]]
    
    return resolved_record
