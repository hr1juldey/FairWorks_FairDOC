"""
NHS-specific authentication data models for Fairdoc Medical AI Backend.
Enhanced with comprehensive timestamp tracking for authentication events and audit trails.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, EmailStr, field_validator, SecretStr
from enum import Enum

from .base_models import (
    BaseEntity, BaseResponse, TimestampMixin,
    ValidationMixin, MetadataMixin
)

# ============================================================================
# NHS USER TYPES AND ROLES
# ============================================================================

class NHSUserType(str, Enum):
    """NHS-specific user categories."""
    PUBLIC_PATIENT = "public_patient"          # Public users with NHS ID
    NHS_DOCTOR = "nhs_doctor"                  # Registered NHS doctors
    NHS_NURSE = "nhs_nurse"                    # Registered NHS nurses
    DEVELOPER = "developer"                    # System developers
    DATA_SCIENTIST = "data_scientist"          # ML engineers/data scientists
    SYSTEM_ADMIN = "system_admin"              # System administrators

class DoctorSpecialty(str, Enum):
    """Medical specialties for NHS doctors."""
    CARDIOLOGY = "cardiology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    INTERNAL_MEDICINE = "internal_medicine"
    FAMILY_MEDICINE = "family_medicine"
    CRITICAL_CARE = "critical_care"
    RADIOLOGY = "radiology"
    ANESTHESIOLOGY = "anesthesiology"
    SURGERY = "surgery"
    NEUROLOGY = "neurology"
    PSYCHIATRY = "psychiatry"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    OTHER = "other"

class DoctorSeniority(str, Enum):
    """NHS doctor seniority levels."""
    FOUNDATION_YEAR_1 = "fy1"                 # Foundation Year 1
    FOUNDATION_YEAR_2 = "fy2"                 # Foundation Year 2
    SPECIALTY_TRAINEE = "st"                  # Specialty Trainee (ST1-ST8)
    REGISTRAR = "registrar"                   # Specialty Registrar
    CONSULTANT = "consultant"                 # Consultant
    ASSOCIATE_SPECIALIST = "associate_spec"    # Associate Specialist
    STAFF_GRADE = "staff_grade"               # Staff Grade
    CLINICAL_FELLOW = "clinical_fellow"       # Clinical Fellow

class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_NHS_VERIFICATION = "pending_nhs_verification"
    PENDING_GMC_VERIFICATION = "pending_gmc_verification"
    LOCKED = "locked"
    NHS_VERIFICATION_FAILED = "nhs_verification_failed"

class PermissionScope(str, Enum):
    """System permission scopes."""
    # Patient data permissions
    PATIENT_OWN_READ = "patient:own:read"
    PATIENT_OWN_WRITE = "patient:own:write"
    PATIENT_ALL_READ = "patient:all:read"
    PATIENT_ALL_WRITE = "patient:all:write"
    
    # Medical records permissions
    MEDICAL_RECORDS_READ = "medical:records:read"
    MEDICAL_RECORDS_WRITE = "medical:records:write"
    MEDICAL_PRESCRIBE = "medical:prescribe"
    MEDICAL_DIAGNOSE = "medical:diagnose"
    
    # AI/ML permissions
    AI_MODEL_USE = "ai:model:use"
    AI_MODEL_TRAIN = "ai:model:train"
    AI_MODEL_DEPLOY = "ai:model:deploy"
    AI_BIAS_MONITOR = "ai:bias:monitor"
    AI_BIAS_ADJUST = "ai:bias:adjust"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # Data access permissions
    DATA_EXPORT_ANON = "data:export:anonymized"
    DATA_EXPORT_FULL = "data:export:full"
    DATA_RESEARCH = "data:research"

class TokenType(str, Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET_PASSWORD = "reset_password"
    EMAIL_VERIFICATION = "email_verification"

# ============================================================================
# AUTHENTICATION EVENT TRACKING WITH TIMESTAMPS
# ============================================================================

class AuthEvent(str, Enum):
    """Authentication events for audit trail."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    EMAIL_VERIFIED = "email_verified"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class AuthAttemptLog(TimestampMixin, ValidationMixin):
    """Real-time authentication attempt tracking with timestamps."""
    
    # Authentication attempt details
    attempt_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None  # None for failed attempts with unknown user
    username: Optional[str] = None
    nhs_number: Optional[str] = None  # For NHS ID login attempts
    nhs_user_type: Optional[NHSUserType] = None
    
    # Event details
    event_type: AuthEvent
    success: bool = Field(default=False)
    failure_reason: Optional[str] = None
    
    # Request context with timestamps
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    geolocation: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Security analysis
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calculated risk score")
    is_suspicious: bool = Field(default=False)
    security_flags: List[str] = Field(default_factory=list)
    
    # Session tracking
    session_id: Optional[str] = None
    previous_login: Optional[datetime] = None
    login_streak: int = Field(default=0, description="Consecutive successful logins")
    failed_attempts_today: int = Field(default=0, description="Failed attempts in last 24h")
    
    # Verification status at time of attempt
    nhs_verified_at_attempt: bool = Field(default=False)
    gmc_verified_at_attempt: bool = Field(default=False)
    
    # Additional metadata
    additional_context: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def create_login_attempt(cls, username: str, user_id: Optional[UUID] = None, **kwargs):
        """Create a login attempt log entry."""
        return cls(
            user_id=user_id,
            username=username,
            event_type=AuthEvent.LOGIN_ATTEMPT,
            **kwargs
        )
    
    @classmethod
    def create_success_log(cls, user_id: UUID, username: str, nhs_user_type: NHSUserType, **kwargs):
        """Create a successful authentication log entry."""
        return cls(
            user_id=user_id,
            username=username,
            nhs_user_type=nhs_user_type,
            event_type=AuthEvent.LOGIN_SUCCESS,
            success=True,
            **kwargs
        )
    
    @classmethod
    def create_failure_log(cls, username: str, failure_reason: str, **kwargs):
        """Create a failed authentication log entry."""
        return cls(
            username=username,
            event_type=AuthEvent.LOGIN_FAILED,
            success=False,
            failure_reason=failure_reason,
            **kwargs
        )

class AuthSessionLog(TimestampMixin, ValidationMixin):
    """Session lifecycle tracking with detailed timestamps."""
    
    session_log_id: UUID = Field(default_factory=uuid4)
    session_id: str = Field(..., description="Session token or identifier")
    user_id: UUID
    username: str
    nhs_user_type: NHSUserType
    
    # Session lifecycle timestamps
    session_started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    session_expires_at: datetime
    session_ended_at: Optional[datetime] = None
    
    # Session context
    login_method: str = Field(..., description="nhs_id, username_password, sso")
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Activity tracking
    total_requests: int = Field(default=0)
    pages_visited: List[str] = Field(default_factory=list)
    actions_performed: List[str] = Field(default_factory=list)
    data_accessed: List[str] = Field(default_factory=list)
    
    # Security monitoring
    concurrent_sessions: int = Field(default=1)
    location_changes: int = Field(default=0)
    suspicious_activities: List[str] = Field(default_factory=list)
    
    # Session termination
    termination_reason: Optional[str] = None  # logout, timeout, forced, suspicious
    terminated_by: Optional[str] = None  # user, system, admin
    
    def update_activity(self, action: str, page: Optional[str] = None):
        """Update session activity with timestamp."""
        self.last_activity_at = datetime.utcnow()
        self.total_requests += 1
        self.actions_performed.append(f"{datetime.utcnow().isoformat()}: {action}")
        if page:
            self.pages_visited.append(f"{datetime.utcnow().isoformat()}: {page}")
    
    def add_suspicious_activity(self, activity: str):
        """Log suspicious activity with timestamp."""
        timestamped_activity = f"{datetime.utcnow().isoformat()}: {activity}"
        self.suspicious_activities.append(timestamped_activity)
    
    def end_session(self, reason: str = "logout", terminated_by: str = "user"):
        """End session with timestamp."""
        self.session_ended_at = datetime.utcnow()
        self.termination_reason = reason
        self.terminated_by = terminated_by

# ============================================================================
# NHS CREDENTIALS AND VERIFICATION WITH TIMESTAMPS
# ============================================================================

class NHSCredentials(BaseModel):
    """NHS-specific credentials for patient authentication."""
    nhs_number: str = Field(..., regex=r'^\d{3}\s?\d{3}\s?\d{4}$')
    date_of_birth: datetime = Field(..., description="Date of birth for NHS verification")
    postcode: str = Field(..., min_length=5, max_length=8, description="UK postcode")
    
    @field_validator('nhs_number')
    def validate_nhs_number(cls, v):
        """Validate NHS number format and checksum."""
        nhs_clean = v.replace(' ', '')
        
        if len(nhs_clean) != 10 or not nhs_clean.isdigit():
            raise ValueError("NHS number must be 10 digits")
        
        # NHS number checksum validation
        total = sum(int(digit) * (10 - i) for i, digit in enumerate(nhs_clean[:9]))
        remainder = total % 11
        check_digit = 11 - remainder if remainder != 0 else 0
        
        if check_digit == 10 or int(nhs_clean[9]) != check_digit:
            raise ValueError("Invalid NHS number checksum")
        
        return nhs_clean

class GMCCredentials(BaseModel):
    """General Medical Council credentials for NHS doctors."""
    gmc_number: str = Field(..., regex=r'^\d{7}$', description="7-digit GMC number")
    gmc_status: str = Field(..., description="GMC registration status")
    specialty: DoctorSpecialty
    seniority: DoctorSeniority
    hospital_trust: str = Field(..., max_length=200)
    department: Optional[str] = Field(None, max_length=100)

class NHSVerificationLog(TimestampMixin, ValidationMixin):
    """NHS verification attempt tracking with timestamps."""
    
    verification_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    nhs_number: str
    
    # Verification attempt details
    verification_method: str = Field(default="pds_api")  # Patient Demographic Service
    verification_status: str = Field(..., description="pending, success, failed, error")
    
    # Timestamps for verification process
    verification_requested_at: datetime = Field(default_factory=datetime.utcnow)
    verification_completed_at: Optional[datetime] = None
    verification_expires_at: Optional[datetime] = None
    
    # Verification data
    verification_data: Dict[str, Any] = Field(default_factory=dict)
    error_details: Optional[str] = None
    retry_count: int = Field(default=0)
    
    # Security context
    requested_from_ip: Optional[str] = None
    verification_token: Optional[str] = None
    
    def complete_verification(self, success: bool, error_details: Optional[str] = None):
        """Complete verification process with timestamp."""
        self.verification_completed_at = datetime.utcnow()
        self.verification_status = "success" if success else "failed"
        if error_details:
            self.error_details = error_details

class GMCVerificationLog(TimestampMixin, ValidationMixin):
    """GMC verification attempt tracking with timestamps."""
    
    verification_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    gmc_number: str
    
    # Verification details
    verification_documents: List[str] = Field(default_factory=list)
    verification_status: str = Field(..., description="pending, success, failed, manual_review")
    
    # Timestamps
    verification_requested_at: datetime = Field(default_factory=datetime.utcnow)
    documents_uploaded_at: Optional[datetime] = None
    manual_review_started_at: Optional[datetime] = None
    verification_completed_at: Optional[datetime] = None
    
    # Review process
    reviewed_by: Optional[str] = None
    review_comments: Optional[str] = None
    
    def start_manual_review(self, reviewer: str) -> None:
        """Start manual review process with timestamp."""
        self.manual_review_started_at = datetime.utcnow()
        self.reviewed_by = reviewer
        self.verification_status = "manual_review"

# ============================================================================
# USER MODELS WITH ENHANCED TIMESTAMP TRACKING
# ============================================================================

class UserBase(ValidationMixin):
    """Base user fields."""
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: Optional[EmailStr] = None  # Optional for public patients
    full_name: str = Field(..., max_length=100)
    nhs_user_type: NHSUserType
    is_active: bool = Field(default=True)
    status: UserStatus = Field(default=UserStatus.PENDING_NHS_VERIFICATION)

class User(BaseEntity, UserBase, MetadataMixin):
    """Complete user model with NHS-specific fields and timestamp tracking."""
    password_hash: Optional[str] = None  # None for NHS ID-only authentication
    
    # Authentication tracking with timestamps
    last_login: Optional[datetime] = None
    last_logout: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    # Security tracking
    failed_login_attempts: int = Field(default=0)
    last_failed_login: Optional[datetime] = None
    locked_until: Optional[datetime] = None
    lock_reason: Optional[str] = None
    locked_at: Optional[datetime] = None
    
    # NHS verification with timestamps
    nhs_number: Optional[str] = None
    nhs_verified: bool = Field(default=False)
    nhs_verified_at: Optional[datetime] = None
    nhs_verification_method: Optional[str] = None
    nhs_verification_expires_at: Optional[datetime] = None
    
    # Medical professional verification with timestamps
    gmc_number: Optional[str] = None
    gmc_verified: bool = Field(default=False)
    gmc_verified_at: Optional[datetime] = None
    gmc_verification_expires_at: Optional[datetime] = None
    specialty: Optional[DoctorSpecialty] = None
    seniority: Optional[DoctorSeniority] = None
    hospital_trust: Optional[str] = None
    
    # Developer fields
    organization: Optional[str] = None
    security_clearance: Optional[str] = None
    security_clearance_granted_at: Optional[datetime] = None
    
    # Permissions and access
    permissions: List[PermissionScope] = Field(default_factory=list)
    permissions_last_updated: Optional[datetime] = None
    
    # Consent tracking with timestamps
    consent_data_processing: bool = Field(default=False)
    consent_ai_analysis: bool = Field(default=False)
    consent_research: bool = Field(default=False)
    consent_updated_at: Optional[datetime] = None
    
    # Activity statistics
    total_logins: int = Field(default=0)
    total_sessions: int = Field(default=0)
    average_session_duration: Optional[int] = None  # seconds
    
    def update_login_stats(self):
        """Update login statistics with current timestamp."""
        now = datetime.utcnow()
        self.last_login = now
        self.last_activity = now
        self.total_logins += 1
        self.failed_login_attempts = 0  # Reset on successful login
    
    def record_failed_login(self):
        """Record failed login attempt with timestamp."""
        self.failed_login_attempts += 1
        self.last_failed_login = datetime.utcnow()
    
    def lock_account(self, reason: str, duration_minutes: int = 30):
        """Lock account with timestamp and reason."""
        now = datetime.utcnow()
        self.locked_at = now
        self.locked_until = now + timedelta(minutes=duration_minutes)
        self.lock_reason = reason
    
    def unlock_account(self):
        """Unlock account and clear lock timestamps."""
        self.locked_until = None
        self.locked_at = None
        self.lock_reason = None
        self.failed_login_attempts = 0
    
    def update_permissions(self, new_permissions: List[PermissionScope]):
        """Update user permissions with timestamp."""
        self.permissions = new_permissions
        self.permissions_last_updated = datetime.utcnow()
    
    def update_consent(self, data_processing: bool, ai_analysis: bool, research: bool):
        """Update consent preferences with timestamp."""
        self.consent_data_processing = data_processing
        self.consent_ai_analysis = ai_analysis
        self.consent_research = research
        self.consent_updated_at = datetime.utcnow()
    
    def can_login(self) -> bool:
        """Check if user can login based on verification status."""
        if not self.is_active or self.is_locked():
            return False
        
        # Check if verifications are still valid
        now = datetime.utcnow()
        
        # Public patients only need NHS verification
        if self.nhs_user_type == NHSUserType.PUBLIC_PATIENT:
            if not self.nhs_verified:
                return False
            if self.nhs_verification_expires_at and now > self.nhs_verification_expires_at:
                return False
            return True
        
        # Doctors need both NHS and GMC verification
        if self.nhs_user_type in [NHSUserType.NHS_DOCTOR, NHSUserType.NHS_NURSE]:
            if not (self.nhs_verified and self.gmc_verified):
                return False
            if self.nhs_verification_expires_at and now > self.nhs_verification_expires_at:
                return False
            if self.gmc_verification_expires_at and now > self.gmc_verification_expires_at:
                return False
            return True
        
        # Developers need standard verification
        if self.nhs_user_type in [NHSUserType.DEVELOPER, NHSUserType.DATA_SCIENTIST, NHSUserType.SYSTEM_ADMIN]:
            return self.status == UserStatus.ACTIVE
        
        return False
    
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False

# ============================================================================
# AUTHENTICATION REQUEST/RESPONSE MODELS
# ============================================================================

class NHSLoginRequest(BaseModel):
    """NHS ID-based login for public patients."""
    nhs_credentials: NHSCredentials
    chat_session_id: Optional[str] = None

class StandardLoginRequest(BaseModel):
    """Username/password login for doctors and developers."""
    username: str = Field(..., min_length=3, max_length=50)
    password: SecretStr
    remember_me: bool = Field(default=False)

class TokenPayload(BaseModel):
    """JWT token payload with timestamp information."""
    sub: str  # User ID
    username: str
    nhs_user_type: NHSUserType
    nhs_number: Optional[str] = None
    gmc_number: Optional[str] = None
    specialty: Optional[DoctorSpecialty] = None
    permissions: List[PermissionScope]
    consent_flags: Dict[str, bool] = Field(default_factory=dict)
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    last_login: Optional[datetime] = None
    session_id: str
    jti: str

class TokenResponse(BaseResponse):
    """Token response for login/refresh endpoints with timestamp tracking."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    user_profile: Dict[str, Any]
    session_id: str

# ============================================================================
# SESSION MANAGEMENT WITH ENHANCED TIMESTAMPS
# ============================================================================

class SessionStatus(str, Enum):
    """Session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPICIOUS = "suspicious"

class UserSession(BaseEntity):
    """User session model with comprehensive timestamp tracking."""
    user_id: UUID
    session_token: str = Field(..., unique=True)
    nhs_user_type: NHSUserType
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)
    
    # Session lifecycle timestamps
    session_created_at: datetime = Field(default_factory=datetime.utcnow)
    session_expires_at: datetime
    session_last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    session_terminated_at: Optional[datetime] = None
    
    # Session context
    nhs_context: Dict[str, Any] = Field(default_factory=dict)
    medical_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Session tracking
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    location: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Activity metrics
    total_requests: int = Field(default=0)
    last_request_at: Optional[datetime] = None
    pages_accessed: int = Field(default=0)
    data_queries_made: int = Field(default=0)
    
    # Security monitoring
    ip_changes: int = Field(default=0)
    location_changes: int = Field(default=0)
    suspicious_activity_count: int = Field(default=0)
    last_suspicious_activity_at: Optional[datetime] = None
    
    def update_activity(self):
        """Update session activity with current timestamp."""
        now = datetime.utcnow()
        self.session_last_accessed_at = now
        self.last_request_at = now
        self.total_requests += 1
    
    def record_suspicious_activity(self):
        """Record suspicious activity with timestamp."""
        self.suspicious_activity_count += 1
        self.last_suspicious_activity_at = datetime.utcnow()
        if self.suspicious_activity_count >= 5:
            self.status = SessionStatus.SUSPICIOUS
    
    def terminate_session(self, reason: str = "logout"):
        """Terminate session with timestamp and reason."""
        self.session_terminated_at = datetime.utcnow()
        self.status = SessionStatus.TERMINATED
        if not hasattr(self, 'termination_reason'):
            self.termination_reason = reason
    
    def is_active(self) -> bool:
        """Check if session is active and not expired."""
        return (
            self.status == SessionStatus.ACTIVE and
            datetime.utcnow() < self.session_expires_at
        )

# ============================================================================
# PERMISSION MAPPINGS
# ============================================================================


NHS_ROLE_PERMISSIONS = {
    NHSUserType.PUBLIC_PATIENT: [
        PermissionScope.PATIENT_OWN_READ,
        PermissionScope.PATIENT_OWN_WRITE,
        PermissionScope.AI_MODEL_USE
    ],
    NHSUserType.NHS_DOCTOR: [
        PermissionScope.PATIENT_ALL_READ,
        PermissionScope.PATIENT_ALL_WRITE,
        PermissionScope.MEDICAL_RECORDS_READ,
        PermissionScope.MEDICAL_RECORDS_WRITE,
        PermissionScope.MEDICAL_PRESCRIBE,
        PermissionScope.MEDICAL_DIAGNOSE,
        PermissionScope.AI_MODEL_USE,
        PermissionScope.AI_BIAS_MONITOR
    ],
    NHSUserType.NHS_NURSE: [
        PermissionScope.PATIENT_ALL_READ,
        PermissionScope.MEDICAL_RECORDS_READ,
        PermissionScope.AI_MODEL_USE
    ],
    NHSUserType.DATA_SCIENTIST: [
        PermissionScope.DATA_EXPORT_ANON,
        PermissionScope.DATA_RESEARCH,
        PermissionScope.AI_MODEL_TRAIN,
        PermissionScope.AI_MODEL_DEPLOY,
        PermissionScope.AI_BIAS_MONITOR,
        PermissionScope.AI_BIAS_ADJUST,
        PermissionScope.SYSTEM_MONITOR
    ],
    NHSUserType.DEVELOPER: [
        PermissionScope.SYSTEM_ADMIN,
        PermissionScope.SYSTEM_CONFIG,
        PermissionScope.SYSTEM_MONITOR,
        PermissionScope.AI_MODEL_TRAIN,
        PermissionScope.AI_MODEL_DEPLOY,
        PermissionScope.DATA_EXPORT_FULL
    ],
    NHSUserType.SYSTEM_ADMIN: [
        # All permissions
        PermissionScope.SYSTEM_ADMIN,
        PermissionScope.SYSTEM_CONFIG,
        PermissionScope.SYSTEM_MONITOR,
        PermissionScope.PATIENT_ALL_READ,
        PermissionScope.PATIENT_ALL_WRITE,
        PermissionScope.MEDICAL_RECORDS_READ,
        PermissionScope.MEDICAL_RECORDS_WRITE,
        PermissionScope.AI_MODEL_TRAIN,
        PermissionScope.AI_MODEL_DEPLOY,
        PermissionScope.AI_BIAS_MONITOR,
        PermissionScope.AI_BIAS_ADJUST,
        PermissionScope.DATA_EXPORT_FULL
    ]
}
