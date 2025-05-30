"""
V1 Authentication Data Models
Pure data structures for users, roles, and authentication - NO SECURITY LOGIC
Security operations are handled in ../core/security.py
"""

from pydantic import BaseModel, Field, EmailStr, validator
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.postgresql import JSONB
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

from datamodels.sqlalchemy_models import Base

# =============================================================================
# USER ROLES AND PERMISSIONS (V1 SPECIFIC)
# =============================================================================

class UserRole(str, Enum):
    """V1 User roles with medical-grade RBAC"""
    DEVELOPER = "developer"        # God mode - full system access
    ADMIN = "admin"               # System administration - no code access
    DOCTOR = "doctor"             # Verified medical professional
    DATA_SCIENTIST = "data_scientist"  # AI/ML specialist (v2 focus)
    PATIENT = "patient"           # End user seeking medical assistance
    GUEST = "guest"              # Unauthenticated user

class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_VERIFICATION = "pending_verification"
    SUSPENDED = "suspended"
    BANNED = "banned"
    LOCKED = "locked"

class DoctorVerificationStatus(str, Enum):
    """Doctor verification status for medical professionals"""
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNDER_REVIEW = "under_review"

class DoctorSpecialty(str, Enum):
    """Medical specialties for NHS doctors"""
    EMERGENCY_MEDICINE = "emergency_medicine"
    CARDIOLOGY = "cardiology"
    INTERNAL_MEDICINE = "internal_medicine"
    FAMILY_MEDICINE = "family_medicine"
    CRITICAL_CARE = "critical_care"
    ANESTHESIOLOGY = "anesthesiology"
    SURGERY = "surgery"
    NEUROLOGY = "neurology"
    PSYCHIATRY = "psychiatry"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    OTHER = "other"

class AuthEventType(str, Enum):
    """Authentication event types for audit trail"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"
    PERMISSION_DENIED = "permission_denied"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MASTER_PASSWORD_REQUIRED = "master_password_required"
    EMERGENCY_ACCESS = "emergency_access"

# =============================================================================
# SQLALCHEMY DATABASE MODELS (DATA STRUCTURES ONLY)
# =============================================================================

class UserDB(Base):
    """User database model - data structure only, no validation logic"""
    __tablename__ = "users"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone_number = Column(String(20), nullable=True)
    
    # Role and status
    role = Column(String(20), default=UserRole.PATIENT.value, nullable=False)
    status = Column(String(20), default=UserStatus.ACTIVE.value, nullable=False)
    
    # Authentication tracking
    is_email_verified = Column(Boolean, default=False)
    is_phone_verified = Column(Boolean, default=False)
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # API keys for frontend integration
    api_key = Column(String(255), nullable=True)
    api_key_expires = Column(DateTime, nullable=True)
    
    # Master password verification tracking
    master_password_verified = Column(Boolean, default=False)
    master_password_verified_at = Column(DateTime, nullable=True)
    
    # Relationships
    doctor_profile = relationship("DoctorProfileDB", back_populates="user", uselist=False)
    sessions = relationship("UserSessionDB", back_populates="user")
    auth_events = relationship("AuthEventDB", back_populates="user")

class DoctorProfileDB(Base):
    """Doctor profile data model"""
    __tablename__ = "doctor_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Medical credentials
    medical_license_number = Column(String(100), nullable=False)
    specialization = Column(String(50), default=DoctorSpecialty.OTHER.value)
    years_of_experience = Column(Integer, nullable=False)
    hospital_affiliation = Column(String(200), nullable=True)
    department = Column(String(100), nullable=True)
    
    # Verification status
    verification_status = Column(String(20), default=DoctorVerificationStatus.UNVERIFIED.value)
    verification_documents = Column(MutableDict.as_mutable(JSONB), nullable=True)
    verified_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    verification_expires = Column(DateTime, nullable=True)
    verification_notes = Column(Text, nullable=True)
    
    # Emergency access
    can_call_emergency = Column(Boolean, default=False)
    emergency_contact_verified = Column(Boolean, default=False)
    
    # Payment information (placeholder)
    payment_method = Column(String(50), nullable=True)
    payment_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("UserDB", back_populates="doctor_profile")

class UserSessionDB(Base):
    """User session data model"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Token information
    access_token = Column(String(500), nullable=False)
    refresh_token = Column(String(500), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    # Device and location information
    device_info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Session status
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # WebSocket information
    websocket_connected = Column(Boolean, default=False)
    websocket_last_ping = Column(DateTime, nullable=True)
    
    # Security context
    master_password_verified = Column(Boolean, default=False)
    security_level = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("UserDB", back_populates="sessions")

class AuthEventDB(Base):
    """Authentication event logging model"""
    __tablename__ = "auth_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False)
    success = Column(Boolean, default=False)
    failure_reason = Column(String(255), nullable=True)
    
    # Request context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    device_fingerprint = Column(String(255), nullable=True)
    
    # Security analysis
    risk_score = Column(Integer, nullable=True)  # 0-100 scale
    is_suspicious = Column(Boolean, default=False)
    security_flags = Column(MutableDict.as_mutable(JSONB), nullable=True)
    
    # Session context
    session_id = Column(String(50), nullable=True)
    master_password_used = Column(Boolean, default=False)
    
    # Additional metadata
    additional_context = Column(MutableDict.as_mutable(JSONB), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("UserDB", back_populates="auth_events")

class ApiKeyDB(Base):
    """API key management model"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # API key details
    api_key = Column(String(255), unique=True, nullable=False)
    key_name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    
    # Key status and permissions
    is_active = Column(Boolean, default=True)
    permissions = Column(MutableDict.as_mutable(JSONB), nullable=True)
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    # Usage tracking
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    
    # Expiration
    expires_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# =============================================================================
# PYDANTIC API MODELS (REQUEST/RESPONSE SCHEMAS)
# =============================================================================

class UserBase(BaseModel):
    """Base user fields for API requests"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone_number: Optional[str] = Field(None, regex=r'^\+?1?\d{9,15}$')

class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.PATIENT
    master_password: Optional[str] = None  # Required for medical roles
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    """User update model"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[EmailStr] = None

class UserResponse(UserBase):
    """User response model (excludes sensitive data)"""
    user_id: str
    role: UserRole
    status: UserStatus
    is_email_verified: bool
    is_phone_verified: bool
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class DoctorProfileCreate(BaseModel):
    """Doctor profile creation model"""
    medical_license_number: str = Field(..., min_length=5, max_length=100)
    specialization: DoctorSpecialty = DoctorSpecialty.OTHER
    years_of_experience: int = Field(..., ge=0, le=60)
    hospital_affiliation: Optional[str] = Field(None, max_length=200)
    department: Optional[str] = Field(None, max_length=100)
    verification_documents: Optional[Dict[str, Any]] = None

class DoctorProfileResponse(BaseModel):
    """Doctor profile response model"""
    medical_license_number: str
    specialization: DoctorSpecialty
    years_of_experience: int
    hospital_affiliation: Optional[str]
    department: Optional[str]
    verification_status: DoctorVerificationStatus
    verified_at: Optional[datetime]
    can_call_emergency: bool
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")
    master_password: Optional[str] = Field(None, description="Master password for medical roles")
    remember_me: bool = Field(default=False)

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
    session_id: str
    security_level: str
    medical_access: bool = False

class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[str] = []
    security_level: Optional[str] = None
    medical_access: bool = False
    master_verified: bool = False
    api_version: str = "v1"

class APIKeyRequest(BaseModel):
    """API key creation request"""
    key_name: str = Field(..., max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    expires_in_days: int = Field(default=30, ge=1, le=365)
    permissions: Optional[List[str]] = None

class APIKeyResponse(BaseModel):
    """API key response model"""
    key_id: str
    api_key: str
    key_name: str
    description: Optional[str]
    expires_at: Optional[datetime]
    permissions: Optional[List[str]]
    created_at: datetime

class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8)
    master_password: Optional[str] = None  # Required for medical roles

class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr

class WebSocketConnectionInfo(BaseModel):
    """WebSocket connection information"""
    session_id: str
    user_id: str
    connected_at: datetime
    last_ping: Optional[datetime]

class SecurityStatusResponse(BaseModel):
    """Security status response"""
    user_id: str
    role: str
    status: str
    active_sessions: int
    failed_login_attempts: int
    is_email_verified: bool
    is_phone_verified: bool
    last_login: Optional[datetime]
    api_key_active: bool
    api_key_expires: Optional[datetime]
    master_password_verified: bool
    security_level: str

class AuthEventLog(BaseModel):
    """Authentication event log entry"""
    event_id: str
    user_id: Optional[str]
    event_type: AuthEventType
    success: bool
    failure_reason: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    risk_score: Optional[int]
    is_suspicious: bool
    master_password_used: bool
    created_at: datetime

# =============================================================================
# V1 SPECIFIC ROLE PERMISSIONS MAPPING (DATA ONLY)
# =============================================================================


V1_ROLE_PERMISSIONS = {
    UserRole.DEVELOPER: [
        "*"  # God mode - all permissions
    ],
    UserRole.ADMIN: [
        "user:read", "user:write", "user:delete",
        "doctor:verify", "doctor:read", "doctor:write",
        "case:read", "case:write", "case:delete",
        "protocol:read", "protocol:write",
        "service:manage", "billing:manage",
        "frontend:marry", "backend:marry",
        "reports:read", "analytics:read",
        "emergency:view", "system:monitor"
    ],
    UserRole.DOCTOR: [
        "patient:read", "patient:write",
        "case:read", "case:write", "case:emergency",
        "protocol:read", "reports:receive",
        "emergency:call", "profile:read", "profile:write",
        "verification:request", "medical:diagnose",
        "prescription:write", "referral:create"
    ],
    UserRole.DATA_SCIENTIST: [
        "model:read", "model:write", "model:deploy",
        "data:read", "data:export", "analytics:read",
        "analytics:write", "service:read", "service:write",
        "ml:train", "ml:evaluate", "bias:monitor"
    ],
    UserRole.PATIENT: [
        "chat:ai", "chat:doctor", "chat:admin",
        "case:create", "case:read_own",
        "file:upload", "report:download_own",
        "complaint:create", "profile:read", "profile:write"
    ],
    UserRole.GUEST: [
        "health:check", "protocols:read",
        "auth:login", "auth:register"
    ]
}

# Security levels for each role
V1_ROLE_SECURITY_LEVELS = {
    UserRole.DEVELOPER: "developer",
    UserRole.ADMIN: "admin",
    UserRole.DOCTOR: "medical",
    UserRole.DATA_SCIENTIST: "medical",
    UserRole.PATIENT: "basic",
    UserRole.GUEST: "public"
}

# Master password requirements
V1_MASTER_PASSWORD_REQUIRED_ROLES = [
    UserRole.DEVELOPER,
    UserRole.ADMIN,
    UserRole.DOCTOR
]

# =============================================================================
# UTILITY FUNCTIONS (DATA ONLY)
# =============================================================================

def get_role_permissions(role: UserRole) -> List[str]:
    """Get permissions for a role - data retrieval only"""
    return V1_ROLE_PERMISSIONS.get(role, [])

def get_role_security_level(role: UserRole) -> str:
    """Get security level for a role - data retrieval only"""
    return V1_ROLE_SECURITY_LEVELS.get(role, "public")

def requires_master_password(role: UserRole) -> bool:
    """Check if role requires master password - data check only"""
    return role in V1_MASTER_PASSWORD_REQUIRED_ROLES

def is_medical_role(role: UserRole) -> bool:
    """Check if role is medical - data check only"""
    return role in [UserRole.DOCTOR, UserRole.ADMIN, UserRole.DEVELOPER]

def is_emergency_role(role: UserRole) -> bool:
    """Check if role can access emergency functions - data check only"""
    return role in [UserRole.DOCTOR, UserRole.ADMIN, UserRole.DEVELOPER]

# =============================================================================
# EXPORT
# =============================================================================


__all__ = [
    # Enums
    "UserRole", "UserStatus", "DoctorVerificationStatus",
    "DoctorSpecialty", "AuthEventType",
    
    # Database Models
    "UserDB", "DoctorProfileDB", "UserSessionDB",
    "AuthEventDB", "ApiKeyDB",
    
    # API Models
    "UserCreate", "UserUpdate", "UserResponse",
    "DoctorProfileCreate", "DoctorProfileResponse",
    "LoginRequest", "LoginResponse", "TokenData",
    "APIKeyRequest", "APIKeyResponse",
    "PasswordChangeRequest", "PasswordResetRequest",
    "WebSocketConnectionInfo", "SecurityStatusResponse",
    "AuthEventLog",
    
    # Data mappings
    "V1_ROLE_PERMISSIONS", "V1_ROLE_SECURITY_LEVELS",
    "V1_MASTER_PASSWORD_REQUIRED_ROLES",
    
    # Utility functions
    "get_role_permissions", "get_role_security_level",
    "requires_master_password", "is_medical_role", "is_emergency_role"
]
