"""
Base data models for Fairdoc Medical AI Backend.
Provides common fields, timestamps, and base functionality for all data models.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, validator
from enum import Enum

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

class UUIDMixin(BaseModel):
    """Mixin for UUID primary key."""
    id: UUID = Field(default_factory=uuid4)

class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete functionality."""
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = None
    
    def soft_delete(self):
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

class EnvironmentMixin(BaseModel):
    """Mixin for environment tracking."""
    environment: Optional[str] = Field(default=None, description="Environment where record was created")

class BaseEntity(UUIDMixin, TimestampMixin, SoftDeleteMixin, EnvironmentMixin):
    """Base entity with all common fields."""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class BaseResponse(BaseModel):
    """Base response model for API responses."""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PaginationMixin(BaseModel):
    """Mixin for pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size

class PaginatedResponse(BaseResponse):
    """Paginated response wrapper."""
    page: int
    size: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool
    data: List[Any]
    
    @classmethod
    def create(cls, data: List[Any], page: int, size: int, total: int):
        """Create paginated response."""
        pages = (total + size - 1) // size  # Ceiling division
        return cls(
            data=data,
            page=page,
            size=size,
            total=total,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )

class RiskLevel(str, Enum):
    """Medical risk levels for triage."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class UrgencyLevel(str, Enum):
    """Urgency levels for medical conditions."""
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENT = "emergent"
    IMMEDIATE = "immediate"

class Gender(str, Enum):
    """Gender options."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class Ethnicity(str, Enum):
    """Ethnicity categories (NHS compatible)."""
    WHITE_BRITISH = "white_british"
    WHITE_IRISH = "white_irish"
    WHITE_OTHER = "white_other"
    MIXED_WHITE_BLACK_CARIBBEAN = "mixed_white_black_caribbean"
    MIXED_WHITE_BLACK_AFRICAN = "mixed_white_black_african"
    MIXED_WHITE_ASIAN = "mixed_white_asian"
    MIXED_OTHER = "mixed_other"
    ASIAN_INDIAN = "asian_indian"
    ASIAN_PAKISTANI = "asian_pakistani"
    ASIAN_BANGLADESHI = "asian_bangladeshi"
    ASIAN_OTHER = "asian_other"
    BLACK_CARIBBEAN = "black_caribbean"
    BLACK_AFRICAN = "black_african"
    BLACK_OTHER = "black_other"
    CHINESE = "chinese"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class UserRole(str, Enum):
    """User roles in the system."""
    PATIENT = "patient"
    DOCTOR = "doctor"
    NURSE = "nurse"
    ADMIN = "admin"
    RESEARCHER = "researcher"

class ValidationMixin(BaseModel):
    """Mixin for common validation methods."""
    
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        """Convert empty strings to None."""
        if v == '':
            return None
        return v

class MetadataMixin(BaseModel):
    """Mixin for storing additional metadata."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None):
        """Get metadata value by key."""
        return self.metadata.get(key, default)

class VersionMixin(BaseModel):
    """Mixin for versioning records."""
    version: int = Field(default=1, description="Record version for optimistic locking")
    
    def increment_version(self):
        """Increment version number."""
        self.version += 1

class AuditMixin(BaseModel):
    """Mixin for audit trail."""
    created_by: Optional[UUID] = None
    updated_by: Optional[UUID] = None
    
    def set_created_by(self, user_id: UUID):
        """Set creator user ID."""
        self.created_by = user_id
    
    def set_updated_by(self, user_id: UUID):
        """Set updater user ID."""
        self.updated_by = user_id

class BaseMedicalEntity(BaseEntity, ValidationMixin, MetadataMixin, VersionMixin, AuditMixin):
    """Base entity for medical records with full audit trail."""
    
    # NHS Number validation for UK patients
    nhs_number: Optional[str] = Field(
        None, 
        regex=r'^\d{3}\s?\d{3}\s?\d{4}$',
        description="NHS Number (10 digits)"
    )
    
    @validator('nhs_number')
    def validate_nhs_number(cls, v):
        """Validate NHS number format and checksum."""
        if v is None:
            return v
        
        # Remove spaces
        nhs_clean = v.replace(' ', '')
        
        if len(nhs_clean) != 10 or not nhs_clean.isdigit():
            raise ValueError("NHS number must be 10 digits")
        
        # NHS number checksum validation
        total = sum(int(digit) * (10 - i) for i, digit in enumerate(nhs_clean[:9]))
        remainder = total % 11
        check_digit = 11 - remainder if remainder != 0 else 0
        
        if check_digit == 10 or int(nhs_clean[9]) != check_digit:
            raise ValueError("Invalid NHS number checksum")
        
        return v

class ErrorResponse(BaseModel):
    """Standard error response format."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
