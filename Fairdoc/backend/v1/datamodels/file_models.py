"""
V1 File Management Data Models - Production Grade
Pydantic models with comprehensive field validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import re

# =============================================================================
# FILE ENUMS
# =============================================================================

class FileCategory(str, Enum):
    """Medical file categories"""
    MEDICAL_REPORT = "medical_report"
    MEDICAL_IMAGE = "medical_image"
    DICOM_IMAGE = "dicom_image"
    AUDIO_RECORDING = "audio_recording"
    LABORATORY_RESULT = "laboratory_result"
    PRESCRIPTION = "prescription"
    REFERRAL_LETTER = "referral_letter"
    DISCHARGE_SUMMARY = "discharge_summary"
    OTHER = "other"

class FileStatus(str, Enum):
    """File processing status"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"
    QUARANTINED = "quarantined"
    VERIFIED = "verified"

class FileAccessLevel(str, Enum):
    """File access permissions"""
    PRIVATE = "private"           # Only uploader
    CASE_PARTICIPANTS = "case_participants"  # Patient + assigned doctors
    MEDICAL_STAFF = "medical_staff"         # All verified doctors
    PUBLIC = "public"             # Public research (anonymized)

class SecurityThreatLevel(str, Enum):
    """Security threat levels"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"

# =============================================================================
# FILE REQUEST/RESPONSE MODELS WITH VALIDATORS
# =============================================================================

class FileUploadRequest(BaseModel):
    """File upload request metadata with validation"""
    case_id: Optional[str] = Field(None, description="Associated case ID")
    description: Optional[str] = Field(None, max_length=500, description="File description")
    file_category: Optional[FileCategory] = Field(FileCategory.OTHER, description="File category")
    access_level: FileAccessLevel = Field(FileAccessLevel.CASE_PARTICIPANTS, description="Access permissions")
    tags: List[str] = Field(default=[], description="File tags for organization")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    @field_validator('case_id')
    @classmethod
    def validate_case_id(cls, v):
        if v is not None:
            # UUID format validation
            if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', v, re.IGNORECASE):
                raise ValueError('case_id must be a valid UUID format')
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', ';']
            if any(char in v for char in dangerous_chars):
                raise ValueError('Description contains potentially dangerous characters')
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        for tag in v:
            if len(tag) > 50:
                raise ValueError('Tag length cannot exceed 50 characters')
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValueError('Tags can only contain alphanumeric characters, underscores, and hyphens')
        return v

class FileMetadata(BaseModel):
    """Complete file metadata with validation"""
    file_id: str = Field(..., description="Unique file identifier")
    original_filename: str = Field(..., description="Original filename from upload")
    secure_filename: str = Field(..., description="Secure filename for storage")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_category: FileCategory = Field(..., description="File category")
    description: Optional[str] = Field(None, description="File description")
    
    # Storage information
    minio_object_path: str = Field(..., description="MinIO storage path")
    file_url: str = Field(..., description="Access URL")
    minio_etag: str = Field(..., description="MinIO ETag for integrity")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    
    # Ownership and access
    uploaded_by: str = Field(..., description="User who uploaded the file")
    case_id: Optional[str] = Field(None, description="Associated case ID")
    access_level: FileAccessLevel = Field(..., description="Access permissions")
    tags: List[str] = Field(default=[], description="File tags")
    
    # Status and timestamps
    status: FileStatus = Field(FileStatus.UPLOADED, description="Processing status")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion time")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration timestamp")
    
    # Security and analysis
    security_scan: Optional[Dict[str, Any]] = Field(None, description="Security scan results")
    medical_analysis: Optional[Dict[str, Any]] = Field(None, description="AI analysis results")
    compliance_flags: Dict[str, bool] = Field(default={}, description="Compliance check flags")
    
    @field_validator('file_id')
    @classmethod
    def validate_file_id(cls, v):
        if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', v, re.IGNORECASE):
            raise ValueError('file_id must be a valid UUID format')
        return v
    
    @field_validator('original_filename')
    @classmethod
    def validate_original_filename(cls, v):
        # Security validation for filename
        dangerous_patterns = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*', ';', '&']
        if any(pattern in v for pattern in dangerous_patterns):
            raise ValueError('Filename contains potentially dangerous characters')
        if len(v) > 255:
            raise ValueError('Filename too long (max 255 characters)')
        return v
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        max_size = 100 * 1024 * 1024  # 100MB default
        if v > max_size:
            raise ValueError(f'File size {v} exceeds maximum allowed size {max_size}')
        return v
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        allowed_types = [
            'application/pdf', 'image/jpeg', 'image/png', 'image/dicom',
            'audio/wav', 'audio/mp3', 'audio/mpeg', 'text/plain',
            'application/json', 'application/xml'
        ]
        if v not in allowed_types:
            raise ValueError(f'Content type {v} not allowed')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FileUploadResponse(BaseModel):
    """File upload response with tracking"""
    file_metadata: FileMetadata
    upload_status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Human-readable message")
    access_info: Dict[str, Any] = Field(..., description="Access information")
    processing_started: bool = Field(default=False, description="Whether background processing started")
    estimated_processing_time: Optional[str] = Field(None, description="Estimated processing time")

class BatchUploadResponse(BaseModel):
    """Batch file upload response with comprehensive tracking"""
    batch_upload_status: str = Field(..., description="Batch upload status")
    batch_id: str = Field(..., description="Batch processing ID")
    total_files: int = Field(..., description="Total files processed")
    successful_uploads: int = Field(..., description="Successfully uploaded files")
    failed_uploads: int = Field(..., description="Failed uploads")
    upload_results: List[Dict[str, Any]] = Field(..., description="Per-file results")
    uploaded_files: List[FileMetadata] = Field(..., description="Successfully uploaded file metadata")
    processing_queue_position: Optional[int] = Field(None, description="Position in processing queue")
    message: str = Field(..., description="Summary message")
    started_at: datetime = Field(default_factory=datetime.now, description="Batch start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class FileAccessResponse(BaseModel):
    """File access information response"""
    file_metadata: FileMetadata
    case_id: Optional[str] = Field(None, description="Associated case ID")
    access_url: str = Field(..., description="Secure access URL")
    access_expires: str = Field(..., description="Access expiration info")
    file_permissions: Dict[str, bool] = Field(..., description="User permissions for this file")
    access_logged: bool = Field(default=True, description="Whether access was logged")
    accessed_at: datetime = Field(default_factory=datetime.now, description="Access timestamp")

class FileListResponse(BaseModel):
    """File listing response with pagination"""
    files: List[FileMetadata] = Field(..., description="List of files")
    total_files: int = Field(..., description="Total files available")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Page size")
    total_pages: int = Field(..., description="Total pages available")
    filters: Dict[str, Any] = Field(..., description="Applied filters")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    generated_at: datetime = Field(default_factory=datetime.now, description="Response generation time")

# =============================================================================
# FILE VALIDATION MODELS WITH ENHANCED SECURITY
# =============================================================================

class FileValidationResult(BaseModel):
    """Comprehensive file validation result"""
    validation_passed: bool = Field(..., description="Whether validation passed")
    original_filename: str = Field(..., description="Original filename")
    secure_filename: str = Field(..., description="Generated secure filename")
    content_type: str = Field(..., description="Detected content type")
    file_size: int = Field(..., description="File size in bytes")
    file_category: FileCategory = Field(..., description="Determined file category")
    validation_errors: List[str] = Field(default=[], description="Validation error messages")
    validation_warnings: List[str] = Field(default=[], description="Validation warnings")
    security_checks: Dict[str, bool] = Field(default={}, description="Security check results")
    file_hash: str = Field(..., description="File content hash")
    validated_at: datetime = Field(default_factory=datetime.now, description="Validation timestamp")

class FileSecurityScan(BaseModel):
    """Comprehensive file security scan results"""
    is_safe: bool = Field(..., description="Whether file passed security scan")
    threat_level: SecurityThreatLevel = Field(..., description="Assessed threat level")
    scan_timestamp: datetime = Field(..., description="When scan was performed")
    scan_duration_ms: int = Field(..., description="Scan duration in milliseconds")
    threats_detected: List[str] = Field(default=[], description="Detected threats")
    file_signature_valid: bool = Field(..., description="File signature validation")
    content_analysis: Dict[str, Any] = Field(default={}, description="Content analysis results")
    virus_scan_result: str = Field(default="clean", description="Antivirus scan result")
    malware_detected: bool = Field(default=False, description="Whether malware was detected")
    quarantine_required: bool = Field(default=False, description="Whether file should be quarantined")
    scan_engine_version: str = Field(..., description="Security scan engine version")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in scan results")

# =============================================================================
# MEDICAL FILE SPECIFIC MODELS
# =============================================================================

class DicomFileInfo(BaseModel):
    """DICOM-specific file information with validation"""
    patient_id: Optional[str] = Field(None, description="DICOM Patient ID")
    study_date: Optional[datetime] = Field(None, description="Study date")
    modality: Optional[str] = Field(None, description="DICOM modality")
    body_part: Optional[str] = Field(None, description="Body part examined")
    institution: Optional[str] = Field(None, description="Institution name")
    manufacturer: Optional[str] = Field(None, description="Equipment manufacturer")
    series_description: Optional[str] = Field(None, description="Series description")
    pixel_spacing: Optional[List[float]] = Field(None, description="Pixel spacing")
    
    @field_validator('modality')
    @classmethod
    def validate_modality(cls, v):
        if v is not None:
            valid_modalities = ['CT', 'MR', 'US', 'XR', 'CR', 'DR', 'MG', 'PT', 'NM']
            if v not in valid_modalities:
                raise ValueError(f'Invalid DICOM modality: {v}')
        return v

class AudioRecordingInfo(BaseModel):
    """Audio recording specific information with validation"""
    duration_seconds: Optional[float] = Field(None, description="Recording duration")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    bit_depth: Optional[int] = Field(None, description="Audio bit depth")
    transcript: Optional[str] = Field(None, description="Speech-to-text transcript")
    language_detected: Optional[str] = Field(None, description="Detected language")
    noise_level: Optional[float] = Field(None, ge=0, le=1, description="Background noise level")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Audio quality score")
    
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v is not None:
            valid_rates = [8000, 16000, 22050, 44100, 48000, 96000]
            if v not in valid_rates:
                raise ValueError(f'Invalid sample rate: {v}')
        return v
    
    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if v is not None and (v < 1 or v > 8):
            raise ValueError('Audio channels must be between 1 and 8')
        return v

class ImageAnalysis(BaseModel):
    """Medical image analysis results with validation"""
    image_type: Optional[str] = Field(None, description="Type of medical image")
    body_part_detected: Optional[str] = Field(None, description="Detected body part")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Image quality score")
    resolution: Optional[List[int]] = Field(None, description="Image resolution [width, height]")
    ai_findings: List[str] = Field(default=[], description="AI-detected findings")
    confidence_scores: Dict[str, float] = Field(default={}, description="Confidence scores for findings")
    processing_time_ms: Optional[int] = Field(None, description="Analysis processing time")
    analysis_model_version: Optional[str] = Field(None, description="AI model version used")
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Quality score must be between 0 and 1')
        return v

# =============================================================================
# BATCH PROCESSING MODELS
# =============================================================================

class BatchProcessingRequest(BaseModel):
    """Batch processing request with validation"""
    batch_id: str = Field(..., description="Unique batch identifier")
    file_ids: List[str] = Field(..., description="List of file IDs to process")
    processing_type: str = Field(..., description="Type of processing")
    priority: int = Field(default=1, ge=1, le=5, description="Processing priority")
    parameters: Dict[str, Any] = Field(default={}, description="Processing parameters")
    notification_webhook: Optional[str] = Field(None, description="Webhook URL for completion notification")
    
    @field_validator('file_ids')
    @classmethod
    def validate_file_ids(cls, v):
        if len(v) == 0:
            raise ValueError('At least one file ID required')
        if len(v) > 100:
            raise ValueError('Maximum 100 files per batch')
        for file_id in v:
            if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', file_id, re.IGNORECASE):
                raise ValueError(f'Invalid file ID format: {file_id}')
        return v

class BatchProcessingResponse(BaseModel):
    """Batch processing response with comprehensive tracking"""
    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Processing status")
    total_files: int = Field(..., description="Total files in batch")
    processed_files: int = Field(..., description="Files processed so far")
    failed_files: int = Field(..., description="Files that failed processing")
    processing_results: List[Dict[str, Any]] = Field(..., description="Per-file processing results")
    started_at: datetime = Field(..., description="Processing start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    completed_at: Optional[datetime] = Field(None, description="Actual completion time")
    error_summary: Optional[str] = Field(None, description="Summary of errors if any")

# =============================================================================
# VALIDATORS UTILITY CLASS
# =============================================================================

class FileValidators:
    """Enhanced file validation utility functions"""
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security"""
        if not filename:
            return False
        
        # Check for path traversal attempts
        dangerous_patterns = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*', ';', '&', '$', '`']
        if any(pattern in filename for pattern in dangerous_patterns):
            return False
        
        # Check filename length
        if len(filename) > 255:
            return False
        
        # Check for valid characters only
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return False
        
        return True
    
    @staticmethod
    def validate_file_size(size: int, max_size: int = 100 * 1024 * 1024) -> bool:
        """Validate file size"""
        return 0 < size <= max_size
    
    @staticmethod
    def validate_content_type(content_type: str, allowed_types: List[str]) -> bool:
        """Validate content type"""
        return content_type in allowed_types
    
    @staticmethod
    def validate_medical_content_type(content_type: str) -> bool:
        """Validate medical-specific content types"""
        medical_types = [
            'application/pdf',           # Medical reports
            'image/jpeg', 'image/png',   # Medical images
            'image/dicom',               # DICOM images
            'audio/wav', 'audio/mp3',    # Audio recordings
            'text/plain',                # Text reports
            'application/json'           # Structured data
        ]
        return content_type in medical_types

# =============================================================================
# EXPORT
# =============================================================================


__all__ = [
    # Enums
    "FileCategory", "FileStatus", "FileAccessLevel", "SecurityThreatLevel",
    
    # Request/Response Models
    "FileUploadRequest", "FileMetadata", "FileUploadResponse",
    "BatchUploadResponse", "FileAccessResponse", "FileListResponse",
    
    # Validation Models
    "FileValidationResult", "FileSecurityScan",
    
    # Medical File Models
    "DicomFileInfo", "AudioRecordingInfo", "ImageAnalysis",
    
    # Batch Processing Models
    "BatchProcessingRequest", "BatchProcessingResponse",
    
    # Validators
    "FileValidators"
]
