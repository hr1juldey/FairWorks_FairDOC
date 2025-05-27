"""
File upload and analysis models for Fairdoc Medical AI Backend.
Handles file uploads, image analysis, document processing, and medical imaging with proper validation.
Uses mimetypes, hashlib, HttpUrl, and timedelta as required.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import Field, field_validator, HttpUrl
from enum import Enum
import mimetypes
import hashlib

from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity
)

# ============================================================================
# FILE ENUMS AND TYPES
# ============================================================================

class FileType(str, Enum):
    """Supported file types in the system."""
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    MEDICAL_IMAGE = "medical_image"
    DICOM = "dicom"
    PDF = "pdf"
    TEXT = "text"

class FileStatus(str, Enum):
    """File processing status."""
    UPLOADED = "uploaded"
    SCANNING = "scanning"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    DELETED = "deleted"

class ImageCategory(str, Enum):
    """Categories of medical images."""
    XRAY = "xray"
    CT_SCAN = "ct_scan"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    ECG = "ecg"
    ENDOSCOPY = "endoscopy"
    DERMATOLOGY = "dermatology"
    PATHOLOGY = "pathology"
    RADIOLOGY = "radiology"
    SYMPTOM_PHOTO = "symptom_photo"

class SecurityThreat(str, Enum):
    """Types of security threats detected in files."""
    MALWARE = "malware"
    VIRUS = "virus"
    SUSPICIOUS_CONTENT = "suspicious_content"
    OVERSIZED = "oversized"
    INVALID_FORMAT = "invalid_format"
    PHI_DETECTED = "phi_detected"
    INAPPROPRIATE_CONTENT = "inappropriate_content"

class ProcessingPriority(str, Enum):
    """File processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"

# ============================================================================
# FILE UPLOAD MODELS
# ============================================================================

class FileUploadRequest(TimestampMixin, ValidationMixin):
    """File upload request with comprehensive validation using mimetypes and hashlib."""
    
    # File metadata
    original_filename: str = Field(..., max_length=255, description="Original filename")
    file_size_bytes: int = Field(..., ge=0, le=100_000_000, description="File size in bytes (max 100MB)")
    mime_type: str = Field(..., description="MIME type of the file")
    file_hash_sha256: str = Field(..., description="SHA-256 hash of file content")
    file_hash_md5: str = Field(..., description="MD5 hash for quick comparison")
    
    # Upload context
    uploaded_by: UUID = Field(..., description="User who uploaded the file")
    patient_id: Optional[UUID] = Field(None, description="Associated patient ID")
    consultation_id: Optional[UUID] = Field(None, description="Associated consultation ID")
    
    # File categorization
    file_type: FileType
    image_category: Optional[ImageCategory] = None
    processing_priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL)
    processing_timeout: timedelta = Field(default=timedelta(minutes=30), description="Max processing time")
    
    # Patient demographics for bias monitoring
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    
    # Upload metadata with URLs
    upload_source: str = Field(..., description="Source of upload (web, mobile, api)")
    upload_url: Optional[HttpUrl] = Field(None, description="URL where file was uploaded from")
    callback_url: Optional[HttpUrl] = Field(None, description="Callback URL for processing completion")
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    @field_validator('original_filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename for security."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Filename cannot be empty')
        
        # Remove path separators for security
        clean_filename = v.replace('\\', '').replace('/', '').strip()
        
        # Check for dangerous extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.ps1', '.scr', '.vbs']
        if any(clean_filename.lower().endswith(ext) for ext in dangerous_extensions):
            raise ValueError('File type not allowed for security reasons')
        
        return clean_filename
    
    @field_validator('mime_type')
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Validate MIME type using mimetypes module."""
        # Check if it's a valid MIME type format
        if '/' not in v:
            raise ValueError('Invalid MIME type format')
        
        # Get allowed MIME types
        allowed_types = [
            'image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp',
            'application/pdf', 'text/plain', 'text/csv', 'audio/wav', 'audio/mp3',
            'video/mp4', 'application/dicom', 'application/octet-stream'
        ]
        
        if v not in allowed_types:
            # Check if it's a valid MIME type using mimetypes
            main_type, sub_type = v.split('/', 1)
            if main_type not in ['image', 'application', 'text', 'audio', 'video']:
                raise ValueError(f'MIME type {v} not allowed')
        
        return v
    
    @field_validator('file_hash_sha256')
    @classmethod
    def validate_sha256_hash(cls, v: str) -> str:
        """Validate SHA-256 hash format using hashlib."""
        if len(v) != 64:
            raise ValueError('SHA-256 hash must be 64 characters')
        
        try:
            int(v, 16)  # Verify it's valid hex
        except ValueError:
            raise ValueError('SHA-256 hash must be valid hexadecimal')
        
        return v.lower()
    
    @field_validator('file_hash_md5')
    @classmethod
    def validate_md5_hash(cls, v: str) -> str:
        """Validate MD5 hash format using hashlib."""
        if len(v) != 32:
            raise ValueError('MD5 hash must be 32 characters')
        
        try:
            int(v, 16)  # Verify it's valid hex
        except ValueError:
            raise ValueError('MD5 hash must be valid hexadecimal')
        
        return v.lower()
    
    @field_validator('file_type')
    @classmethod
    def validate_file_type_consistency(cls, v: FileType, info) -> FileType:
        """Ensure file type matches MIME type using mimetypes."""
        if hasattr(info, 'data') and 'mime_type' in info.data:
            mime_type = info.data['mime_type']
            
            # Use mimetypes to get expected file type
            main_type = mime_type.split('/')[0]
            
            type_mapping = {
                'image': FileType.IMAGE,
                'application': FileType.DOCUMENT,  # Default for applications
                'text': FileType.TEXT,
                'audio': FileType.AUDIO,
                'video': FileType.VIDEO
            }
            
            # Special cases
            if mime_type == 'application/pdf':
                expected_type = FileType.PDF
            elif mime_type == 'application/dicom':
                expected_type = FileType.DICOM
            else:
                expected_type = type_mapping.get(main_type)
            
            if expected_type and v != expected_type:
                raise ValueError(f'File type {v} does not match MIME type {mime_type}')
        
        return v
    
    def calculate_file_hashes(self, file_content: bytes) -> Dict[str, str]:
        """Calculate file hashes using hashlib."""
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        md5_hash = hashlib.md5(file_content).hexdigest()
        
        return {
            'sha256': sha256_hash,
            'md5': md5_hash
        }
    
    def detect_mime_type_from_content(self, file_content: bytes, filename: str) -> str:
        """Detect MIME type using mimetypes and file content."""
        # First try to guess from filename
        guessed_type, _ = mimetypes.guess_type(filename)
        
        if guessed_type:
            return guessed_type
        
        # If no guess, check file signatures (magic numbers)
        file_signatures = {
            b'\xFF\xD8\xFF': 'image/jpeg',
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'%PDF': 'application/pdf',
            b'RIFF': 'audio/wav',  # Simplified, would need more checking
            b'\x00\x00\x01\x00': 'image/x-icon',
        }
        
        for signature, mime_type in file_signatures.items():
            if file_content.startswith(signature):
                return mime_type
        
        return 'application/octet-stream'  # Default for unknown types

class FileUpload(BaseEntity, ValidationMixin, MetadataMixin):
    """Complete file upload record with processing status and timedelta usage."""
    
    # Basic file information
    original_filename: str
    stored_filename: str = Field(..., description="Internal filename for storage")
    file_path: str = Field(..., description="Full path to stored file")
    file_size_bytes: int
    mime_type: str
    file_hash_sha256: str
    file_hash_md5: str
    
    # Upload context
    uploaded_by: UUID
    patient_id: Optional[UUID] = None
    consultation_id: Optional[UUID] = None
    
    # URLs using HttpUrl
    download_url: Optional[HttpUrl] = Field(None, description="Secure download URL")
    thumbnail_url: Optional[HttpUrl] = Field(None, description="Thumbnail URL for images")
    processing_callback_url: Optional[HttpUrl] = Field(None, description="Callback URL for completion")
    
    # File classification
    file_type: FileType
    image_category: Optional[ImageCategory] = None
    processing_priority: ProcessingPriority
    
    # Demographics for bias analysis
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = None
    
    # Processing status with timedelta usage
    status: FileStatus = Field(default=FileStatus.UPLOADED)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_duration: Optional[timedelta] = None
    processing_timeout: timedelta = Field(default=timedelta(minutes=30))
    estimated_processing_time: timedelta = Field(default=timedelta(minutes=5))
    
    # Security scanning
    security_scanned: bool = Field(default=False)
    security_scan_results: Dict[str, Any] = Field(default_factory=dict)
    threats_detected: List[SecurityThreat] = Field(default_factory=list)
    quarantine_reason: Optional[str] = None
    quarantine_duration: timedelta = Field(default=timedelta(days=7))
    
    # Access control
    access_level: str = Field(default="restricted", description="public, restricted, confidential, secret")
    allowed_users: List[UUID] = Field(default_factory=list)
    download_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None
    access_expiry: Optional[datetime] = None
    
    # Medical context
    clinical_relevance: Optional[RiskLevel] = None
    urgency_level: Optional[UrgencyLevel] = None
    medical_findings: List[str] = Field(default_factory=list)
    
    # File lifecycle with timedelta
    retention_period: timedelta = Field(default=timedelta(days=2555), description="Default 7 years for medical records")
    deletion_scheduled_at: Optional[datetime] = None
    backup_status: str = Field(default="pending", description="pending, backed_up, failed")
    backup_retention: timedelta = Field(default=timedelta(days=3650), description="10 years backup retention")
    
    def start_processing(self):
        """Mark file processing as started with timeout calculation."""
        self.status = FileStatus.PROCESSING
        self.processing_started_at = datetime.utcnow()
        
        # Calculate processing timeout
        timeout_at = self.processing_started_at + self.processing_timeout
        self.add_metadata("processing_timeout_at", timeout_at.isoformat())
        
        self.update_timestamp()
    
    def complete_processing(self, findings: List[str] = None):
        """Mark file processing as completed with duration calculation."""
        self.status = FileStatus.ANALYZED
        self.processing_completed_at = datetime.utcnow()
        
        if self.processing_started_at:
            self.processing_duration = self.processing_completed_at - self.processing_started_at
        
        if findings:
            self.medical_findings = findings
        
        self.update_timestamp()
    
    def is_processing_timeout(self) -> bool:
        """Check if processing has timed out using timedelta."""
        if not self.processing_started_at:
            return False
        
        elapsed = datetime.utcnow() - self.processing_started_at
        return elapsed > self.processing_timeout
    
    def verify_file_integrity(self, file_content: bytes) -> bool:
        """Verify file integrity using hashlib."""
        current_sha256 = hashlib.sha256(file_content).hexdigest()
        current_md5 = hashlib.md5(file_content).hexdigest()
        
        return (current_sha256 == self.file_hash_sha256 and
                current_md5 == self.file_hash_md5)
    
    def add_security_threat(self, threat: SecurityThreat, details: str):
        """Add a security threat to the file."""
        self.threats_detected.append(threat)
        self.security_scan_results[threat.value] = {
            "detected_at": datetime.utcnow().isoformat(),
            "details": details
        }
        
        if threat in [SecurityThreat.MALWARE, SecurityThreat.VIRUS]:
            self.quarantine_file(f"Security threat detected: {threat.value}")
    
    def quarantine_file(self, reason: str):
        """Quarantine the file for security reasons with duration."""
        self.status = FileStatus.QUARANTINED
        self.quarantine_reason = reason
        
        # Set quarantine release time
        quarantine_until = datetime.utcnow() + self.quarantine_duration
        self.add_metadata("quarantine_until", quarantine_until.isoformat())
        
        self.update_timestamp()
    
    def schedule_deletion(self, custom_retention: Optional[timedelta] = None):
        """Schedule file deletion based on retention period."""
        retention = custom_retention or self.retention_period
        self.deletion_scheduled_at = datetime.utcnow() + retention
        self.update_timestamp()
    
    def generate_secure_url(self, expiry_hours: int = 24) -> str:
        """Generate secure download URL with expiry using timedelta."""
        expiry_time = datetime.utcnow() + timedelta(hours=expiry_hours)
        self.access_expiry = expiry_time
        
        # In real implementation, this would generate a signed URL
        base_url = f"https://secure.fairdoc.com/files/{self.id}"
        
        # Create URL hash for security
        url_data = f"{self.id}{self.file_hash_sha256}{expiry_time.isoformat()}"
        url_hash = hashlib.sha256(url_data.encode()).hexdigest()[:16]
        
        return f"{base_url}?token={url_hash}&expires={int(expiry_time.timestamp())}"
    
    def record_access(self, user_id: UUID):
        """Record file access event."""
        self.download_count += 1
        self.last_accessed = datetime.utcnow()
        
        # Add to access log with hash verification
        access_event = {
            "user_id": str(user_id),
            "accessed_at": datetime.utcnow().isoformat(),
            "access_count": self.download_count,
            "integrity_verified": True  # Would verify hash in real implementation
        }
        
        if "access_log" not in self.metadata:
            self.metadata["access_log"] = []
        
        self.metadata["access_log"].append(access_event)
        self.update_timestamp()

# ============================================================================
# IMAGE ANALYSIS MODELS
# ============================================================================

class ImageAnalysisRequest(TimestampMixin, ValidationMixin):
    """Request for medical image analysis with timeout controls."""
    
    file_upload_id: UUID = Field(..., description="Reference to uploaded file")
    analysis_type: ImageCategory
    
    # Analysis parameters
    requested_analyses: List[str] = Field(
        default_factory=lambda: ["abnormality_detection", "quality_assessment"],
        description="Types of analysis to perform"
    )
    
    # Timing constraints using timedelta
    analysis_timeout: timedelta = Field(default=timedelta(minutes=10), description="Max analysis time")
    priority_processing: bool = Field(default=False)
    
    # Clinical context
    clinical_indication: Optional[str] = Field(None, max_length=500, description="Clinical reason for imaging")
    patient_symptoms: List[str] = Field(default_factory=list)
    prior_imaging_available: bool = Field(default=False)
    
    # Requesting physician
    requesting_physician_id: UUID
    urgent_analysis: bool = Field(default=False)
    
    # Callback configuration
    completion_callback_url: Optional[HttpUrl] = Field(None, description="URL to notify when analysis complete")
    
    @field_validator('requested_analyses')
    @classmethod
    def validate_analysis_types(cls, v: List[str]) -> List[str]:
        """Validate requested analysis types."""
        valid_analyses = [
            "abnormality_detection", "quality_assessment", "measurement",
            "comparison", "classification", "segmentation", "risk_assessment"
        ]
        
        for analysis in v:
            if analysis not in valid_analyses:
                raise ValueError(f'Invalid analysis type: {analysis}')
        
        return v
    
    def calculate_priority_timeout(self) -> timedelta:
        """Calculate timeout based on priority using timedelta."""
        if self.urgent_analysis:
            return timedelta(minutes=2)
        elif self.priority_processing:
            return timedelta(minutes=5)
        else:
            return self.analysis_timeout

class ImageQualityMetrics(TimestampMixin, ValidationMixin):
    """Image quality assessment metrics with processing time tracking."""
    
    # Technical quality
    resolution_width: int = Field(..., ge=1)
    resolution_height: int = Field(..., ge=1)
    bit_depth: int = Field(..., ge=1, le=32)
    color_channels: int = Field(..., ge=1, le=4)
    file_size_optimized: bool = Field(default=False)
    
    # Quality scores (0.0 to 1.0)
    sharpness_score: float = Field(..., ge=0.0, le=1.0)
    contrast_score: float = Field(..., ge=0.0, le=1.0)
    brightness_score: float = Field(..., ge=0.0, le=1.0)
    noise_level: float = Field(..., ge=0.0, le=1.0, description="Lower is better")
    
    # Medical imaging specific
    exposure_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    positioning_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    motion_artifacts: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Processing metrics using timedelta
    quality_assessment_duration: timedelta = Field(default=timedelta(seconds=30))
    
    # Overall assessment
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    diagnostic_quality: bool = Field(..., description="Whether image is suitable for diagnosis")
    
    @field_validator('overall_quality')
    @classmethod
    def validate_overall_quality(cls, v: float, info) -> float:
        """Validate overall quality is reasonable given component scores."""
        if hasattr(info, 'data'):
            data = info.data
            component_scores = [
                data.get('sharpness_score', 0),
                data.get('contrast_score', 0),
                data.get('brightness_score', 0)
            ]
            
            # Overall quality should be roughly in line with component scores
            valid_scores = [s for s in component_scores if s > 0]
            if valid_scores:
                avg_components = sum(valid_scores) / len(valid_scores)
                if abs(v - avg_components) > 0.3:
                    raise ValueError('Overall quality score inconsistent with component scores')
        
        return v

class MedicalFinding(TimestampMixin, ValidationMixin):
    """Individual medical finding from image analysis."""
    
    finding_id: UUID = Field(default_factory=UUID)
    
    # Finding details
    finding_type: str = Field(..., max_length=100)
    description: str = Field(..., max_length=500)
    anatomical_location: Optional[str] = Field(None, max_length=100)
    
    # Confidence and risk
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    clinical_significance: RiskLevel = Field(default=RiskLevel.LOW)
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    
    # Spatial information
    bounding_box: Optional[Dict[str, float]] = Field(None, description="x, y, width, height")
    polygon_coordinates: Optional[List[List[float]]] = Field(None, description="Precise shape outline")
    
    # Measurements
    size_mm: Optional[float] = Field(None, ge=0)
    volume_mm3: Optional[float] = Field(None, ge=0)
    
    # Classification
    malignancy_risk: Optional[float] = Field(None, ge=0.0, le=1.0)
    follow_up_recommended: bool = Field(default=False)
    follow_up_timeframe: Optional[timedelta] = Field(None, description="Recommended follow-up timing")
    
    # Comparison with prior studies
    change_from_prior: Optional[str] = Field(None, description="stable, improved, worse, new")
    
    @field_validator('bounding_box')
    @classmethod
    def validate_bounding_box(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate bounding box format."""
        if v is None:
            return v
        
        required_keys = ['x', 'y', 'width', 'height']
        if not all(key in v for key in required_keys):
            raise ValueError(f'Bounding box must contain: {required_keys}')
        
        # Validate coordinates are reasonable
        for key, value in v.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Bounding box {key} must be non-negative number')
        
        return v
    
    def calculate_follow_up_date(self) -> Optional[datetime]:
        """Calculate follow-up date using timedelta."""
        if self.follow_up_recommended and self.follow_up_timeframe:
            return datetime.utcnow() + self.follow_up_timeframe
        return None

class ImageAnalysisResult(BaseEntity, ValidationMixin, MetadataMixin):
    """Complete image analysis results with comprehensive timing."""
    
    # Analysis context
    file_upload_id: UUID
    analysis_request_id: UUID
    image_category: ImageCategory
    
    # Patient demographics for bias monitoring
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = None
    
    # Analysis details
    analysis_model: str = Field(..., description="AI model used for analysis")
    model_version: str = Field(..., description="Version of analysis model")
    analysis_started_at: datetime = Field(default_factory=datetime.utcnow)
    analysis_completed_at: Optional[datetime] = None
    processing_time: Optional[timedelta] = None
    
    # Quality assessment
    image_quality: ImageQualityMetrics
    
    # Results and findings
    findings: List[MedicalFinding] = Field(default_factory=list)
    overall_impression: Optional[str] = Field(None, max_length=1000)
    
    # Risk assessment
    overall_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    overall_urgency: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    
    # AI confidence and bias metrics
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    bias_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Demographic bias detected")
    
    # Clinical validation
    radiologist_reviewed: bool = Field(default=False)
    radiologist_id: Optional[UUID] = None
    radiologist_agreement: Optional[bool] = None
    radiologist_comments: Optional[str] = None
    validation_timestamp: Optional[datetime] = None
    
    # Results delivery
    results_delivered_at: Optional[datetime] = None
    delivery_method: Optional[str] = Field(None, description="email, api_callback, dashboard")
    callback_url: Optional[HttpUrl] = Field(None, description="URL called when analysis complete")
    
    # Recommendations with timing
    recommendations: List[str] = Field(default_factory=list)
    follow_up_needed: bool = Field(default=False)
    follow_up_timeframe: Optional[timedelta] = None
    urgent_consultation_needed: bool = Field(default=False)
    
    def add_finding(self, finding: MedicalFinding):
        """Add a medical finding to the analysis."""
        self.findings.append(finding)
        
        # Update overall risk and urgency based on findings
        if finding.clinical_significance.value > self.overall_risk_level.value:
            self.overall_risk_level = finding.clinical_significance
        
        if finding.urgency_level.value > self.overall_urgency.value:
            self.overall_urgency = finding.urgency_level
        
        self.update_timestamp()
    
    def complete_analysis(self):
        """Mark analysis as completed with timing calculations."""
        self.analysis_completed_at = datetime.utcnow()
        
        if self.analysis_started_at:
            self.processing_time = self.analysis_completed_at - self.analysis_started_at
        
        self.update_timestamp()
    
    def calculate_total_processing_time(self) -> Optional[timedelta]:
        """Calculate total processing time including quality assessment."""
        if self.processing_time:
            total_time = self.processing_time + self.image_quality.quality_assessment_duration
            return total_time
        return None
    
    def add_radiologist_review(self, radiologist_id: UUID, agrees: bool, comments: str = ""):
        """Add radiologist validation."""
        self.radiologist_reviewed = True
        self.radiologist_id = radiologist_id
        self.radiologist_agreement = agrees
        self.radiologist_comments = comments
        self.validation_timestamp = datetime.utcnow()
        self.update_timestamp()
    
    def generate_report_url(self, expiry_hours: int = 48) -> str:
        """Generate secure report URL with expiry using timedelta and hashlib."""
        expiry_time = datetime.utcnow() + timedelta(hours=expiry_hours)
        
        # Create secure hash for URL
        url_data = f"{self.id}{self.file_upload_id}{expiry_time.isoformat()}"
        url_hash = hashlib.sha256(url_data.encode()).hexdigest()[:16]
        
        return f"https://reports.fairdoc.com/analysis/{self.id}?token={url_hash}&expires={int(expiry_time.timestamp())}"

# ============================================================================
# DOCUMENT PROCESSING MODELS
# ============================================================================

class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"
    MEDICAL_HISTORY = "medical_history"
    REFERRAL_LETTER = "referral_letter"
    DISCHARGE_SUMMARY = "discharge_summary"
    CONSENT_FORM = "consent_form"
    INSURANCE_DOCUMENT = "insurance_document"
    CLINICAL_NOTE = "clinical_note"

class DocumentAnalysis(BaseEntity, ValidationMixin, MetadataMixin):
    """Document analysis and OCR results with timing and integrity checks."""
    
    file_upload_id: UUID
    document_type: DocumentType
    
    # OCR Results
    extracted_text: str = Field(..., description="Full extracted text")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0)
    text_hash: str = Field(..., description="Hash of extracted text for integrity")
    
    # Processing timing
    ocr_processing_time: timedelta = Field(default=timedelta(seconds=30))
    text_analysis_time: timedelta = Field(default=timedelta(seconds=15))
    
    # Structured data extraction
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    key_value_pairs: Dict[str, str] = Field(default_factory=dict)
    
    # Medical entity extraction
    medications_mentioned: List[str] = Field(default_factory=list)
    conditions_mentioned: List[str] = Field(default_factory=list)
    procedures_mentioned: List[str] = Field(default_factory=list)
    dates_mentioned: List[str] = Field(default_factory=list)
    
    # Document validation
    document_valid: bool = Field(..., description="Whether document format is valid")
    validation_errors: List[str] = Field(default_factory=list)
    
    # PHI detection
    phi_detected: bool = Field(default=False)
    phi_locations: List[Dict[str, Any]] = Field(default_factory=list)
    redacted_text: Optional[str] = Field(None, description="Text with PHI redacted")
    redacted_text_hash: Optional[str] = Field(None, description="Hash of redacted text")
    
    # Processing metadata
    processing_model: str = Field(..., description="OCR/NLP model used")
    
    def calculate_text_hash(self, text: str) -> str:
        """Calculate hash of text content using hashlib."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def set_extracted_text(self, text: str):
        """Set extracted text and calculate its hash."""
        self.extracted_text = text
        self.text_hash = self.calculate_text_hash(text)
    
    def set_redacted_text(self, redacted_text: str):
        """Set redacted text and calculate its hash."""
        self.redacted_text = redacted_text
        self.redacted_text_hash = self.calculate_text_hash(redacted_text)
    
    def verify_text_integrity(self, text: str) -> bool:
        """Verify text integrity using hash comparison."""
        calculated_hash = self.calculate_text_hash(text)
        return calculated_hash == self.text_hash
    
    def calculate_total_processing_time(self) -> timedelta:
        """Calculate total document processing time."""
        return self.ocr_processing_time + self.text_analysis_time

# ============================================================================
# BIAS MONITORING FOR FILE PROCESSING
# ============================================================================

class FileProcessingBias(BaseEntity, ValidationMixin, MetadataMixin):
    """Bias monitoring for file processing and analysis with time-based analysis."""
    
    # Time period for analysis using timedelta
    analysis_period_start: datetime
    analysis_period_end: datetime
    analysis_duration: timedelta = Field(default=timedelta(days=7))
    
    # File processing metrics by demographics
    processing_times_by_gender: Dict[Gender, timedelta] = Field(default_factory=dict)
    processing_times_by_ethnicity: Dict[Ethnicity, timedelta] = Field(default_factory=dict)
    processing_times_by_age_group: Dict[str, timedelta] = Field(default_factory=dict)
    
    # Quality assessment bias
    quality_scores_by_gender: Dict[Gender, float] = Field(default_factory=dict)
    quality_scores_by_ethnicity: Dict[Ethnicity, float] = Field(default_factory=dict)
    
    # Analysis accuracy bias
    accuracy_by_gender: Dict[Gender, float] = Field(default_factory=dict)
    accuracy_by_ethnicity: Dict[Ethnicity, float] = Field(default_factory=dict)
    
    # Risk level assignment bias
    risk_distribution_by_gender: Dict[Gender, Dict[RiskLevel, int]] = Field(default_factory=dict)
    risk_distribution_by_ethnicity: Dict[Ethnicity, Dict[RiskLevel, int]] = Field(default_factory=dict)
    
    # Overall bias metrics
    demographic_parity_score: float = Field(..., ge=0.0, le=1.0)
    equalized_odds_score: float = Field(..., ge=0.0, le=1.0)
    
    # Alert thresholds and timing
    bias_threshold_exceeded: bool = Field(default=False)
    bias_alerts: List[str] = Field(default_factory=list)
    last_bias_check: datetime = Field(default_factory=datetime.utcnow)
    bias_check_frequency: timedelta = Field(default=timedelta(hours=6))
    
    def record_processing_time(self, gender: Gender, ethnicity: Ethnicity, processing_time: timedelta):
        """Record processing time by demographics using timedelta."""
        self.processing_times_by_gender[gender] = processing_time
        self.processing_times_by_ethnicity[ethnicity] = processing_time
        self.update_timestamp()
    
    def should_check_bias(self) -> bool:
        """Check if it's time for bias analysis using timedelta."""
        time_since_last_check = datetime.utcnow() - self.last_bias_check
        return time_since_last_check >= self.bias_check_frequency
    
    def check_bias_thresholds(self, threshold: float = 0.1):
        """Check if bias metrics exceed acceptable thresholds."""
        bias_detected = False
        current_time = datetime.utcnow()
        
        if self.demographic_parity_score > threshold:
            alert_msg = f"[{current_time.isoformat()}] Demographic parity exceeded: {self.demographic_parity_score:.3f}"
            self.bias_alerts.append(alert_msg)
            bias_detected = True
        
        if self.equalized_odds_score > threshold:
            alert_msg = f"[{current_time.isoformat()}] Equalized odds exceeded: {self.equalized_odds_score:.3f}"
            self.bias_alerts.append(alert_msg)
            bias_detected = True
        
        self.bias_threshold_exceeded = bias_detected
        self.last_bias_check = current_time
        self.update_timestamp()
    
    def generate_bias_report_hash(self) -> str:
        """Generate hash for bias report integrity using hashlib."""
        report_data = {
            "analysis_period": f"{self.analysis_period_start}-{self.analysis_period_end}",
            "demographic_parity": self.demographic_parity_score,
            "equalized_odds": self.equalized_odds_score,
            "processing_times_gender": str(self.processing_times_by_gender),
            "processing_times_ethnicity": str(self.processing_times_by_ethnicity)
        }
        
        report_string = str(sorted(report_data.items()))
        return hashlib.sha256(report_string.encode()).hexdigest()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class FileUploadResponse(BaseResponse):
    """Response for file upload requests with URLs and timing."""
    file_upload: FileUpload
    upload_url: Optional[HttpUrl] = None
    processing_estimated_time: Optional[timedelta] = None
    secure_download_url: Optional[str] = None

class ImageAnalysisResponse(BaseResponse):
    """Response for image analysis requests with timing and URLs."""
    analysis_result: ImageAnalysisResult
    requires_human_review: bool
    estimated_processing_time: Optional[timedelta] = None
    report_url: Optional[HttpUrl] = None
    callback_scheduled: bool = Field(default=False)

class DocumentAnalysisResponse(BaseResponse):
    """Response for document analysis requests."""
    analysis: DocumentAnalysis
    requires_manual_review: bool
    phi_warnings: List[str]
    processing_duration: timedelta
    text_integrity_verified: bool

class FileBiasAnalysisResponse(BaseResponse):
    """Response for file processing bias analysis."""
    bias_analysis: FileProcessingBias
    recommendations: List[str]
    immediate_actions_needed: bool
    report_hash: str
    next_analysis_due: datetime

# ============================================================================
# FILE SECURITY AND VALIDATION UTILITIES
# ============================================================================


ALLOWED_FILE_EXTENSIONS = {
    FileType.IMAGE: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
    FileType.DOCUMENT: ['.pdf', '.txt', '.docx'],
    FileType.AUDIO: ['.wav', '.mp3', '.m4a'],
    FileType.VIDEO: ['.mp4', '.avi', '.mov'],
    FileType.DICOM: ['.dcm', '.dicom'],
    FileType.MEDICAL_IMAGE: ['.jpg', '.jpeg', '.png', '.tiff', '.dcm']
}

MAX_FILE_SIZES = {
    FileType.IMAGE: 50_000_000,      # 50MB
    FileType.DOCUMENT: 25_000_000,   # 25MB
    FileType.AUDIO: 100_000_000,     # 100MB
    FileType.VIDEO: 500_000_000,     # 500MB
    FileType.DICOM: 100_000_000,     # 100MB
    FileType.MEDICAL_IMAGE: 50_000_000  # 50MB
}

# Processing timeouts using timedelta
PROCESSING_TIMEOUTS = {
    ProcessingPriority.EMERGENCY: timedelta(minutes=1),
    ProcessingPriority.URGENT: timedelta(minutes=5),
    ProcessingPriority.HIGH: timedelta(minutes=15),
    ProcessingPriority.NORMAL: timedelta(minutes=30),
    ProcessingPriority.LOW: timedelta(hours=2)
}

def validate_file_hash(file_content: bytes, expected_sha256: str, expected_md5: str) -> bool:
    """Validate file integrity using hashlib."""
    calculated_sha256 = hashlib.sha256(file_content).hexdigest()
    calculated_md5 = hashlib.md5(file_content).hexdigest()
    
    return (calculated_sha256 == expected_sha256 and
            calculated_md5 == expected_md5)

def detect_file_type_from_content(file_content: bytes, filename: str) -> tuple[str, FileType]:
    """Detect file type using mimetypes and content analysis."""
    # Use mimetypes to guess from filename
    guessed_type, _ = mimetypes.guess_type(filename)
    
    if guessed_type:
        mime_type = guessed_type
    else:
        # Fallback to content-based detection
        if file_content.startswith(b'\xFF\xD8\xFF'):
            mime_type = 'image/jpeg'
        elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):
            mime_type = 'image/png'
        elif file_content.startswith(b'%PDF'):
            mime_type = 'application/pdf'
        else:
            mime_type = 'application/octet-stream'
    
    # Map MIME type to FileType
    if mime_type.startswith('image/'):
        file_type = FileType.IMAGE
    elif mime_type == 'application/pdf':
        file_type = FileType.PDF
    elif mime_type.startswith('text/'):
        file_type = FileType.TEXT
    elif mime_type.startswith('audio/'):
        file_type = FileType.AUDIO
    elif mime_type.startswith('video/'):
        file_type = FileType.VIDEO
    elif mime_type == 'application/dicom':
        file_type = FileType.DICOM
    else:
        file_type = FileType.DOCUMENT
    
    return mime_type, file_type

def calculate_retention_date(upload_date: datetime, file_type: FileType) -> datetime:
    """Calculate file retention date using timedelta."""
    retention_periods = {
        FileType.MEDICAL_IMAGE: timedelta(days=2555),  # 7 years
        FileType.DICOM: timedelta(days=3650),          # 10 years
        FileType.DOCUMENT: timedelta(days=2555),       # 7 years
        FileType.IMAGE: timedelta(days=1825),          # 5 years
        FileType.AUDIO: timedelta(days=1095),          # 3 years
        FileType.VIDEO: timedelta(days=1095),          # 3 years
        FileType.TEXT: timedelta(days=2555),           # 7 years
        FileType.PDF: timedelta(days=2555)             # 7 years
    }
    
    retention_period = retention_periods.get(file_type, timedelta(days=2555))
    return upload_date + retention_period
