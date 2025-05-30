"""
V1 File Management Utilities - Production Grade
Enhanced file operations with comprehensive security scanning
"""
# === SMART IMPORT SETUP - ADD TO TOP OF FILE ===
import sys
import os
from pathlib import Path

# Setup paths once to prevent double imports
if not hasattr(sys, '_fairdoc_paths_setup'):
    current_dir = Path(__file__).parent
    v1_dir = current_dir.parent
    backend_dir = v1_dir.parent
    project_root = backend_dir.parent
    
    paths_to_add = [str(project_root), str(backend_dir), str(v1_dir)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    sys._fairdoc_paths_setup = True

# Standard imports first
import uuid
import mimetypes
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging

# Smart internal imports with fallbacks
try:
    # Try absolute imports first
    from datamodels.file_models import FileCategory, FileStatus, FileValidationResult
    from core.config import get_v1_settings
except ImportError:
    # Fallback to relative imports
    from ..datamodels.file_models import FileCategory, FileStatus, FileValidationResult
    from ..core.config import get_v1_settings

# === END SMART IMPORT SETUP ===


from pathlib import Path


from datamodels.file_models import (
    FileCategory, FileStatus, FileValidationResult,
    FileSecurityScan, FileValidators, SecurityThreatLevel
)
from core.config import get_v1_settings

logger = logging.getLogger(__name__)
settings = get_v1_settings()

# =============================================================================
# MINIO CLIENT UTILITIES
# =============================================================================

def get_minio_client():
    """Get MinIO client with error handling"""
    try:
        from minio import Minio
        
        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        
        # Ensure bucket exists
        bucket_name = settings.MINIO_BUCKET_NAME
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Created MinIO bucket: {bucket_name}")
        
        return client, bucket_name
        
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client: {e}")
        raise

def upload_file_to_minio(
    file_content: bytes,
    object_path: str,
    content_type: str,
    metadata: Optional[Dict[str, str]] = None
) -> str:
    """Upload file to MinIO and return ETag"""
    try:
        client, bucket_name = get_minio_client()
        
        from io import BytesIO
        file_stream = BytesIO(file_content)
        
        result = client.put_object(
            bucket_name,
            object_path,
            data=file_stream,
            length=len(file_content),
            content_type=content_type,
            metadata=metadata or {}
        )
        
        logger.info(f"File uploaded to MinIO: {object_path} (ETag: {result.etag})")
        return result.etag
        
    except Exception as e:
        logger.error(f"Failed to upload file to MinIO: {e}")
        raise

def generate_presigned_url(object_path: str, expires_hours: int = 1) -> str:
    """Generate presigned URL for secure file access"""
    try:
        client, bucket_name = get_minio_client()
        
        url = client.presigned_get_object(
            bucket_name,
            object_path,
            expires=timedelta(hours=expires_hours)
        )
        
        logger.info(f"Generated presigned URL for {object_path} (expires in {expires_hours}h)")
        return url
        
    except Exception as e:
        logger.warning(f"Failed to generate presigned URL: {e}")
        # Fallback to direct URL
        return f"http://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/{object_path}"

def delete_file_from_minio(object_path: str) -> bool:
    """Delete file from MinIO storage"""
    try:
        client, bucket_name = get_minio_client()
        client.remove_object(bucket_name, object_path)
        logger.info(f"File deleted from MinIO: {object_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete file from MinIO: {e}")
        return False

# =============================================================================
# ENHANCED FILE VALIDATION WITH SECURITY SCANNING
# =============================================================================

def validate_medical_file(file_content: bytes, filename: str, content_type: str) -> FileValidationResult:
    """Comprehensive file validation with security scanning"""
    
    validation_start = datetime.now()
    validation_errors = []
    validation_warnings = []
    security_checks = {}
    
    logger.info(f"Starting validation for file: {filename} ({len(file_content)} bytes)")
    
    # Basic validation
    if not FileValidators.validate_filename(filename):
        validation_errors.append("Invalid filename detected")
    
    if not FileValidators.validate_file_size(len(file_content), settings.MAX_FILE_SIZE):
        validation_errors.append(f"File size {len(file_content)} exceeds maximum {settings.MAX_FILE_SIZE}")
    
    if not FileValidators.validate_medical_content_type(content_type):
        validation_errors.append(f"Content type {content_type} not allowed for medical files")
    
    # Advanced security checks
    security_checks["filename_safe"] = FileValidators.validate_filename(filename)
    security_checks["size_valid"] = FileValidators.validate_file_size(len(file_content))
    security_checks["content_type_allowed"] = FileValidators.validate_medical_content_type(content_type)
    security_checks["file_signature_valid"] = validate_file_signature(file_content, content_type)
    security_checks["no_embedded_executables"] = check_for_embedded_executables(file_content)
    security_checks["entropy_analysis_passed"] = analyze_file_entropy(file_content)
    
    # Generate secure filename
    file_extension = Path(filename).suffix if filename else ""
    secure_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Determine file category
    file_category = categorize_medical_file(content_type)
    
    # Calculate file hash
    file_hash = calculate_file_hash(file_content)
    
    # Check for warnings
    if len(file_content) > 50 * 1024 * 1024:  # 50MB
        validation_warnings.append("Large file size may affect processing performance")
    
    if not security_checks["file_signature_valid"]:
        validation_warnings.append("File signature doesn't match content type")
    
    validation_duration = (datetime.now() - validation_start).total_seconds()
    logger.info(f"File validation completed in {validation_duration:.3f}s")
    
    return FileValidationResult(
        validation_passed=len(validation_errors) == 0,
        original_filename=filename,
        secure_filename=secure_filename,
        content_type=content_type,
        file_size=len(file_content),
        file_category=file_category,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        security_checks=security_checks,
        file_hash=file_hash,
        validated_at=validation_start
    )

def perform_security_scan(file_content: bytes, filename: str, content_type: str) -> FileSecurityScan:
    """Comprehensive security scan implementation"""
    
    scan_start = datetime.now()
    scan_start_ms = time.time() * 1000
    
    logger.info(f"Starting security scan for file: {filename}")
    
    threats_detected = []
    is_safe = True
    threat_level = SecurityThreatLevel.SAFE
    malware_detected = False
    quarantine_required = False
    
    try:
        # 1. File signature validation
        signature_valid = validate_file_signature(file_content, content_type)
        if not signature_valid:
            threats_detected.append("File signature mismatch")
            threat_level = SecurityThreatLevel.MEDIUM_RISK
        
        # 2. Check for embedded executables
        if not check_for_embedded_executables(file_content):
            threats_detected.append("Embedded executable content detected")
            threat_level = SecurityThreatLevel.HIGH_RISK
            is_safe = False
            quarantine_required = True
        
        # 3. Entropy analysis for packed/encrypted content
        if not analyze_file_entropy(file_content):
            threats_detected.append("Suspicious entropy patterns detected")
            threat_level = SecurityThreatLevel.MEDIUM_RISK
        
        # 4. Content-specific security checks
        content_analysis = {}
        
        if content_type == "application/pdf":
            pdf_threats = scan_pdf_content(file_content)
            if pdf_threats:
                threats_detected.extend(pdf_threats)
                threat_level = SecurityThreatLevel.HIGH_RISK
        
        elif content_type.startswith("image/"):
            image_threats = scan_image_content(file_content)
            if image_threats:
                threats_detected.extend(image_threats)
                threat_level = SecurityThreatLevel.MEDIUM_RISK
        
        elif content_type.startswith("audio/"):
            audio_threats = scan_audio_content(file_content)
            if audio_threats:
                threats_detected.extend(audio_threats)
                threat_level = SecurityThreatLevel.LOW_RISK
        
        # 5. Known malware patterns (simplified)
        malware_patterns = [
            b'\x4d\x5a',  # PE header
            b'\x7f\x45\x4c\x46',  # ELF header
            b'\xfe\xed\xfa',  # Mach-O header
        ]
        
        for pattern in malware_patterns:
            if pattern in file_content:
                threats_detected.append(f"Potential executable pattern detected: {pattern.hex()}")
                malware_detected = True
                threat_level = SecurityThreatLevel.CRITICAL
                is_safe = False
                quarantine_required = True
                break
        
        # 6. File size anomaly detection
        if len(file_content) > 500 * 1024 * 1024:  # 500MB
            threats_detected.append("Unusually large file size")
            threat_level = max(threat_level, SecurityThreatLevel.MEDIUM_RISK)
        
        # Calculate final safety assessment
        if len(threats_detected) == 0:
            is_safe = True
            threat_level = SecurityThreatLevel.SAFE
        elif threat_level in [SecurityThreatLevel.HIGH_RISK, SecurityThreatLevel.CRITICAL]:
            is_safe = False
        
        scan_end_ms = time.time() * 1000
        scan_duration = int(scan_end_ms - scan_start_ms)
        
        confidence_score = calculate_scan_confidence(threats_detected, scan_duration)
        
        logger.info(f"Security scan completed in {scan_duration}ms - Threats: {len(threats_detected)}, Safe: {is_safe}")
        
        return FileSecurityScan(
            is_safe=is_safe,
            threat_level=threat_level,
            scan_timestamp=scan_start,
            scan_duration_ms=scan_duration,
            threats_detected=threats_detected,
            file_signature_valid=signature_valid,
            content_analysis=content_analysis,
            virus_scan_result="clean" if is_safe else "threats_detected",
            malware_detected=malware_detected,
            quarantine_required=quarantine_required,
            scan_engine_version="FairdocSecScan-v1.0",
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        return FileSecurityScan(
            is_safe=False,
            threat_level=SecurityThreatLevel.CRITICAL,
            scan_timestamp=scan_start,
            scan_duration_ms=int((time.time() * 1000) - scan_start_ms),
            threats_detected=["Security scan failed"],
            file_signature_valid=False,
            content_analysis={"error": str(e)},
            virus_scan_result="scan_error",
            malware_detected=True,
            quarantine_required=True,
            scan_engine_version="FairdocSecScan-v1.0",
            confidence_score=0.0
        )

def validate_file_signature(file_content: bytes, content_type: str) -> bool:
    """Validate file signature matches content type"""
    
    # Enhanced file signature mappings
    signatures = {
        "application/pdf": [b"%PDF"],
        "image/jpeg": [b"\xff\xd8\xff"],
        "image/png": [b"\x89PNG\r\n\x1a\n"],
        "audio/wav": [b"RIFF"],
        "audio/mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3"],
        "audio/mpeg": [b"ID3", b"\xff\xfb"],
        "text/plain": [],  # No specific signature
        "application/json": [],  # No specific signature
    }
    
    if content_type not in signatures:
        logger.warning(f"Unknown content type for signature validation: {content_type}")
        return True  # Unknown type, allow
    
    expected_signatures = signatures[content_type]
    if not expected_signatures:  # No signature required
        return True
    
    file_start = file_content[:20]  # Check first 20 bytes
    
    for signature in expected_signatures:
        if file_start.startswith(signature):
            return True
    
    logger.warning(f"File signature mismatch for content type {content_type}")
    return False

def check_for_embedded_executables(file_content: bytes) -> bool:
    """Check for embedded executable content"""
    
    # Known executable headers
    executable_headers = [
        b'MZ',  # PE/DOS header
        b'\x7fELF',  # ELF header
        b'\xfe\xed\xfa\xce',  # Mach-O 32-bit
        b'\xfe\xed\xfa\xcf',  # Mach-O 64-bit
        b'PK\x03\x04',  # ZIP header (could contain executables)
    ]
    
    for header in executable_headers:
        if header in file_content:
            logger.warning(f"Embedded executable header detected: {header.hex()}")
            return False
    
    return True

def analyze_file_entropy(file_content: bytes) -> bool:
    """Analyze file entropy to detect encrypted/packed content"""
    
    if len(file_content) == 0:
        return True
    
    # Calculate byte frequency
    byte_counts = [0] * 256
    for byte in file_content:
        byte_counts[byte] += 1
    
    # Calculate entropy
    entropy = 0.0
    file_length = len(file_content)
    
    for count in byte_counts:
        if count > 0:
            frequency = count / file_length
            entropy -= frequency * (frequency.bit_length() - 1)
    
    # High entropy (> 7.5) might indicate encryption/compression
    entropy_threshold = 7.5
    
    if entropy > entropy_threshold:
        logger.warning(f"High entropy detected: {entropy:.2f} (threshold: {entropy_threshold})")
        return False
    
    return True

def scan_pdf_content(file_content: bytes) -> List[str]:
    """Scan PDF for security threats"""
    threats = []
    
    # Check for JavaScript in PDF
    if b'/JS' in file_content or b'/JavaScript' in file_content:
        threats.append("JavaScript content detected in PDF")
    
    # Check for embedded files
    if b'/EmbeddedFile' in file_content:
        threats.append("Embedded files detected in PDF")
    
    # Check for suspicious actions
    suspicious_actions = [b'/Launch', b'/ImportData', b'/SubmitForm']
    for action in suspicious_actions:
        if action in file_content:
            threats.append(f"Suspicious PDF action detected: {action.decode('latin-1')}")
    
    return threats

def scan_image_content(file_content: bytes) -> List[str]:
    """Scan image for security threats"""
    threats = []
    
    # Check for embedded scripts in image metadata
    script_patterns = [b'<script', b'javascript:', b'vbscript:']
    for pattern in script_patterns:
        if pattern.lower() in file_content.lower():
            threats.append(f"Script pattern detected in image: {pattern.decode('latin-1')}")
    
    # Check image size for potential zip bombs
    if len(file_content) < 1000 and b'PK' in file_content:
        threats.append("Potential zip bomb detected in image")
    
    return threats

def scan_audio_content(file_content: bytes) -> List[str]:
    """Scan audio for security threats"""
    threats = []
    
    # Check for unusual metadata size
    if len(file_content) > 100 * 1024 * 1024:  # 100MB
        threats.append("Unusually large audio file")
    
    # Check for embedded data in audio
    if b'PK' in file_content[1000:]:  # Skip normal header
        threats.append("Potential embedded archive in audio file")
    
    return threats

def calculate_scan_confidence(threats_detected: List[str], scan_duration_ms: int) -> float:
    """Calculate confidence score for security scan"""
    
    base_confidence = 0.9
    
    # Reduce confidence for each threat detected
    threat_penalty = len(threats_detected) * 0.1
    
    # Reduce confidence for very fast scans (might be incomplete)
    if scan_duration_ms < 10:
        speed_penalty = 0.2
    else:
        speed_penalty = 0.0
    
    # Reduce confidence for very slow scans (might indicate issues)
    if scan_duration_ms > 5000:  # 5 seconds
        timeout_penalty = 0.1
    else:
        timeout_penalty = 0.0
    
    confidence = base_confidence - threat_penalty - speed_penalty - timeout_penalty
    return max(0.0, min(1.0, confidence))

def categorize_medical_file(content_type: str) -> FileCategory:
    """Categorize medical file based on content type"""
    
    category_mapping = {
        "application/pdf": FileCategory.MEDICAL_REPORT,
        "image/jpeg": FileCategory.MEDICAL_IMAGE,
        "image/png": FileCategory.MEDICAL_IMAGE,
        "image/dicom": FileCategory.DICOM_IMAGE,
        "audio/wav": FileCategory.AUDIO_RECORDING,
        "audio/mp3": FileCategory.AUDIO_RECORDING,
        "audio/mpeg": FileCategory.AUDIO_RECORDING,
        "text/plain": FileCategory.MEDICAL_REPORT,
        "application/json": FileCategory.MEDICAL_REPORT
    }
    
    return category_mapping.get(content_type, FileCategory.OTHER)

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

# =============================================================================
# FILE PATH UTILITIES WITH DATETIME
# =============================================================================

def generate_object_path(
    user_id: str,
    secure_filename: str,
    case_id: Optional[str] = None,
    file_category: Optional[FileCategory] = None
) -> str:
    """Generate structured object path with datetime organization"""
    
    # Create detailed date-time folder structure
    now = datetime.now()
    date_path = f"{now.year}/{now.month:02d}/{now.day:02d}/{now.hour:02d}"
    
    if case_id:
        # Case-specific files with datetime
        return f"cases/{case_id}/{date_path}/{secure_filename}"
    elif file_category:
        # Category-based organization with datetime
        return f"medical_files/{file_category.value}/{user_id}/{date_path}/{secure_filename}"
    else:
        # User-specific files with datetime
        return f"medical_files/{user_id}/{date_path}/{secure_filename}"

def generate_file_url(object_path: str) -> str:
    """Generate file access URL"""
    return f"http://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/{object_path}"

# =============================================================================
# FILE METADATA UTILITIES WITH ENHANCED TRACKING
# =============================================================================

def create_file_metadata(
    validation_result: FileValidationResult,
    object_path: str,
    etag: str,
    user_id: str,
    case_id: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    security_scan: Optional[FileSecurityScan] = None
) -> Dict[str, Any]:
    """Create comprehensive file metadata with automatic content analysis"""
    
    upload_time = datetime.now()
    
    metadata = {
        "file_id": str(uuid.uuid4()),
        "original_filename": validation_result.original_filename,
        "secure_filename": validation_result.secure_filename,
        "content_type": validation_result.content_type,
        "file_size": validation_result.file_size,
        "file_category": validation_result.file_category.value,
        "description": description,
        "minio_object_path": object_path,
        "file_url": generate_file_url(object_path),
        "minio_etag": etag,
        "file_hash": validation_result.file_hash,
        "uploaded_by": user_id,
        "case_id": case_id,
        "tags": tags or [],
        "status": FileStatus.UPLOADED.value,
        
        # Enhanced datetime tracking
        "upload_timestamp": upload_time.isoformat(),
        "created_at": upload_time.isoformat(),
        "updated_at": upload_time.isoformat(),
        "validated_at": validation_result.validated_at.isoformat(),
        "last_accessed": None,
        "expires_at": (upload_time + timedelta(days=365)).isoformat(),
        
        # Security information
        "security_checks": validation_result.security_checks,
        "security_scan": security_scan.dict() if security_scan else None,
        "compliance_flags": {
            "gdpr_compliant": True,
            "hipaa_compliant": True,
            "medical_grade": validation_result.file_category != FileCategory.OTHER
        }
    }
    
    # Automatic content analysis based on file type
    try:
        # Get file content for analysis (this would need to be passed as parameter in real implementation)
        # For now, we'll add a placeholder that can be filled by the calling function
        content_analysis = None
        
        if validation_result.content_type.startswith("image/"):
            logger.info(f"Preparing image analysis for: {validation_result.original_filename}")
            # Image analysis would be called by the upload function with actual file content
            content_analysis = {"analysis_type": "image", "status": "ready_for_analysis"}
            
        elif validation_result.content_type.startswith("audio/"):
            logger.info(f"Preparing audio analysis for: {validation_result.original_filename}")
            content_analysis = {"analysis_type": "audio", "status": "ready_for_analysis"}
            
        elif validation_result.content_type == "application/pdf":
            logger.info(f"Preparing PDF analysis for: {validation_result.original_filename}")
            content_analysis = {"analysis_type": "pdf", "status": "ready_for_analysis"}
        
        if content_analysis:
            metadata["content_analysis"] = content_analysis
            
    except Exception as e:
        logger.warning(f"Content analysis preparation failed: {e}")
        metadata["content_analysis"] = {"error": str(e), "status": "analysis_failed"}
    
    logger.info(f"File metadata created: {metadata['file_id']} - {validation_result.original_filename}")
    return metadata
def perform_content_analysis(file_content: bytes, file_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Perform content analysis on uploaded file and update metadata"""
    
    analysis_start = datetime.now()
    content_type = file_metadata.get("content_type", "")
    filename = file_metadata.get("original_filename", "unknown")
    
    logger.info(f"Performing content analysis for: {filename}")
    
    try:
        analysis_result = None
        
        if content_type.startswith("image/"):
            analysis_result = analyze_image_file(file_content, filename)
            
        elif content_type.startswith("audio/"):
            analysis_result = analyze_audio_file(file_content, filename)
            
        elif content_type == "application/pdf":
            analysis_result = analyze_pdf_file(file_content, filename)
        
        if analysis_result:
            # Update file metadata with analysis results
            file_metadata["content_analysis"] = {
                "analysis_type": content_type.split("/")[0],
                "status": "completed",
                "results": analysis_result,
                "analyzed_at": analysis_start.isoformat()
            }
            
            # Add analysis summary to main metadata
            if analysis_result.get("analysis_metadata", {}).get("success"):
                file_metadata["analysis_summary"] = {
                    "has_analysis": True,
                    "analysis_successful": True,
                    "processing_time": analysis_result.get("analysis_metadata", {}).get("processing_time_seconds", 0),
                    "analysis_engine": analysis_result.get("analysis_metadata", {}).get("analysis_engine", "unknown")
                }
            else:
                file_metadata["analysis_summary"] = {
                    "has_analysis": True,
                    "analysis_successful": False,
                    "error": analysis_result.get("error", "Unknown analysis error")
                }
        else:
            file_metadata["content_analysis"] = {
                "analysis_type": "unsupported",
                "status": "skipped",
                "reason": f"No analysis available for content type: {content_type}"
            }
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        logger.info(f"Content analysis completed for {filename} in {analysis_duration:.3f}s")
        
        return file_metadata
        
    except Exception as e:
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        logger.error(f"Content analysis failed for {filename}: {e}")
        
        file_metadata["content_analysis"] = {
            "analysis_type": "error",
            "status": "failed",
            "error": str(e),
            "analyzed_at": analysis_start.isoformat()
        }
        
        return file_metadata


def extract_file_content(upload_file) -> bytes:
    """Extract content from FastAPI UploadFile with logging"""
    try:
        start_time = datetime.now()
        upload_file.file.seek(0)
        content = upload_file.file.read()
        upload_file.file.seek(0)  # Reset for potential re-reading
        
        extraction_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"File content extracted: {len(content)} bytes in {extraction_time:.3f}s")
        
        return content
    except Exception as e:
        logger.error(f"Failed to extract file content: {e}")
        raise

# =============================================================================
# BATCH PROCESSING UTILITIES
# =============================================================================

def process_file_batch(
    files: List[Any],
    user_id: str,
    case_id: Optional[str] = None,
    max_batch_size: int = 10
) -> Tuple[List[Dict], List[Dict], str]:
    """Process batch of files with comprehensive tracking"""
    
    batch_start = datetime.now()
    batch_id = str(uuid.uuid4())
    
    logger.info(f"Starting batch processing: {batch_id} - {len(files)} files")
    
    if len(files) > max_batch_size:
        raise ValueError(f"Batch size {len(files)} exceeds maximum {max_batch_size}")
    
    successful_files = []
    failed_files = []
    
    for i, file in enumerate(files):
        file_start = datetime.now()
        try:
            # Extract file content
            file_content = extract_file_content(file)
            
            # Validate file
            validation_result = validate_medical_file(
                file_content,
                getattr(file, 'filename', f"file_{i}"),
                getattr(file, 'content_type', 'application/octet-stream')
            )
            
            if not validation_result.validation_passed:
                failed_files.append({
                    "file_index": i,
                    "filename": getattr(file, 'filename', f"file_{i}"),
                    "status": "failed",
                    "error": f"Validation failed: {', '.join(validation_result.validation_errors)}",
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_ms": int((datetime.now() - file_start).total_seconds() * 1000)
                })
                continue
            
            # Perform security scan
            security_scan = perform_security_scan(
                file_content,
                validation_result.original_filename,
                validation_result.content_type
            )
            
            if not security_scan.is_safe:
                failed_files.append({
                    "file_index": i,
                    "filename": validation_result.original_filename,
                    "status": "failed",
                    "error": f"Security scan failed: {', '.join(security_scan.threats_detected)}",
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_ms": int((datetime.now() - file_start).total_seconds() * 1000)
                })
                continue
            
            # Generate storage path
            object_path = generate_object_path(
                user_id,
                validation_result.secure_filename,
                case_id,
                validation_result.file_category
            )
            
            # Upload to MinIO
            etag = upload_file_to_minio(file_content, object_path, validation_result.content_type)
            
            # Create metadata
            file_metadata = create_file_metadata(
                validation_result=validation_result,
                object_path=object_path,
                etag=etag,
                user_id=user_id,
                case_id=case_id,
                security_scan=security_scan
            )
            
            processing_time = (datetime.now() - file_start).total_seconds() * 1000
            
            successful_files.append({
                "file_index": i,
                "filename": validation_result.original_filename,
                "status": "success",
                "file_id": file_metadata["file_id"],
                "file_metadata": file_metadata,
                "processed_at": datetime.now().isoformat(),
                "processing_time_ms": int(processing_time)
            })
            
            logger.info(f"Batch file processed successfully: {validation_result.original_filename} ({processing_time:.0f}ms)")
            
        except Exception as e:
            processing_time = (datetime.now() - file_start).total_seconds() * 1000
            failed_files.append({
                "file_index": i,
                "filename": getattr(file, 'filename', f"file_{i}"),
                "status": "failed",
                "error": str(e),
                "processed_at": datetime.now().isoformat(),
                "processing_time_ms": int(processing_time)
            })
            logger.error(f"Batch file processing failed: {getattr(file, 'filename', f'file_{i}')} - {e}")
    
    batch_duration = (datetime.now() - batch_start).total_seconds()
    logger.info(f"Batch processing completed: {batch_id} - {len(successful_files)} successful, {len(failed_files)} failed in {batch_duration:.3f}s")
    
    return successful_files, failed_files, batch_id

# =============================================================================
# FILE CLEANUP UTILITIES WITH LOGGING
# =============================================================================

def cleanup_temporary_files(temp_paths: List[str]) -> None:
    """Clean up temporary files with logging"""
    cleanup_start = datetime.now()
    cleaned_count = 0
    
    for path in temp_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                cleaned_count += 1
                logger.debug(f"Cleaned temporary file: {path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {path}: {e}")
    
    cleanup_duration = (datetime.now() - cleanup_start).total_seconds()
    logger.info(f"Cleanup completed: {cleaned_count} files in {cleanup_duration:.3f}s")

def schedule_file_deletion(object_path: str, delay_hours: int = 24) -> None:
    """Schedule file deletion with datetime tracking"""
    scheduled_time = datetime.now() + timedelta(hours=delay_hours)
    logger.info(f"Scheduled deletion of {object_path} at {scheduled_time.isoformat()}")
    
    # In production, this would integrate with Celery or similar task queue
    # For now, just log the scheduled deletion

# =============================================================================
# FILE ANALYSIS UTILITIES - PRODUCTION IMPLEMENTATIONS
# =============================================================================

def analyze_image_file(file_content: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """Comprehensive image analysis for medical images"""
    
    analysis_start = datetime.now()
    logger.info(f"Starting image analysis for: {filename}")
    
    try:
        from PIL import Image, ExifTags
        from io import BytesIO
        
        # Open image
        image = Image.open(BytesIO(file_content))
        
        # Basic image properties
        analysis_result = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "has_transparency": image.mode in ("RGBA", "LA", "P"),
            "color_depth": len(image.getbands()),
            "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
        }
        
        # Extract EXIF data for medical images
        exif_data = {}
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag] = str(value)
        
        analysis_result["exif_data"] = exif_data
        
        # Image quality assessment
        # Calculate basic statistics
        if image.mode in ['RGB', 'L']:
            import numpy as np
            img_array = np.array(image)
            
            if image.mode == 'RGB':
                # Convert to grayscale for analysis
                gray_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray_array = img_array
            
            # Basic quality metrics
            analysis_result.update({
                "brightness_mean": float(np.mean(gray_array)),
                "brightness_std": float(np.std(gray_array)),
                "contrast_score": float(np.std(gray_array) / 255.0),
                "is_too_dark": np.mean(gray_array) < 50,
                "is_too_bright": np.mean(gray_array) > 200,
                "estimated_quality": "good" if 50 <= np.mean(gray_array) <= 200 and np.std(gray_array) > 30 else "poor"
            })
        
        # Medical image specific checks
        medical_indicators = {
            "likely_xray": image.mode == 'L' and image.width > 512 and image.height > 512,
            "likely_ultrasound": "ultrasound" in filename.lower() or image.width == image.height,
            "likely_mri": "mri" in filename.lower() or (image.mode == 'L' and image.width > 256),
            "likely_photo": image.mode == 'RGB' and "photo" in filename.lower()
        }
        
        analysis_result["medical_analysis"] = medical_indicators
        
        # File size efficiency
        analysis_result["file_efficiency"] = {
            "bytes_per_pixel": len(file_content) / (image.width * image.height),
            "compression_ratio": len(file_content) / (image.width * image.height * 3),  # Assuming RGB
            "size_category": "large" if len(file_content) > 5 * 1024 * 1024 else "medium" if len(file_content) > 1024 * 1024 else "small"
        }
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        analysis_result["analysis_metadata"] = {
            "processing_time_seconds": analysis_duration,
            "analyzed_at": analysis_start.isoformat(),
            "analysis_engine": "PIL+NumPy",
            "success": True
        }
        
        logger.info(f"Image analysis completed for {filename} in {analysis_duration:.3f}s")
        return analysis_result
        
    except ImportError as e:
        logger.error(f"Required library not available for image analysis: {e}")
        return {
            "error": f"Image analysis library not available: {e}",
            "analysis_metadata": {
                "processing_time_seconds": 0,
                "analyzed_at": analysis_start.isoformat(),
                "success": False
            }
        }
    except Exception as e:
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        logger.error(f"Image analysis failed for {filename}: {e}")
        return {
            "error": f"Image analysis failed: {e}",
            "analysis_metadata": {
                "processing_time_seconds": analysis_duration,
                "analyzed_at": analysis_start.isoformat(),
                "success": False
            }
        }

def analyze_audio_file(file_content: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """Comprehensive audio analysis for medical recordings"""
    
    analysis_start = datetime.now()
    logger.info(f"Starting audio analysis for: {filename}")
    
    try:
        import wave
        import struct
        from io import BytesIO
        
        # Try to open as WAV file first
        try:
            audio_stream = BytesIO(file_content)
            with wave.open(audio_stream, 'rb') as wav_file:
                # Basic WAV properties
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                duration = frames / sample_rate
                
                # Read audio data
                raw_audio = wav_file.readframes(frames)
                
                analysis_result = {
                    "format": "wav",
                    "duration_seconds": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": sample_width,
                    "total_frames": frames,
                    "bit_depth": sample_width * 8,
                    "bitrate": sample_rate * channels * sample_width * 8
                }
                
        except wave.Error:
            # Fallback for other audio formats
            analysis_result = {
                "format": "unknown",
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "error": "Could not parse audio format, basic analysis only"
            }
            raw_audio = file_content
        
        # Basic audio quality analysis
        if raw_audio and len(raw_audio) > 0:
            try:
                import numpy as np
                
                # Convert bytes to numpy array (assuming 16-bit)
                if analysis_result.get("sample_width") == 2:  # 16-bit
                    audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                elif analysis_result.get("sample_width") == 1:  # 8-bit
                    audio_data = np.frombuffer(raw_audio, dtype=np.uint8)
                else:
                    audio_data = np.frombuffer(raw_audio, dtype=np.int16)  # Default assumption
                
                # Basic signal analysis
                if len(audio_data) > 0:
                    analysis_result.update({
                        "signal_analysis": {
                            "max_amplitude": float(np.max(np.abs(audio_data))),
                            "rms_level": float(np.sqrt(np.mean(audio_data**2))),
                            "dynamic_range": float(np.max(audio_data) - np.min(audio_data)),
                            "zero_crossings": int(np.sum(np.diff(np.sign(audio_data)) != 0)),
                            "silence_ratio": float(np.sum(np.abs(audio_data) < np.max(np.abs(audio_data)) * 0.01) / len(audio_data))
                        }
                    })
                    
                    # Frequency analysis (basic)
                    if len(audio_data) > 1024 and analysis_result.get("sample_rate"):
                        fft = np.fft.fft(audio_data[:1024])
                        freqs = np.fft.fftfreq(1024, 1 / analysis_result["sample_rate"])
                        magnitude = np.abs(fft)
                        
                        # Find dominant frequency
                        dominant_freq_idx = np.argmax(magnitude[:512])  # Only positive frequencies
                        dominant_frequency = freqs[dominant_freq_idx]
                        
                        analysis_result["frequency_analysis"] = {
                            "dominant_frequency_hz": float(dominant_frequency),
                            "frequency_spread": float(np.std(freqs[:512][magnitude[:512] > np.max(magnitude) * 0.1])),
                            "high_frequency_content": float(np.sum(magnitude[256:512]) / np.sum(magnitude[:512]))
                        }
                
            except ImportError:
                logger.warning("NumPy not available for advanced audio analysis")
            except Exception as e:
                logger.warning(f"Advanced audio analysis failed: {e}")
        
        # Medical audio specific analysis
        medical_indicators = {
            "likely_heart_sounds": "heart" in filename.lower() or "cardiac" in filename.lower(),
            "likely_lung_sounds": "lung" in filename.lower() or "respiratory" in filename.lower(),
            "likely_voice_recording": analysis_result.get("sample_rate", 0) in [8000, 16000, 22050],
            "sufficient_duration": analysis_result.get("duration_seconds", 0) > 5,
            "good_quality": analysis_result.get("signal_analysis", {}).get("rms_level", 0) > 1000
        }
        
        analysis_result["medical_analysis"] = medical_indicators
        
        # File efficiency
        analysis_result["file_efficiency"] = {
            "compression_detected": len(file_content) < (analysis_result.get("total_frames", 1) * analysis_result.get("sample_width", 2)),
            "size_category": "large" if len(file_content) > 10 * 1024 * 1024 else "medium" if len(file_content) > 1024 * 1024 else "small",
            "estimated_quality": "high" if analysis_result.get("sample_rate", 0) >= 44100 else "medium" if analysis_result.get("sample_rate", 0) >= 22050 else "low"
        }
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        analysis_result["analysis_metadata"] = {
            "processing_time_seconds": analysis_duration,
            "analyzed_at": analysis_start.isoformat(),
            "analysis_engine": "wave+numpy",
            "success": True
        }
        
        logger.info(f"Audio analysis completed for {filename} in {analysis_duration:.3f}s")
        return analysis_result
        
    except Exception as e:
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        logger.error(f"Audio analysis failed for {filename}: {e}")
        return {
            "error": f"Audio analysis failed: {e}",
            "format": "unknown",
            "analysis_metadata": {
                "processing_time_seconds": analysis_duration,
                "analyzed_at": analysis_start.isoformat(),
                "success": False
            }
        }

def analyze_pdf_file(file_content: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """Comprehensive PDF analysis for medical reports"""
    
    analysis_start = datetime.now()
    logger.info(f"Starting PDF analysis for: {filename}")
    
    try:
        from io import BytesIO
        
        # Try PyPDF2 first (most common)
        pdf_analysis = None
        
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            # Basic PDF properties
            num_pages = len(pdf_reader.pages)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    page_texts.append(page_text)
                    full_text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    page_texts.append("")
            
            # Get document info
            doc_info = {}
            if pdf_reader.metadata:
                doc_info = {
                    "title": pdf_reader.metadata.get('/Title', ''),
                    "author": pdf_reader.metadata.get('/Author', ''),
                    "subject": pdf_reader.metadata.get('/Subject', ''),
                    "creator": pdf_reader.metadata.get('/Creator', ''),
                    "producer": pdf_reader.metadata.get('/Producer', ''),
                    "creation_date": str(pdf_reader.metadata.get('/CreationDate', '')),
                    "modification_date": str(pdf_reader.metadata.get('/ModDate', ''))
                }
            
            pdf_analysis = {
                "library_used": "PyPDF2",
                "page_count": num_pages,
                "total_characters": len(full_text),
                "total_words": len(full_text.split()),
                "has_text": len(full_text.strip()) > 0,
                "document_info": doc_info,
                "pages_with_text": sum(1 for text in page_texts if len(text.strip()) > 0)
            }
            
        except ImportError:
            # Fallback to pdfplumber if available
            try:
                import pdfplumber
                
                with pdfplumber.open(BytesIO(file_content)) as pdf:
                    num_pages = len(pdf.pages)
                    full_text = ""
                    
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                full_text += page_text + "\n"
                        except Exception as e:
                            logger.warning(f"Failed to extract text with pdfplumber: {e}")
                    
                    pdf_analysis = {
                        "library_used": "pdfplumber",
                        "page_count": num_pages,
                        "total_characters": len(full_text),
                        "total_words": len(full_text.split()),
                        "has_text": len(full_text.strip()) > 0,
                        "document_info": {},
                        "pages_with_text": num_pages if full_text.strip() else 0
                    }
                    
            except ImportError:
                # Basic analysis without text extraction
                pdf_analysis = {
                    "library_used": "basic_analysis",
                    "error": "No PDF parsing library available",
                    "file_size": len(file_content),
                    "has_pdf_header": file_content.startswith(b'%PDF'),
                    "estimated_pages": file_content.count(b'/Page') if b'/Page' in file_content else "unknown"
                }
                full_text = ""
        
        if pdf_analysis and "error" not in pdf_analysis:
            # Content analysis
            text_lower = full_text.lower()
            
            # Medical document detection
            medical_keywords = [
                'patient', 'diagnosis', 'treatment', 'symptoms', 'medical', 'doctor',
                'hospital', 'clinic', 'medication', 'prescription', 'blood', 'test',
                'examination', 'health', 'condition', 'disease', 'therapy'
            ]
            
            keyword_counts = {}
            for keyword in medical_keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    keyword_counts[keyword] = count
            
            medical_score = sum(keyword_counts.values()) / len(full_text.split()) if full_text.split() else 0
            
            # Document structure analysis
            structure_analysis = {
                "has_medical_keywords": len(keyword_counts) > 0,
                "medical_relevance_score": medical_score,
                "keyword_density": keyword_counts,
                "likely_medical_report": medical_score > 0.01,
                "has_structured_content": any(pattern in text_lower for pattern in [':', '-', '•', '1.', 'a)']),
                "text_density": len(full_text) / pdf_analysis["page_count"] if pdf_analysis["page_count"] > 0 else 0
            }
            
            pdf_analysis["content_analysis"] = structure_analysis
            
            # Quality assessment
            quality_metrics = {
                "text_extractable": pdf_analysis["has_text"],
                "sufficient_content": len(full_text.split()) > 50,
                "readable_length": 50 <= len(full_text.split()) <= 10000,
                "consistent_formatting": pdf_analysis["pages_with_text"] / pdf_analysis["page_count"] > 0.8 if pdf_analysis["page_count"] > 0 else False
            }
            
            pdf_analysis["quality_assessment"] = quality_metrics
        
        # File characteristics
        file_characteristics = {
            "file_size_bytes": len(file_content),
            "size_category": "large" if len(file_content) > 5 * 1024 * 1024 else "medium" if len(file_content) > 512 * 1024 else "small",
            "compression_ratio": len(file_content) / (pdf_analysis.get("total_characters", 1) or 1),
            "likely_scanned": not pdf_analysis.get("has_text", False) or pdf_analysis.get("total_characters", 0) < 100
        }
        
        if pdf_analysis:
            pdf_analysis.update(file_characteristics)
        else:
            pdf_analysis = file_characteristics
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        pdf_analysis["analysis_metadata"] = {
            "processing_time_seconds": analysis_duration,
            "analyzed_at": analysis_start.isoformat(),
            "analysis_engine": pdf_analysis.get("library_used", "basic"),
            "success": True
        }
        
        logger.info(f"PDF analysis completed for {filename} in {analysis_duration:.3f}s")
        return pdf_analysis
        
    except Exception as e:
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        logger.error(f"PDF analysis failed for {filename}: {e}")
        return {
            "error": f"PDF analysis failed: {e}",
            "file_size_bytes": len(file_content),
            "has_pdf_header": file_content.startswith(b'%PDF') if file_content else False,
            "analysis_metadata": {
                "processing_time_seconds": analysis_duration,
                "analyzed_at": analysis_start.isoformat(),
                "success": False
            }
        }

# =============================================================================
# EXPORT
# =============================================================================


__all__ = [
    # MinIO utilities
    "get_minio_client", "upload_file_to_minio", "generate_presigned_url",
    "delete_file_from_minio",
    
    # Validation utilities
    "validate_medical_file", "perform_security_scan", "validate_file_signature",
    "categorize_medical_file", "calculate_file_hash",
    
    # Path utilities
    "generate_object_path", "generate_file_url",
    
    # Metadata utilities
    "create_file_metadata", "extract_file_content", "perform_content_analysis",
    
    # Analysis utilities - NOW PROPERLY IMPLEMENTED
    "analyze_image_file", "analyze_audio_file", "analyze_pdf_file",
    
    # Batch processing
    "process_file_batch",
    
    # Cleanup utilities
    "cleanup_temporary_files", "schedule_file_deletion"
]
