# msp/utils/document_helpers.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, List, Tuple, Optional

def validate_medical_document(file: me.UploadedFile) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded medical document
    Returns: (is_valid, error_message)
    """
    # File size limit (10MB)
    max_size = 10 * 1024 * 1024
    if file.size > max_size:
        return False, f"File size ({format_file_size(file.size)}) exceeds 10MB limit"
    
    # Allowed MIME types
    allowed_types = [
        "application/pdf",
        "image/jpeg",
        "image/png", 
        "application/dicom",
        "text/plain"
    ]
    
    if file.mime_type not in allowed_types:
        return False, f"File type '{file.mime_type}' is not supported"
    
    # Additional validation for file name
    if not file.name or len(file.name.strip()) == 0:
        return False, "Invalid file name"
    
    return True, None

def get_file_type_icon_mapping() -> Dict[str, str]:
    """Get mapping of MIME types to Material Design icons"""
    return {
        "application/pdf": "picture_as_pdf",
        "image/jpeg": "image",
        "image/png": "image",
        "application/dicom": "medical_services",
        "text/plain": "description"
    }

def get_file_icon(mime_type: str) -> str:
    """Get appropriate icon for file type"""
    icon_mapping = get_file_type_icon_mapping()
    return icon_mapping.get(mime_type, "insert_drive_file")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def get_supported_file_types() -> List[str]:
    """Get list of supported MIME types for file upload"""
    return [
        "application/pdf",
        "image/jpeg", 
        "image/png",
        "application/dicom",  # Medical imaging
        "text/plain"
    ]

def get_file_type_description() -> str:
    """Get human-readable description of supported file types"""
    return "Supported: PDF, Images, DICOM, Text files"

def calculate_upload_progress_steps(file_size: int) -> int:
    """Calculate number of progress steps based on file size"""
    # Larger files get more granular progress updates
    if file_size > 5 * 1024 * 1024:  # > 5MB
        return 20
    elif file_size > 1 * 1024 * 1024:  # > 1MB
        return 10
    else:
        return 5

def simulate_upload_progress_calculation(step: int, total_steps: int) -> float:
    """Calculate upload progress percentage for simulation"""
    return min((step / total_steps) * 100, 100.0)

def get_file_category(mime_type: str) -> str:
    """Categorize file type for processing"""
    if mime_type.startswith("image/"):
        return "image"
    elif mime_type == "application/pdf":
        return "document"
    elif mime_type == "application/dicom":
        return "medical_imaging"
    elif mime_type == "text/plain":
        return "text"
    else:
        return "other"

def validate_file_list(files: List[me.UploadedFile]) -> Tuple[bool, List[str]]:
    """Validate a list of uploaded files"""
    errors = []
    
    if not files:
        return True, []
    
    # Check total size limit (50MB for all files combined)
    total_size = sum(file.size for file in files)
    max_total_size = 50 * 1024 * 1024
    
    if total_size > max_total_size:
        errors.append(f"Total file size ({format_file_size(total_size)}) exceeds 50MB limit")
    
    # Check individual files
    for i, file in enumerate(files):
        is_valid, error = validate_medical_document(file)
        if not is_valid:
            errors.append(f"File {i + 1} ({file.name}): {error}")
    
    return len(errors) == 0, errors

def create_file_metadata(file: me.UploadedFile) -> Dict[str, any]:
    """Create metadata dictionary for uploaded file"""
    return {
        "name": file.name,
        "size": file.size,
        "size_formatted": format_file_size(file.size),
        "mime_type": file.mime_type,
        "category": get_file_category(file.mime_type),
        "icon": get_file_icon(file.mime_type),
        "upload_timestamp": None  # Would be set when actually uploaded
    }

def process_file_for_preview(file: me.UploadedFile) -> Dict[str, any]:
    """Process file for preview/display purposes"""
    metadata = create_file_metadata(file)
    
    # Add preview-specific information
    can_preview = file.mime_type in ["application/pdf", "image/jpeg", "image/png", "text/plain"]
    metadata["can_preview"] = can_preview
    
    # Add security information
    metadata["is_safe"] = validate_medical_document(file)[0]
    
    return metadata

def get_compliance_message() -> str:
    """Get NHS compliance message for document uploads"""
    return (
        "All uploaded documents are encrypted and comply with NHS Digital standards. "
        "Data is processed in accordance with GDPR and NHS Information Governance requirements. "
        "Files are automatically deleted after consultation completion."
    )

def get_max_file_size_mb() -> int:
    """Get maximum file size in MB"""
    return 10

def get_max_total_files() -> int:
    """Get maximum number of files that can be uploaded"""
    return 10
