# msp/state/document_states.py

from dataclasses import dataclass

@dataclass
class UploadedFileState:
    """State for uploaded medical documents"""
    file_id: str = ""
    original_filename: str = ""
    file_size: int = 0
    mime_type: str = ""
    upload_timestamp: str = ""
    file_category: str = "medical_document"
