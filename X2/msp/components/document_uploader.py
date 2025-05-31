# msp/components/document_uploader.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
import base64
from typing import List, Dict, Any, Optional
from styles.government_digital_styles import GOVERNMENT_COLORS

@me.stateclass
class DocumentUploaderState:
    uploaded_files: List[me.UploadedFile] = None
    upload_progress: float = 0.0
    is_uploading: bool = False

def render_medical_document_uploader():
    """Render medical document uploader with NHS compliance"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(20),
        margin=me.Margin(bottom=16)
    )):
        me.text("📎 Upload Medical Documents", type="headline-6", style=me.Style(margin=me.Margin(bottom=16)))
        
        # Upload area
        with me.content_uploader(
            accepted_file_types=[
                "application/pdf",
                "image/jpeg", 
                "image/png",
                "application/dicom",  # Medical imaging
                "text/plain"
            ],
            on_upload=handle_document_upload,
            type="flat",
            color="primary",
            style=me.Style(
                width="100%",
                padding=me.Padding.all(20),
                border=me.Border.all(me.BorderSide(width=2, style="dashed", color=GOVERNMENT_COLORS["primary"])),
                border_radius="8px",
                text_align="center"
            )
        ):
            with me.box(style=me.Style(display="flex", flex_direction="column", align_items="center", gap=8)):
                me.icon("cloud_upload", style=me.Style(font_size="48px", color=GOVERNMENT_COLORS["primary"]))
                me.text("Upload Medical Documents", style=me.Style(font_weight="600"))
                me.text("Supported: PDF, Images, DICOM, Text files", 
                       style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["text_secondary"]))
        
        # Upload progress
        state = me.state(DocumentUploaderState)
        if state.is_uploading:
            render_upload_progress(state.upload_progress)
        
        # Uploaded files list
        if state.uploaded_files:
            render_uploaded_files_list(state.uploaded_files)
        
        # Compliance notice
        render_compliance_notice()

def handle_document_upload(event: me.UploadEvent):
    """Handle medical document upload with validation"""
    
    state = me.state(DocumentUploaderState)
    
    if state.uploaded_files is None:
        state.uploaded_files = []
    
    # Validate file
    if validate_medical_document(event.file):
        state.uploaded_files.append(event.file)
        state.is_uploading = True
        # Simulate upload progress
        simulate_upload_progress()
    else:
        # Show error for invalid file
        pass

def validate_medical_document(file: me.UploadedFile) -> bool:
    """Validate uploaded medical document"""
    
    # File size limit (10MB)
    max_size = 10 * 1024 * 1024
    if file.size > max_size:
        return False
    
    # Allowed MIME types
    allowed_types = [
        "application/pdf",
        "image/jpeg",
        "image/png", 
        "application/dicom",
        "text/plain"
    ]
    
    return file.mime_type in allowed_types

def simulate_upload_progress():
    """Simulate upload progress for demo"""
    # In real implementation, this would track actual upload progress
    pass

def render_upload_progress(progress: float):
    """Render upload progress bar"""
    
    with me.box(style=me.Style(margin=me.Margin(top=16, bottom=16))):
        me.text("Uploading...", style=me.Style(font_weight="500", margin=me.Margin(bottom=8)))
        
        with me.box(style=me.Style(
            background=GOVERNMENT_COLORS["light_grey"],
            height="8px",
            border_radius="4px",
            overflow="hidden"
        )):
            with me.box(style=me.Style(
                background=GOVERNMENT_COLORS["primary"],
                height="8px",
                width=f"{progress}%",
                transition="width 0.3s ease"
            )):
                pass
        
        me.text(f"{progress:.1f}%", style=me.Style(
            font_size="0.9rem",
            color=GOVERNMENT_COLORS["text_secondary"],
            margin=me.Margin(top=4)
        ))

def render_uploaded_files_list(files: List[me.UploadedFile]):
    """Render list of uploaded files"""
    
    with me.box(style=me.Style(margin=me.Margin(top=16))):
        me.text("Uploaded Documents:", style=me.Style(font_weight="600", margin=me.Margin(bottom=12)))
        
        for i, file in enumerate(files):
            render_file_item(file, i)

def render_file_item(file: me.UploadedFile, index: int):
    """Render individual uploaded file item"""
    
    # File type icon mapping
    file_icons = {
        "application/pdf": "picture_as_pdf",
        "image/jpeg": "image",
        "image/png": "image",
        "application/dicom": "medical_services",
        "text/plain": "description"
    }
    
    icon = file_icons.get(file.mime_type, "insert_drive_file")
    
    with me.box(style=me.Style(
        display="flex",
        align_items="center",
        padding=me.Padding.all(12),
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["light_grey"])),
        border_radius="8px",
        margin=me.Margin(bottom=8)
    )):
        me.icon(icon, style=me.Style(margin=me.Margin(right=12), color=GOVERNMENT_COLORS["primary"]))
        
        with me.box(style=me.Style(flex_grow=1)):
            me.text(file.name, style=me.Style(font_weight="500"))
            me.text(f"{format_file_size(file.size)} • {file.mime_type}", 
                   style=me.Style(font_size="0.8rem", color=GOVERNMENT_COLORS["text_secondary"]))
        
        # Action buttons
        with me.box(style=me.Style(display="flex", gap=8)):
            me.button(
                "View",
                on_click=lambda e, idx=index: view_file(e, idx),
                style=me.Style(
                    background="transparent",
                    color=GOVERNMENT_COLORS["primary"],
                    border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["primary"])),
                    padding=me.Padding.symmetric(horizontal=12, vertical=6),
                    font_size="0.9rem"
                )
            )
            me.button(
                "Remove",
                on_click=lambda e, idx=index: remove_file(e, idx),
                style=me.Style(
                    background="transparent",
                    color=GOVERNMENT_COLORS["error"],
                    border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["error"])),
                    padding=me.Padding.symmetric(horizontal=12, vertical=6),
                    font_size="0.9rem"
                )
            )

def render_compliance_notice():
    """Render NHS data compliance notice"""
    
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_accent"],
        padding=me.Padding.all(16),
        border_radius="8px",
        margin=me.Margin(top=16)
    )):
        with me.box(style=me.Style(display="flex", align_items="center", margin=me.Margin(bottom=8))):
            me.icon("security", style=me.Style(margin=me.Margin(right=8), color=GOVERNMENT_COLORS["primary"]))
            me.text("NHS Data Protection & Security", style=me.Style(font_weight="600"))
        
        me.text(
            "All uploaded documents are encrypted and comply with NHS Digital standards. "
            "Data is processed in accordance with GDPR and NHS Information Governance requirements. "
            "Files are automatically deleted after consultation completion.",
            style=me.Style(font_size="0.9rem", line_height="1.4", color=GOVERNMENT_COLORS["text_secondary"])
        )

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

def view_file(e: me.ClickEvent, file_index: int):
    """View uploaded file"""
    state = me.state(DocumentUploaderState)
    if state.uploaded_files and file_index < len(state.uploaded_files):
        file = state.uploaded_files[file_index]
        # Implement file viewing logic
        pass

def remove_file(e: me.ClickEvent, file_index: int):
    """Remove uploaded file"""
    state = me.state(DocumentUploaderState)
    if state.uploaded_files and file_index < len(state.uploaded_files):
        state.uploaded_files.pop(file_index)
