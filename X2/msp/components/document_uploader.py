# msp/components/document_uploader.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import List, Optional, Dict
from styles.government_digital_styles import GOVERNMENT_COLORS
from utils.document_helpers import (
    validate_medical_document,
    get_file_icon,
    format_file_size,
    get_supported_file_types,
    get_file_type_description,
    get_compliance_message,
    process_file_for_preview,
    validate_file_list
)

@me.stateclass
class DocumentUploaderState:
    uploaded_files: List[me.UploadedFile] = None
    upload_progress: float = 0.0
    is_uploading: bool = False
    upload_error: Optional[str] = None

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
        
        render_upload_area()
        render_upload_status()
        render_uploaded_files_section()
        render_compliance_notice()

def render_upload_area():
    """Render the file upload area"""
    with me.content_uploader(
        accepted_file_types=get_supported_file_types(),
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
            me.text(get_file_type_description(), 
                   style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["text_secondary"]))

def render_upload_status():
    """Render upload progress and error states"""
    state = me.state(DocumentUploaderState)
    
    if state.upload_error:
        render_upload_error(state.upload_error)
    
    if state.is_uploading:
        render_upload_progress(state.upload_progress)

def render_upload_error(error_message: str):
    """Render upload error message"""
    with me.box(style=me.Style(
        background="#FFEBEE",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["error"])),
        border_radius="8px",
        padding=me.Padding.all(12),
        margin=me.Margin(top=16)
    )):
        with me.box(style=me.Style(display="flex", align_items="center")):
            me.icon("error", style=me.Style(color=GOVERNMENT_COLORS["error"], margin=me.Margin(right=8)))
            me.text(f"Upload Error: {error_message}", style=me.Style(color=GOVERNMENT_COLORS["error"]))

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

def render_uploaded_files_section():
    """Render uploaded files list section"""
    state = me.state(DocumentUploaderState)
    
    if state.uploaded_files:
        with me.box(style=me.Style(margin=me.Margin(top=16))):
            me.text("Uploaded Documents:", style=me.Style(font_weight="600", margin=me.Margin(bottom=12)))
            
            for i, file in enumerate(state.uploaded_files):
                render_file_item(file, i)

def render_file_item(file: me.UploadedFile, index: int):
    """Render individual uploaded file item"""
    file_metadata = process_file_for_preview(file)
    
    with me.box(style=me.Style(
        display="flex",
        align_items="center",
        padding=me.Padding.all(12),
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["light_grey"])),
        border_radius="8px",
        margin=me.Margin(bottom=8)
    )):
        me.icon(file_metadata["icon"], style=me.Style(margin=me.Margin(right=12), color=GOVERNMENT_COLORS["primary"]))
        
        render_file_info(file_metadata)
        render_file_actions(index, file_metadata["can_preview"])

def render_file_info(file_metadata: Dict[str, any]):
    """Render file information section"""
    with me.box(style=me.Style(flex_grow=1)):
        me.text(file_metadata["name"], style=me.Style(font_weight="500"))
        me.text(f"{file_metadata['size_formatted']} • {file_metadata['mime_type']}", 
               style=me.Style(font_size="0.8rem", color=GOVERNMENT_COLORS["text_secondary"]))

def render_file_actions(index: int, can_preview: bool):
    """Render file action buttons"""
    with me.box(style=me.Style(display="flex", gap=8)):
        if can_preview:
            me.button(
                "View",
                on_click=lambda e, idx=index: view_file_action(e, idx),
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
            on_click=lambda e, idx=index: remove_file_action(e, idx),
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
        
        me.text(get_compliance_message(), style=me.Style(
            font_size="0.9rem", 
            line_height="1.4", 
            color=GOVERNMENT_COLORS["text_secondary"]
        ))

# Event Handlers
def handle_document_upload(event: me.UploadEvent):
    """Handle medical document upload with validation"""
    state = me.state(DocumentUploaderState)
    
    # Initialize uploaded_files if None
    if state.uploaded_files is None:
        state.uploaded_files = []
    
    # Clear previous errors
    state.upload_error = None
    
    # Validate file
    is_valid, error_message = validate_medical_document(event.file)
    
    if is_valid:
        # Check if we're not exceeding file limits
        if len(state.uploaded_files) >= 10:  # Max 10 files
            state.upload_error = "Maximum of 10 files allowed"
            return
        
        state.uploaded_files.append(event.file)
        state.is_uploading = True
        
        # Simulate upload progress (in real implementation, this would be actual progress)
        simulate_upload_progress()
    else:
        state.upload_error = error_message

def simulate_upload_progress():
    """Simulate upload progress for demo purposes"""
    state = me.state(DocumentUploaderState)
    
    # Simple simulation - in real implementation, this would track actual upload
    # For now, just set to complete
    state.upload_progress = 100.0
    state.is_uploading = False

def view_file_action(e: me.ClickEvent, file_index: int):
    """Handle file view action - FIXED: Now actually uses the file"""
    state = me.state(DocumentUploaderState)
    
    if state.uploaded_files and file_index < len(state.uploaded_files):
        selected_file = state.uploaded_files[file_index]
        
        # FIXED: Actually use the file variable for viewing logic
        file_metadata = process_file_for_preview(selected_file)
        
        # In a real implementation, this would:
        # - Open file in a modal/new tab
        # - Display file preview
        # - Show file details
        # For now, we'll just show file info (demo)
        
        # Set some state to show file is being viewed (optional)
        # state.currently_viewing_file = file_metadata
        
        # Log the action (in real app, would actually view the file)
        print(f"Viewing file: {selected_file.name} ({file_metadata['size_formatted']})")

def remove_file_action(e: me.ClickEvent, file_index: int):
    """Handle file removal action"""
    state = me.state(DocumentUploaderState)
    
    if state.uploaded_files and file_index < len(state.uploaded_files):
        removed_file = state.uploaded_files.pop(file_index)
        print(f"Removed file: {removed_file.name}")
        
        # Clear error if files were successfully managed
        if state.upload_error:
            state.upload_error = None
