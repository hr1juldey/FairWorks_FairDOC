# msp/components/patient_summary_card.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, Any, Optional
from styles.government_digital_styles import GOVERNMENT_COLORS
from utils.report_helpers import format_timestamp

def render_patient_summary_card(patient_data: Optional[Dict[str, Any]] = None):
    """Render comprehensive patient summary card"""
    
    if not patient_data:
        render_no_patient_data_card()
        return

    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)",
        margin=me.Margin(bottom=16)
    )):
        # Header with patient name and NHS number
        with me.box(style=me.Style(
            display="flex",
            justify_content="space-between",
            align_items="center",
            margin=me.Margin(bottom=20)
        )):
            with me.box():
                me.text("👤 Patient Summary", type="headline-5", style=me.Style(margin=me.Margin(bottom=4)))
                if patient_data.get("name"):
                    me.text(patient_data["name"], type="headline-6", style=me.Style(color=GOVERNMENT_COLORS["primary"]))
            
            # NHS number badge
            if patient_data.get("nhs_number"):
                with me.box(style=me.Style(
                    background=GOVERNMENT_COLORS["bg_accent"],
                    padding=me.Padding.symmetric(horizontal=12, vertical=6),
                    border_radius="20px"
                )):
                    me.text(f"NHS: {patient_data['nhs_number']}", style=me.Style(
                        font_weight="600",
                        font_size="0.9rem",
                        color=GOVERNMENT_COLORS["primary"]
                    ))
        
        # Patient demographics grid
        with me.box(style=me.Style(
            display="grid",
            grid_template_columns="repeat(auto-fit, minmax(200px, 1fr))",
            gap=20,
            margin=me.Margin(bottom=20)
        )):
            render_demographic_item("Age", f"{patient_data.get('age', 'Unknown')} years", "calendar_today")
            render_demographic_item("Gender", patient_data.get("gender", "Not specified").title(), "person")
            render_demographic_item("Date of Birth", patient_data.get("birth_date", "Not available"), "cake")
            render_demographic_item("Contact", patient_data.get("phone", "Not available"), "phone")
        
        # Address information
        if patient_data.get("address"):
            with me.box(style=me.Style(
                background=GOVERNMENT_COLORS["bg_secondary"],
                padding=me.Padding.all(16),
                border_radius="8px",
                margin=me.Margin(bottom=16)
            )):
                with me.box(style=me.Style(display="flex", align_items="center", margin=me.Margin(bottom=8))):
                    me.icon("location_on", style=me.Style(margin=me.Margin(right=8), color=GOVERNMENT_COLORS["primary"]))
                    me.text("Address", style=me.Style(font_weight="600"))
                
                me.text(patient_data["address"], style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))
        
        # Medical alerts/flags
        render_medical_alerts(patient_data)

def render_demographic_item(label: str, value: str, icon: str):
    """Render individual demographic information item"""
    
    with me.box(style=me.Style(
        display="flex",
        align_items="center",
        padding=me.Padding.all(12),
        background=GOVERNMENT_COLORS["bg_secondary"],
        border_radius="8px"
    )):
        me.icon(icon, style=me.Style(
            margin=me.Margin(right=12),
            color=GOVERNMENT_COLORS["primary"],
            font_size="20px"
        ))
        
        with me.box():
            me.text(label, style=me.Style(
                font_size="0.9rem",
                color=GOVERNMENT_COLORS["text_secondary"],
                margin=me.Margin(bottom=2)
            ))
            me.text(value, style=me.Style(
                font_weight="500",
                color=GOVERNMENT_COLORS["text_primary"]
            ))

def render_medical_alerts(patient_data: Dict[str, Any]):
    """Render medical alerts and warnings"""
    
    # Check for high-risk indicators
    alerts = []
    
    # FIXED: Proper None handling for age comparison
    age = patient_data.get("age")
    if age is not None:  # Only check age if it's not None
        try:
            age_int = int(age)  # Convert to int in case it's a string
            if age_int > 65:
                alerts.append("🔸 Elderly patient - Enhanced monitoring recommended")
            elif age_int < 18:
                alerts.append("🔸 Pediatric patient - Age-appropriate protocols required")
        except (ValueError, TypeError):
            # Handle case where age can't be converted to int
            pass
    
    # Check for pregnancy (if applicable)
    if patient_data.get("pregnancy_status"):
        alerts.append("🤰 Pregnancy considerations - Specialized care protocols")
    
    # Allergies
    allergies = patient_data.get("allergies", [])
    if allergies and len(allergies) > 0:
        alerts.append(f"⚠️ Known allergies: {', '.join(allergies)}")
    
    # Current medications
    medications = patient_data.get("current_medications", [])
    if medications and len(medications) > 0:
        alerts.append(f"💊 Current medications: {len(medications)} active prescriptions")
    
    if alerts:
        with me.box(style=me.Style(margin=me.Margin(top=16))):
            me.text("⚕️ Clinical Alerts", style=me.Style(
                font_weight="600",
                margin=me.Margin(bottom=12),
                color=GOVERNMENT_COLORS["text_primary"]
            ))
            
            for alert in alerts:
                with me.box(style=me.Style(
                    background="#FFF3E0",
                    border=me.Border.all(me.BorderSide(width=1, style="solid", color="#FF9800")),
                    border_radius="6px",
                    padding=me.Padding.all(8),
                    margin=me.Margin(bottom=8)
                )):
                    me.text(alert, style=me.Style(
                        font_size="0.9rem",
                        color=GOVERNMENT_COLORS["text_primary"]
                    ))

def render_no_patient_data_card():
    """Render placeholder when no patient data is available"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)",
        margin=me.Margin(bottom=16),
        text_align="center"
    )):
        me.icon("person_off", style=me.Style(
            font_size="48px",
            color=GOVERNMENT_COLORS["medium_grey"],
            margin=me.Margin(bottom=16)
        ))
        
        me.text("No Patient Data Available", type="headline-6", style=me.Style(
            color=GOVERNMENT_COLORS["text_primary"],
            margin=me.Margin(bottom=8)
        ))
        
        me.text(
            "Patient demographics and medical history not available. "
            "Assessment based on chat interaction only.",
            style=me.Style(
                color=GOVERNMENT_COLORS["text_secondary"],
                line_height="1.4"
            )
        )
        
        with me.box(style=me.Style(
            background=GOVERNMENT_COLORS["bg_accent"],
            padding=me.Padding.all(16),
            border_radius="8px",
            margin=me.Margin(top=16)
        )):
            me.text("🔐 To access full patient records, NHS Digital authentication and patient consent are required.", 
                   style=me.Style(
                       font_size="0.9rem",
                       color=GOVERNMENT_COLORS["text_secondary"]
                   ))

def render_session_metadata(session_data: Dict[str, Any]):
    """Render session metadata and timing information"""
    
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        padding=me.Padding.all(16),
        border_radius="8px",
        margin=me.Margin(top=16)
    )):
        me.text("📊 Session Information", style=me.Style(
            font_weight="600",
            margin=me.Margin(bottom=12)
        ))
        
        session_items = [
            ("Session ID", session_data.get("session_id", "Unknown")),
            ("Start Time", format_timestamp(session_data.get("start_time", ""))),
            ("Platform", "Fairdoc AI Triage System"),
            ("Compliance", "NHS Digital Standards")
        ]
        
        with me.box(style=me.Style(
            display="grid",
            grid_template_columns="1fr 1fr",
            gap=12
        )):
            for label, value in session_items:
                with me.box():
                    me.text(label, style=me.Style(
                        font_size="0.8rem",
                        color=GOVERNMENT_COLORS["text_secondary"]
                    ))
                    me.text(value, style=me.Style(
                        font_weight="500",
                        color=GOVERNMENT_COLORS["text_primary"]
                    ))
