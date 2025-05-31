# msp/pages/report_page.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Import state and utilities
from state.state_manager import AppState, initialize_app_state
from utils.extensions.webcomponents import get_webcomponent_security_policy

# Import components
from components.shared_navigation import render_shared_navigation
from components.report_tabs import render_report_tabs
from components.clinical_alerts import render_emergency_alert, render_system_alerts, render_data_quality_alerts
from components.patient_summary_card import render_patient_summary_card
from components.clinical_charts import render_risk_score_chart, render_vitals_chart
from components.document_uploader import render_medical_document_uploader

# Import styles
from styles.government_digital_styles import GOVERNMENT_COLORS, base_page_style

def on_report_page_load(e: me.LoadEvent):
    """Initialize report page state"""
    me.set_theme_mode("system")
    state = me.state(AppState)
    if not state.session_id:
        state.session_id = "report_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    initialize_app_state(state.session_id)

@me.page(
    path="/report",
    title="Fairdoc AI - Clinical Reports & Analysis",
    on_load=on_report_page_load,
    security_policy=get_webcomponent_security_policy()
)
def report_page():
    """
    Main reports page with comprehensive clinical analysis and documentation
    """
    with me.box(style=base_page_style()):
        render_shared_navigation()
        render_report_page_content()

def render_report_page_content():
    """Render main report page content"""
    state = me.state(AppState)
    
    with me.box(style=me.Style(
        max_width="1400px",
        margin=me.Margin.symmetric(horizontal="auto"),
        padding=me.Padding.all(24)
    )):
        # Page Header
        render_report_page_header()
        
        # Emergency Alerts (if any)
        render_emergency_alerts(state)
        
        # Main Content Area
        with me.box(style=me.Style(display="flex", flex_direction="column", gap=32)):
            
            # Report Status Section
            render_report_status_section(state)
            
            # Patient Overview Section
            render_patient_overview_section(state)
            
            # Report Content Section
            render_main_report_section(state)
            
            # Document Management Section
            render_document_management_section()
            
            # System Status Section
            render_system_status_section()

def render_report_page_header():
    """Render report page header"""
    with me.box(style=me.Style(margin=me.Margin(bottom=32))):
        me.text(
            "📊 Clinical Reports & Analysis",
            type="headline-3",
            style=me.Style(
                color=GOVERNMENT_COLORS["text_primary"],
                margin=me.Margin(bottom=8)
            )
        )
        me.text(
            "Comprehensive NHS-compliant clinical assessment reports with AI-powered analysis",
            style=me.Style(
                color=GOVERNMENT_COLORS["text_secondary"],
                font_size="1.1rem",
                line_height="1.5"
            )
        )

def render_emergency_alerts(state: AppState):
    """Render emergency alerts if present"""
    if state.emergency_escalation and state.clinical_analysis:
        urgency_data = {
            "risk_level": state.clinical_analysis.risk_level,
            "recommended_action": state.clinical_analysis.recommended_action
        }
        render_emergency_alert(urgency_data)
    
    # Show active alerts
    if state.active_alerts:
        for alert in state.active_alerts:
            if not alert.get("dismissed", False):
                render_alert_banner(alert)

def render_alert_banner(alert: Dict[str, Any]):
    """Render individual alert banner"""
    alert_color = GOVERNMENT_COLORS["error"] if alert["type"] == "emergency" else GOVERNMENT_COLORS["warning"]
    
    with me.box(style=me.Style(
        background=f"{alert_color}15",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=alert_color)),
        border_left=me.BorderSide(width=4, style="solid", color=alert_color),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text(alert["title"], style=me.Style(font_weight="600", color=alert_color))
        me.text(alert["message"], style=me.Style(color=GOVERNMENT_COLORS["text_primary"]))

def render_report_status_section(state: AppState):
    """Render report generation status"""
    with me.box(style=me.Style(
        background="white",
        border_radius="12px",
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)",
        margin=me.Margin(bottom=24)
    )):
        me.text("🔄 Report Status", type="headline-5", style=me.Style(margin=me.Margin(bottom=16)))
        
        if state.report_is_generating:
            render_report_generating_status(state)
        elif state.report_generation_complete:
            render_report_complete_status(state)
        elif state.report_error_message:
            render_report_error_status(state)
        else:
            render_report_not_started_status(state)

def render_report_generating_status(state: AppState):
    """Render report generation in progress"""
    with me.box(style=me.Style(display="flex", align_items="center", gap=16)):
        me.progress_spinner(style=me.Style(color=GOVERNMENT_COLORS["primary"]))
        with me.box():
            me.text("Generating Clinical Report...", style=me.Style(font_weight="600"))
            if state.report_generation_progress:
                me.text(state.report_generation_progress, style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_report_complete_status(state: AppState):
    """Render completed report status"""
    with me.box(style=me.Style(display="flex", align_items="center", gap=16)):
        me.text("✅", style=me.Style(font_size="24px"))
        with me.box():
            me.text("Report Generated Successfully", style=me.Style(font_weight="600", color=GOVERNMENT_COLORS["success"]))
            me.text(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}", 
                   style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_report_error_status(state: AppState):
    """Render report error status"""
    with me.box(style=me.Style(display="flex", align_items="center", gap=16)):
        me.text("⚠️", style=me.Style(font_size="24px"))
        with me.box():
            me.text("Report Generation Error", style=me.Style(font_weight="600", color=GOVERNMENT_COLORS["error"]))
            me.text(state.report_error_message or "Unknown error occurred", 
                   style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_report_not_started_status(state: AppState):
    """Render no report available status"""
    with me.box():
        me.text("📋 No Active Report", style=me.Style(font_weight="600"))
        me.text("Complete a triage session to generate a clinical report", 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))
        
        me.button(
            "🚀 Start New Triage Session",
            on_click=lambda e: me.navigate("/chat"),
            style=me.Style(
                background=GOVERNMENT_COLORS["primary"],
                color="white",
                padding=me.Padding.symmetric(horizontal=20, vertical=10),
                border_radius="6px",
                font_weight="600",
                border="none",
                cursor="pointer",
                margin=me.Margin(top=12)
            )
        )

def render_patient_overview_section(state: AppState):
    """Render patient overview section"""
    patient_data = None
    if state.patient_data:
        patient_data = {
            "nhs_number": state.patient_data.nhs_number,
            "name": state.patient_data.name,
            "age": state.patient_data.age,
            "gender": state.patient_data.gender,
            "birth_date": state.patient_data.birth_date,
            "address": state.patient_data.address,
            "phone": state.patient_data.phone,
            "allergies": state.patient_data.allergies or [],
            "current_medications": state.patient_data.current_medications or []
        }
    
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=16)):
        render_patient_summary_card(patient_data)
        
        # Clinical charts if analysis available
        if state.clinical_analysis:
            urgency_data = {
                "urgency_score": state.clinical_analysis.urgency_score,
                "risk_level": state.clinical_analysis.risk_level
            }
            render_risk_score_chart(urgency_data)
        
        # Vital signs placeholder
        render_vitals_chart(patient_data)

def render_main_report_section(state: AppState):
    """Render main report content with tabs"""
    if state.report_generation_complete and state.report_data_json:
        try:
            report_data = json.loads(state.report_data_json)
            
            # Add clinical analysis to report data
            if state.clinical_analysis:
                report_data["urgency_analysis"] = {
                    "urgency_score": state.clinical_analysis.urgency_score,
                    "risk_level": state.clinical_analysis.risk_level,
                    "risk_color": state.clinical_analysis.risk_color,
                    "recommended_action": state.clinical_analysis.recommended_action,
                    "flagged_phrases": state.clinical_analysis.flagged_phrases or [],
                    "risk_factors": state.clinical_analysis.risk_factors or []
                }
            
            # Add chat history to report data
            if state.chat_history:
                report_data["chat_history"] = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp
                    }
                    for msg in state.chat_history
                ]
            
            # Add clinical summary
            if state.clinical_analysis and state.clinical_analysis.nice_protocol:
                report_data["clinical_summary"] = {
                    "nice_protocol": state.clinical_analysis.nice_protocol,
                    "primary_symptoms": [],  # Would be populated from analysis
                    "timeline": []  # Would be populated from chat analysis
                }
            
            # Add EHR data if available
            if state.ehr_data_json:
                try:
                    report_data["ehr_data"] = json.loads(state.ehr_data_json)
                except json.JSONDecodeError:
                    pass
            
            render_report_tabs(report_data)
            
        except json.JSONDecodeError:
            render_report_data_error()
    else:
        render_no_report_available()

def render_report_data_error():
    """Render report data parsing error"""
    with me.box(style=me.Style(
        background="#FFEBEE",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["error"])),
        border_radius="8px",
        padding=me.Padding.all(20),
        text_align="center"
    )):
        me.text("⚠️ Report Data Error", style=me.Style(font_weight="600", color=GOVERNMENT_COLORS["error"]))
        me.text("Unable to parse report data. Please regenerate the report.", 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_no_report_available():
    """Render no report available message"""
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        border_radius="12px",
        padding=me.Padding.all(40),
        text_align="center"
    )):
        me.text("📋", style=me.Style(font_size="48px", margin=me.Margin(bottom=16)))
        me.text("No Clinical Report Available", type="headline-5", 
               style=me.Style(margin=me.Margin(bottom=8)))
        me.text("Complete a triage session to generate a comprehensive clinical report with AI analysis.", 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"], margin=me.Margin(bottom=24)))
        
        me.button(
            "🔍 Start Triage Assessment",
            on_click=lambda e: me.navigate("/chat"),
            style=me.Style(
                background=GOVERNMENT_COLORS["primary"],
                color="white",
                padding=me.Padding.symmetric(horizontal=24, vertical=12),
                border_radius="8px",
                font_weight="600",
                border="none",
                cursor="pointer"
            )
        )

def render_document_management_section():
    """Render document upload and management section"""
    with me.box(style=me.Style(margin=me.Margin(top=32))):
        me.text("📎 Medical Document Management", type="headline-5", 
               style=me.Style(margin=me.Margin(bottom=16)))
        render_medical_document_uploader()

def render_system_status_section():
    """Render system status and alerts"""
    with me.box(style=me.Style(margin=me.Margin(top=32))):
        me.text("🔧 System Status", type="headline-5", 
               style=me.Style(margin=me.Margin(bottom=16)))
        render_system_alerts()
        
        # Data quality alerts
        state = me.state(AppState)
        report_data = {}
        if state.ehr_data_json:
            try:
                report_data["ehr_data"] = json.loads(state.ehr_data_json)
            except Exception:
                pass
        if state.chat_history:
            report_data["chat_history"] = state.chat_history
        
        render_data_quality_alerts(report_data)
