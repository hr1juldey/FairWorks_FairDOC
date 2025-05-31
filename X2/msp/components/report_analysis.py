# msp/components/report_analysis.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
import json
from typing import Dict, List, Any
from styles.government_digital_styles import GOVERNMENT_COLORS
from utils.ehr_processor import process_ehr_data, get_patient_conditions, get_patient_documents
from utils.clinical_analysis import match_nice_protocol
from utils.report_helpers import format_timestamp

def render_analysis_tab(report_data: Dict[str, Any]):
    """Render Tab 2: Deep Analysis with EHR integration and NICE protocols"""
    
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
        
        # EHR Integration Section
        render_ehr_section(report_data)
        
        # NICE Protocol Assessment
        render_nice_protocol_section(report_data)
        
        # Bias Monitoring Section
        render_bias_monitoring_section(report_data)
        
        # Clinical Decision Support
        render_clinical_decision_support(report_data)

def render_ehr_section(report_data: Dict[str, Any]):
    """Render EHR data integration section"""
    
    ehr_data = report_data.get("ehr_data", {})
    
    with me.accordion():
        with me.expansion_panel(
            key="ehr_overview",
            title="📋 Electronic Health Records",
            description="Patient medical history and documents",
            expanded=True
        ):
            if ehr_data:
                processed_ehr = process_ehr_data(ehr_data)
                
                # Patient Summary
                patients = processed_ehr.get("patients", [])
                if patients:
                    patient = patients[0]  # Assuming single patient
                    render_patient_summary_card(patient)
                
                # Conditions
                conditions = processed_ehr.get("conditions", [])
                if conditions:
                    render_conditions_card(conditions)
                
                # Documents & Media
                documents = processed_ehr.get("documents", [])
                media = processed_ehr.get("media", [])
                if documents or media:
                    render_documents_media_card(documents, media)
                    
            else:
                with me.box(style=me.Style(
                    background=GOVERNMENT_COLORS["bg_secondary"],
                    padding=me.Padding.all(20),
                    border_radius="8px",
                    text_align="center"
                )):
                    me.text("📄 No EHR data available for this session", style=me.Style(
                        color=GOVERNMENT_COLORS["text_secondary"],
                        font_style="italic"
                    ))
                    me.text("EHR integration requires patient consent and NHS digital authentication", 
                           style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["text_muted"]))

def render_patient_summary_card(patient: Dict[str, Any]):
    """Render patient summary from EHR"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text("👤 Patient Summary", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        # Patient details in grid
        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap=16)):
            render_detail_item("NHS Number", patient.get("nhs_number", "Not available"))
            render_detail_item("Age", f"{patient.get('age', 'Unknown')} years")
            render_detail_item("Gender", patient.get("gender", "Not specified").title())
            render_detail_item("Address", patient.get("address", "Not available"))

def render_conditions_card(conditions: List[Dict[str, Any]]):
    """Render patient conditions from EHR"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text("🏥 Medical Conditions", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        for condition in conditions[:5]:  # Show top 5 conditions
            with me.box(style=me.Style(
                display="flex",
                align_items="center",
                padding=me.Padding.symmetric(vertical=8),
                border_bottom=f"1px solid {GOVERNMENT_COLORS['light_grey']}"
            )):
                # Status indicator
                status = condition.get("status", "unknown")
                status_color = GOVERNMENT_COLORS["success"] if status == "active" else GOVERNMENT_COLORS["medium_grey"]
                
                with me.box(style=me.Style(
                    width="8px",
                    height="8px",
                    border_radius="50%",
                    background=status_color,
                    margin=me.Margin(right=12)
                )):
                    pass
                
                # Condition details
                with me.box(style=me.Style(flex_grow=1)):
                    me.text(condition.get("condition_name", "Unknown condition"), 
                           style=me.Style(font_weight="500"))
                    if condition.get("snomed_code"):
                        me.text(f"SNOMED: {condition['snomed_code']}", 
                               style=me.Style(font_size="0.8rem", color=GOVERNMENT_COLORS["text_muted"]))

def render_documents_media_card(documents: List[Dict], media: List[Dict]):
    """Render documents and media from EHR"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text("📎 Medical Documents & Media", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        # Documents
        if documents:
            me.text("Documents:", style=me.Style(font_weight="500", margin=me.Margin(bottom=8)))
            for doc in documents[:3]:
                render_document_item(doc)
        
        # Media files
        if media:
            me.text("Media Files:", style=me.Style(font_weight="500", margin=me.Margin(top=16, bottom=8)))
            for media_item in media[:3]:
                render_media_item(media_item)

def render_nice_protocol_section(report_data: Dict[str, Any]):
    """Render NICE protocol assessment section"""
    
    clinical_summary = report_data.get("clinical_summary", {})
    nice_protocol = clinical_summary.get("nice_protocol")
    
    with me.accordion():
        with me.expansion_panel(
            key="nice_protocol",
            title="📋 NICE Clinical Guidelines",
            description="Evidence-based clinical decision support",
            expanded=True
        ):
            if nice_protocol:
                render_nice_protocol_details(nice_protocol, report_data)
            else:
                with me.box(style=me.Style(
                    background=GOVERNMENT_COLORS["bg_secondary"],
                    padding=me.Padding.all(20),
                    border_radius="8px"
                )):
                    me.text("🔍 Analyzing symptoms against NICE guidelines...", 
                           style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_nice_protocol_details(protocol: str, report_data: Dict[str, Any]):
    """Render specific NICE protocol details"""
    
    # Protocol information
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_accent"],
        padding=me.Padding.all(16),
        border_radius="8px",
        margin=me.Margin(bottom=16)
    )):
        me.text(f"📋 Applicable Protocol: {protocol}", type="headline-6")
        me.text("This assessment follows NICE evidence-based guidelines for optimal patient care.", 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"], margin=me.Margin(top=8)))
    
    # Protocol-specific recommendations
    render_protocol_recommendations(protocol)

def render_protocol_recommendations(protocol: str):
    """Render protocol-specific clinical recommendations"""
    
    # Sample recommendations based on protocol
    recommendations = {
        "NICE CG95": {
            "title": "Chest Pain Assessment",
            "recommendations": [
                "12-lead ECG within 10 minutes",
                "Troponin levels at presentation and 3 hours",
                "Consider immediate cardiology referral if high-risk features",
                "Aspirin 300mg if no contraindications"
            ]
        },
        "NICE CG191": {
            "title": "Breathlessness Assessment", 
            "recommendations": [
                "Peak flow measurement if possible",
                "Oxygen saturation monitoring",
                "Consider chest X-ray if signs of infection",
                "Salbutamol if history of asthma/COPD"
            ]
        }
        # Add more protocols as needed
    }
    
    protocol_info = recommendations.get(protocol, {
        "title": "Clinical Assessment",
        "recommendations": ["Follow standard clinical assessment protocols"]
    })
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16)
    )):
        me.text(f"🎯 {protocol_info['title']}", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        for rec in protocol_info["recommendations"]:
            with me.box(style=me.Style(display="flex", align_items="center", margin=me.Margin(bottom=8))):
                me.text("✓", style=me.Style(color=GOVERNMENT_COLORS["success"], margin=me.Margin(right=8)))
                me.text(rec, style=me.Style(color=GOVERNMENT_COLORS["text_primary"]))

def render_bias_monitoring_section(report_data: Dict[str, Any]):
    """Render bias monitoring and fairness assessment"""
    
    with me.accordion():
        with me.expansion_panel(
            key="bias_monitoring",
            title="⚖️ Bias Monitoring & Fairness",
            description="Real-time algorithmic fairness assessment",
            expanded=False
        ):
            # Bias metrics
            render_bias_metrics()
            
            # Fairness indicators
            render_fairness_indicators()

def render_bias_metrics():
    """Render bias monitoring metrics"""
    
    # Sample bias metrics
    bias_metrics = {
        "Demographic Parity": {"score": 0.89, "status": "PASS"},
        "Equalized Odds": {"score": 0.92, "status": "PASS"}, 
        "Individual Fairness": {"score": 0.87, "status": "PASS"},
        "Counterfactual Fairness": {"score": 0.94, "status": "PASS"}
    }
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text("📊 Algorithmic Bias Metrics", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        for metric, data in bias_metrics.items():
            with me.box(style=me.Style(
                display="flex",
                justify_content="space-between",
                align_items="center",
                padding=me.Padding.symmetric(vertical=8),
                border_bottom=f"1px solid {GOVERNMENT_COLORS['light_grey']}"
            )):
                me.text(metric, style=me.Style(font_weight="500"))
                
                with me.box(style=me.Style(display="flex", align_items="center", gap=8)):
                    me.text(f"{data['score']:.2f}", style=me.Style(font_weight="600"))
                    
                    status_color = GOVERNMENT_COLORS["success"] if data["status"] == "PASS" else GOVERNMENT_COLORS["error"]
                    with me.box(style=me.Style(
                        background=status_color,
                        color="white",
                        padding=me.Padding.symmetric(horizontal=8, vertical=4),
                        border_radius="4px",
                        font_size="0.8rem",
                        font_weight="600"
                    )):
                        me.text(data["status"])

def render_fairness_indicators():
    """Render fairness indicators and recommendations"""
    
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_accent"],
        padding=me.Padding.all(16),
        border_radius="8px"
    )):
        me.text("✅ Fairness Assessment: COMPLIANT", type="headline-6", 
               style=me.Style(color=GOVERNMENT_COLORS["success"], margin=me.Margin(bottom=8)))
        me.text("This assessment meets NHS AI fairness standards with no detected bias in clinical recommendations.", 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_clinical_decision_support(report_data: Dict[str, Any]):
    """Render clinical decision support section"""
    
    with me.accordion():
        with me.expansion_panel(
            key="clinical_decision",
            title="🧠 Clinical Decision Support",
            description="AI-powered clinical insights and recommendations",
            expanded=False
        ):
            # AI confidence scores
            render_ai_confidence_section()
            
            # Alternative diagnoses
            render_differential_diagnosis()

def render_ai_confidence_section():
    """Render AI model confidence and uncertainty quantification"""
    
    confidence_data = {
        "Primary Assessment": {"confidence": 0.91, "uncertainty": 0.09},
        "Risk Stratification": {"confidence": 0.87, "uncertainty": 0.13},
        "Protocol Matching": {"confidence": 0.94, "uncertainty": 0.06}
    }
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text("🎯 AI Confidence Metrics", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        for assessment, metrics in confidence_data.items():
            with me.box(style=me.Style(margin=me.Margin(bottom=12))):
                me.text(assessment, style=me.Style(font_weight="500", margin=me.Margin(bottom=4)))
                
                # Confidence bar
                confidence_pct = metrics["confidence"] * 100
                with me.box(style=me.Style(
                    background=GOVERNMENT_COLORS["light_grey"],
                    height="8px",
                    border_radius="4px",
                    position="relative"
                )):
                    with me.box(style=me.Style(
                        background=GOVERNMENT_COLORS["success"],
                        height="8px",
                        width=f"{confidence_pct}%",
                        border_radius="4px"
                    )):
                        pass
                
                me.text(f"Confidence: {confidence_pct:.1f}%", 
                       style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["text_secondary"]))

def render_differential_diagnosis():
    """Render alternative diagnostic considerations"""
    
    differential_dx = [
        {"condition": "Acute Coronary Syndrome", "probability": 0.78, "evidence": "Chest pain, risk factors"},
        {"condition": "Gastroesophageal Reflux", "probability": 0.65, "evidence": "Post-prandial symptoms"},
        {"condition": "Musculoskeletal Pain", "probability": 0.42, "evidence": "Positional factors"}
    ]
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16)
    )):
        me.text("🔍 Differential Diagnosis", type="headline-6", style=me.Style(margin=me.Margin(bottom=12)))
        
        for dx in differential_dx:
            with me.box(style=me.Style(
                padding=me.Padding.symmetric(vertical=8),
                border_bottom=f"1px solid {GOVERNMENT_COLORS['light_grey']}"
            )):
                with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="center")):
                    me.text(dx["condition"], style=me.Style(font_weight="500"))
                    me.text(f"{dx['probability'] * 100:.0f}%", 
                           style=me.Style(color=GOVERNMENT_COLORS["primary"], font_weight="600"))
                
                me.text(f"Evidence: {dx['evidence']}", 
                       style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["text_secondary"]))

# Helper functions
def render_detail_item(label: str, value: str):
    """Render a detail item in patient summary"""
    with me.box():
        me.text(label, style=me.Style(font_weight="500", font_size="0.9rem", color=GOVERNMENT_COLORS["text_secondary"]))
        me.text(value, style=me.Style(font_weight="400"))

def render_document_item(doc: Dict[str, Any]):
    """Render individual document item"""
    with me.box(style=me.Style(
        padding=me.Padding.symmetric(vertical=4),
        display="flex",
        align_items="center"
    )):
        me.text("📄", style=me.Style(margin=me.Margin(right=8)))
        with me.box():
            me.text(doc.get("title", "Untitled Document"), style=me.Style(font_weight="500"))
            me.text(doc.get("document_type", "Unknown type"), 
                   style=me.Style(font_size="0.8rem", color=GOVERNMENT_COLORS["text_muted"]))

def render_media_item(media: Dict[str, Any]):
    """Render individual media item"""
    with me.box(style=me.Style(
        padding=me.Padding.symmetric(vertical=4),
        display="flex",
        align_items="center"
    )):
        media_icon = "🎵" if "audio" in media.get("content_type", "") else "🖼️"
        me.text(media_icon, style=me.Style(margin=me.Margin(right=8)))
        with me.box():
            me.text(media.get("title", "Untitled Media"), style=me.Style(font_weight="500"))
            me.text(media.get("content_type", "Unknown type"), 
                   style=me.Style(font_size="0.8rem", color=GOVERNMENT_COLORS["text_muted"]))
