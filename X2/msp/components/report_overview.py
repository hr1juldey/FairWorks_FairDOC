# msp/components/report_overview.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, Any
from styles.government_digital_styles import GOVERNMENT_COLORS
from utils.report_helpers import format_risk_level, get_risk_badge_style, format_timestamp

def render_overview_tab(report_data: Dict[str, Any]):
    """Render Tab 1: Patient Overview with urgency scores"""
    
    urgency_data = report_data.get("urgency_analysis", {})
    clinical_summary = report_data.get("clinical_summary", {})
    
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
        
        # Risk Assessment Card
        render_risk_assessment_card(urgency_data)
        
        # Clinical Summary Card  
        render_clinical_summary_card(clinical_summary)
        
        # Quick Actions Card
        render_quick_actions_card(urgency_data)

def render_risk_assessment_card(urgency_data: Dict):
    """Render risk assessment summary"""
    
    risk_level = urgency_data.get("risk_level", "UNKNOWN")
    risk_color = urgency_data.get("risk_color", "#666")
    urgency_score = urgency_data.get("urgency_score", 0)
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)"
    )):
        me.text("🚨 Risk Assessment", type="headline-5", style=me.Style(margin=me.Margin(bottom=16)))
        
        # Risk level badge
        with me.box(style=me.Style(display="flex", align_items="center", gap=16, margin=me.Margin(bottom=16))):
            with me.box(style=get_risk_badge_style(risk_color)):
                me.text(format_risk_level(risk_level))
            
            me.text(f"Score: {urgency_score:.2f}/1.00", style=me.Style(font_weight="600"))
        
        # Recommended action
        action = urgency_data.get("recommended_action", "Assessment required")
        me.text(f"Recommended Action: {action}", style=me.Style(
            font_size="1.1rem",
            color=GOVERNMENT_COLORS["text_primary"],
            font_weight="500"
        ))
        
        # Risk factors
        risk_factors = urgency_data.get("risk_factors", [])
        if risk_factors:
            me.text("Risk Factors:", style=me.Style(font_weight="600", margin=me.Margin(top=16, bottom=8)))
            for factor in risk_factors[:3]:  # Show top 3
                with me.box(style=me.Style(display="flex", align_items="center", margin=me.Margin(bottom=4))):
                    me.text("⚠️", style=me.Style(margin=me.Margin(right=8)))
                    me.text(factor, style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_clinical_summary_card(clinical_summary: Dict):
    """Render clinical summary"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px", 
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)"
    )):
        me.text("📋 Clinical Summary", type="headline-5", style=me.Style(margin=me.Margin(bottom=16)))
        
        # Primary symptoms
        symptoms = clinical_summary.get("primary_symptoms", [])
        if symptoms:
            me.text("Primary Symptoms:", style=me.Style(font_weight="600", margin=me.Margin(bottom=8)))
            for symptom in symptoms:
                with me.box(style=me.Style(margin=me.Margin(left=16, bottom=4))):
                    me.text(f"• {symptom}", style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))
        
        # Timeline
        timeline = clinical_summary.get("timeline", [])
        if timeline:
            me.text("Timeline:", style=me.Style(font_weight="600", margin=me.Margin(top=16, bottom=8)))
            for event in timeline:
                with me.box(style=me.Style(margin=me.Margin(left=16, bottom=4))):
                    me.text(f"• {event}", style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))
        
        # NICE Protocol
        nice_protocol = clinical_summary.get("nice_protocol")
        if nice_protocol:
            with me.box(style=me.Style(
                background=GOVERNMENT_COLORS["bg_accent"],
                padding=me.Padding.all(12),
                border_radius="8px",
                margin=me.Margin(top=16)
            )):
                me.text(f"NICE Protocol: {nice_protocol}", style=me.Style(font_weight="600"))

def render_quick_actions_card(urgency_data: Dict):
    """Render quick actions based on urgency"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)"
    )):
        me.text("⚡ Quick Actions", type="headline-5", style=me.Style(margin=me.Margin(bottom=16)))
        
        # Action buttons based on risk level
        risk_level = urgency_data.get("risk_level", "ROUTINE")
        
        if risk_level in ["IMMEDIATE", "URGENT"]:
            me.button(
                "🚨 Contact Emergency Services",
                style=me.Style(
                    background=GOVERNMENT_COLORS["error"],
                    color="white",
                    padding=me.Padding.symmetric(horizontal=20, vertical=12),
                    border_radius="8px",
                    font_weight="600",
                    margin=me.Margin(bottom=8)
                )
            )
        
        me.button(
            "📄 Download Full Report",
            on_click=download_report,
            style=me.Style(
                background=GOVERNMENT_COLORS["primary"],
                color="white", 
                padding=me.Padding.symmetric(horizontal=20, vertical=12),
                border_radius="8px",
                font_weight="600",
                margin=me.Margin(bottom=8)
            )
        )
        
        me.button(
            "🔄 Return to Chat",
            on_click=lambda e: me.navigate("/chat"),
            style=me.Style(
                background="transparent",
                color=GOVERNMENT_COLORS["primary"],
                border=me.Border.all(me.BorderSide(width=2, style="solid", color=GOVERNMENT_COLORS["primary"])),
                padding=me.Padding.symmetric(horizontal=20, vertical=12),
                border_radius="8px",
                font_weight="600"
            )
        )

def download_report(e: me.ClickEvent):
    """Trigger report download"""
    # Implementation for downloading report
    # This would generate and download the PDF/JSON report
    pass
