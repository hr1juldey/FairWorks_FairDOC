# msp/components/clinical_charts.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, List, Any
from styles.government_digital_styles import GOVERNMENT_COLORS

def render_risk_score_chart(urgency_data: Dict[str, Any]):
    """Render visual risk score chart"""
    
    urgency_score = urgency_data.get("urgency_score", 0)
    risk_level = urgency_data.get("risk_level", "UNKNOWN")
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(20),
        margin=me.Margin(bottom=16)
    )):
        me.text("🎯 Risk Score Visualization", type="headline-6", style=me.Style(margin=me.Margin(bottom=16)))
        
        # Risk score bar
        score_percentage = urgency_score * 100
        
        with me.box(style=me.Style(margin=me.Margin(bottom=12))):
            me.text(f"Current Risk Score: {score_percentage:.1f}%", style=me.Style(font_weight="600"))
        
        # Visual progress bar
        with me.box(style=me.Style(
            background=GOVERNMENT_COLORS["light_grey"],
            height="20px",
            border_radius="10px",
            position="relative",
            overflow="hidden"
        )):
            # Risk level color mapping
            risk_colors = {
                "IMMEDIATE": GOVERNMENT_COLORS["error"],
                "URGENT": "#FF9800", 
                "STANDARD": "#FFC107",
                "ROUTINE": GOVERNMENT_COLORS["success"],
                "SELF_CARE": "#8BC34A"
            }
            
            bar_color = risk_colors.get(risk_level, GOVERNMENT_COLORS["medium_grey"])
            
            with me.box(style=me.Style(
                background=bar_color,
                height="20px",
                width=f"{score_percentage}%",
                border_radius="10px",
                transition="width 0.5s ease"
            )):
                pass
        
        # Risk level indicators
        with me.box(style=me.Style(
            display="flex",
            justify_content="space-between",
            margin=me.Margin(top=8),
            font_size="0.8rem"
        )):
            me.text("Low", style=me.Style(color=GOVERNMENT_COLORS["success"]))
            me.text("Medium", style=me.Style(color="#FFC107"))
            me.text("High", style=me.Style(color="#FF9800"))
            me.text("Critical", style=me.Style(color=GOVERNMENT_COLORS["error"]))

def render_symptoms_timeline(clinical_summary: Dict[str, Any]):
    """Render timeline of symptoms progression"""
    
    timeline = clinical_summary.get("timeline", [])
    
    if not timeline:
        return
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(20),
        margin=me.Margin(bottom=16)
    )):
        me.text("📅 Symptoms Timeline", type="headline-6", style=me.Style(margin=me.Margin(bottom=16)))
        
        for i, event in enumerate(timeline):
            with me.box(style=me.Style(
                display="flex",
                align_items="center",
                margin=me.Margin(bottom=12),
                position="relative"
            )):
                # Timeline dot
                with me.box(style=me.Style(
                    width="12px",
                    height="12px",
                    background=GOVERNMENT_COLORS["primary"],
                    border_radius="50%",
                    margin=me.Margin(right=16),
                    flex_shrink=0
                )):
                    pass
                
                # Timeline line (except for last item)
                if i < len(timeline) - 1:
                    with me.box(style=me.Style(
                        position="absolute",
                        left="6px",
                        top="12px",
                        width="1px",
                        height="24px",
                        background=GOVERNMENT_COLORS["medium_grey"]
                    )):
                        pass
                
                # Event text
                me.text(event, style=me.Style(color=GOVERNMENT_COLORS["text_primary"]))

def render_vitals_chart(patient_data: Dict[str, Any] = None):
    """Render vital signs chart placeholder"""
    
    # Sample vital signs data - in real implementation would come from EHR
    vitals = {
        "Blood Pressure": "120/80 mmHg",
        "Heart Rate": "72 bpm", 
        "Temperature": "37.0°C",
        "Oxygen Saturation": "98%",
        "Respiratory Rate": "16/min"
    }
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(20),
        margin=me.Margin(bottom=16)
    )):
        me.text("💗 Vital Signs", type="headline-6", style=me.Style(margin=me.Margin(bottom=16)))
        
        with me.box(style=me.Style(
            display="grid",
            grid_template_columns="repeat(auto-fit, minmax(200px, 1fr))",
            gap=16
        )):
            for vital, value in vitals.items():
                render_vital_card(vital, value)

def render_vital_card(label: str, value: str):
    """Render individual vital sign card"""
    
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        padding=me.Padding.all(16),
        border_radius="8px",
        text_align="center"
    )):
        me.text(value, style=me.Style(
            font_size="1.5rem",
            font_weight="700",
            color=GOVERNMENT_COLORS["primary"],
            margin=me.Margin(bottom=4)
        ))
        me.text(label, style=me.Style(
            color=GOVERNMENT_COLORS["text_secondary"],
            font_size="0.9rem"
        ))
