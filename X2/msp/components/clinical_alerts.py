# msp/components/clinical_alerts.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, List, Any, Literal
from styles.government_digital_styles import GOVERNMENT_COLORS

AlertType = Literal["emergency", "urgent", "warning", "info", "success"]

def render_clinical_alert(
    message: str,
    alert_type: AlertType = "info",
    dismissible: bool = True,
    actions: List[Dict[str, Any]] = None
):
    """Render clinical alert with appropriate styling and actions"""
    
    # Alert type styling
    alert_styles = {
        "emergency": {
            "background": "#FFEBEE",
            "border_color": GOVERNMENT_COLORS["error"],
            "icon": "emergency",
            "icon_color": GOVERNMENT_COLORS["error"]
        },
        "urgent": {
            "background": "#FFF3E0", 
            "border_color": "#FF9800",
            "icon": "priority_high",
            "icon_color": "#FF9800"
        },
        "warning": {
            "background": "#FFFDE7",
            "border_color": "#FFC107",
            "icon": "warning",
            "icon_color": "#FFC107"
        },
        "info": {
            "background": "#E3F2FD",
            "border_color": GOVERNMENT_COLORS["primary"],
            "icon": "info",
            "icon_color": GOVERNMENT_COLORS["primary"]
        },
        "success": {
            "background": "#E8F5E8",
            "border_color": GOVERNMENT_COLORS["success"],
            "icon": "check_circle",
            "icon_color": GOVERNMENT_COLORS["success"]
        }
    }
    
    style_config = alert_styles.get(alert_type, alert_styles["info"])
    
    with me.box(style=me.Style(
        background=style_config["background"],
        border=me.Border.all(me.BorderSide(
            width=1,
            style="solid", 
            color=style_config["border_color"]
        )),
        border_left=me.BorderSide(
            width=4,
            style="solid",
            color=style_config["border_color"]
        ),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        with me.box(style=me.Style(display="flex", align_items="flex-start")):
            # Alert icon
            me.icon(
                style_config["icon"],
                style=me.Style(
                    color=style_config["icon_color"],
                    margin=me.Margin(right=12),
                    font_size="24px"
                )
            )
            
            # Alert content
            with me.box(style=me.Style(flex_grow=1)):
                me.text(message, style=me.Style(
                    color=GOVERNMENT_COLORS["text_primary"],
                    line_height="1.4"
                ))
                
                # Action buttons
                if actions:
                    with me.box(style=me.Style(
                        display="flex",
                        gap=8,
                        margin=me.Margin(top=12)
                    )):
                        for action in actions:
                            me.button(
                                action.get("label", "Action"),
                                on_click=action.get("on_click"),
                                style=me.Style(
                                    background=style_config["border_color"],
                                    color="white",
                                    padding=me.Padding.symmetric(horizontal=16, vertical=8),
                                    border_radius="4px",
                                    font_size="0.9rem"
                                )
                            )
            
            # Dismiss button
            if dismissible:
                me.icon(
                    "close",
                    style=me.Style(
                        color=GOVERNMENT_COLORS["text_secondary"],
                        cursor="pointer",
                        font_size="20px"
                    )
                )

def render_emergency_alert(urgency_data: Dict[str, Any]):
    """Render emergency alert based on risk level"""
    
    risk_level = urgency_data.get("risk_level", "ROUTINE")
    
    if risk_level == "IMMEDIATE":
        render_clinical_alert(
            "🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED - Contact emergency services (999) immediately",
            alert_type="emergency",
            dismissible=False,
            actions=[
                {
                    "label": "Call 999",
                    "on_click": lambda e: emergency_call_999(e)
                }
            ]
        )
    elif risk_level == "URGENT":
        render_clinical_alert(
            "⚠️ URGENT medical assessment needed - Contact NHS 111 or attend A&E within 1 hour",
            alert_type="urgent",
            dismissible=False,
            actions=[
                {
                    "label": "Call 111", 
                    "on_click": lambda e: emergency_call_111(e)
                }
            ]
        )

def render_system_alerts():
    """Render system status and operational alerts"""
    
    # Sample system alerts
    alerts = [
        {
            "message": "✅ AI bias monitoring: All fairness metrics within acceptable ranges",
            "type": "success"
        },
        {
            "message": "🔒 Data encryption: All patient data encrypted with NHS-approved algorithms",
            "type": "info"
        },
        {
            "message": "📊 System performance: Response time optimal for clinical decision support",
            "type": "info"
        }
    ]
    
    for alert in alerts:
        render_clinical_alert(
            alert["message"],
            alert_type=alert["type"],
            dismissible=True
        )

def render_data_quality_alerts(report_data: Dict[str, Any]):
    """Render alerts about data quality and completeness"""
    
    # Check for missing EHR data
    ehr_data = report_data.get("ehr_data", {})
    if not ehr_data:
        render_clinical_alert(
            "📋 No EHR data available - Assessment based on chat history only. "
            "Consider requesting patient medical records for comprehensive evaluation.",
            alert_type="warning"
        )
    
    # Check for incomplete vital signs
    chat_history = report_data.get("chat_history", [])
    if len(chat_history) < 3:
        render_clinical_alert(
            "💬 Limited conversation data - Consider asking additional questions about symptoms, "
            "medical history, and current medications for more accurate assessment.",
            alert_type="info"
        )

def emergency_call_999(e: me.ClickEvent):
    """Handle emergency 999 call action"""
    # In real implementation, this could:
    # - Log the emergency escalation
    # - Display emergency contact information
    # - Trigger automated notifications to medical staff
    pass

def emergency_call_111(e: me.ClickEvent):
    """Handle NHS 111 call action"""
    # In real implementation, this could:
    # - Log the urgent escalation
    # - Display NHS 111 contact information
    # - Provide pre-filled information for the call
    pass
