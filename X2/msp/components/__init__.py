# msp/components/__init__.py

"""
Mesop Components for Fairdoc AI Platform

This package contains all UI components used throughout the application.
"""

# Core chat components - explicit imports
from .chat_bubble import render_chat_bubble
from .chat_header import render_chat_header  
from .chat_input import render_chat_input

# Home page components - explicit imports
from .home_page_components import (
    render_hero_section,
    render_features_section,
    render_statistics_section,
    render_trust_section,
    render_cta_section,
    render_footer,
    render_feature_card,
    render_stat,
    render_trust_badge
)

from .shared_navigation import (
    render_shared_navigation,
    render_brand_section,
    render_nav_links,
    render_nav_cta
)

# Report components - explicit imports
from .report_tabs import (
    render_report_tabs,
    render_tab_button,
    switch_tab,
    ReportTabsState
)

from .report_overview import (
    render_overview_tab,
    render_risk_assessment_card,
    render_clinical_summary_card,
    render_quick_actions_card,
    download_report
)

from .report_analysis import (
    render_analysis_tab,
    render_ehr_section,
    render_nice_protocol_section,
    render_bias_monitoring_section,
    render_clinical_decision_support,
    render_patient_summary_card as render_ehr_patient_summary,
    render_conditions_card,
    render_documents_media_card,
    render_nice_protocol_details,
    render_protocol_recommendations,
    render_bias_metrics,
    render_fairness_indicators,
    render_ai_confidence_section,
    render_differential_diagnosis
)

# FIXED: Only import functions that actually exist in report_transcript.py
from .report_transcript import (
    render_transcript_tab,
    render_transcript_overview,
    render_key_quotes_section,
    render_full_transcript,
    render_clinical_inference_summary,
    render_quoted_statement,
    render_transcript_message,
    render_category_summary,
    render_stat_card,
    render_no_quotes_message,
    render_quote_header,
    render_quote_content,
    render_clinical_inference,
    render_quote_footer,
    render_message_header,
    render_message_content,
    render_no_clinical_concerns
    # REMOVED: render_highlighted_content (moved to utils)
)

from .report_display import (
    render_report_header,
    render_generation_progress,
    render_complete_report,
    render_error_state,
    start_report_generation_effect
)

from .report_generator import (
    DeepSeekReportGenerator,
    deepseek_generator
)

# Clinical components - explicit imports
from .clinical_alerts import (
    render_clinical_alert,
    render_emergency_alert,
    render_system_alerts,
    render_data_quality_alerts,
    emergency_call_999,
    emergency_call_111,
    AlertType
)

from .clinical_charts import (
    render_risk_score_chart,
    render_symptoms_timeline,
    render_vitals_chart,
    render_vital_card
)

from .patient_summary_card import (
    render_patient_summary_card,
    render_no_patient_data_card,
    render_session_metadata,
    render_demographic_item,
    render_medical_alerts
)

from .document_uploader import (
    render_medical_document_uploader,
    handle_document_upload,
    simulate_upload_progress,
    view_file_action,
    remove_file_action,
    render_upload_progress,
    render_uploaded_files_section,
    render_file_item,
    render_compliance_notice,
    DocumentUploaderState
)

# Explicitly define what should be available when importing from this package
__all__ = [
    # Navigation
    "render_shared_navigation",
    "render_brand_section", 
    "render_nav_links",
    "render_nav_cta",
    
    # Home page
    "render_hero_section",
    "render_features_section", 
    "render_statistics_section",
    "render_trust_section",
    "render_cta_section",
    "render_footer",
    "render_feature_card",
    "render_stat",
    "render_trust_badge",
    
    # Reports - Tabs
    "render_report_tabs",
    "render_tab_button",
    "switch_tab",
    "ReportTabsState",
    
    # Reports - Overview
    "render_overview_tab",
    "render_risk_assessment_card",
    "render_clinical_summary_card", 
    "render_quick_actions_card",
    "download_report",
    
    # Reports - Analysis
    "render_analysis_tab",
    "render_ehr_section",
    "render_nice_protocol_section",
    "render_bias_monitoring_section",
    "render_clinical_decision_support",
    "render_ehr_patient_summary",
    "render_conditions_card",
    "render_documents_media_card",
    "render_nice_protocol_details",
    "render_protocol_recommendations",
    "render_bias_metrics",
    "render_fairness_indicators", 
    "render_ai_confidence_section",
    "render_differential_diagnosis",
    
    # Reports - Transcript (UPDATED: removed non-existent functions)
    "render_transcript_tab",
    "render_transcript_overview",
    "render_key_quotes_section",
    "render_full_transcript",
    "render_clinical_inference_summary",
    "render_quoted_statement",
    "render_transcript_message",
    "render_category_summary",
    "render_stat_card",
    "render_no_quotes_message",
    "render_quote_header",
    "render_quote_content",
    "render_clinical_inference",
    "render_quote_footer",
    "render_message_header",
    "render_message_content",
    "render_no_clinical_concerns",
    
    # Reports - Display & Generation
    "render_report_header",
    "render_generation_progress",
    "render_complete_report", 
    "render_error_state",
    "start_report_generation_effect",
    "DeepSeekReportGenerator",
    "deepseek_generator",
    
    # Clinical Alerts
    "render_clinical_alert",
    "render_emergency_alert",
    "render_system_alerts",
    "render_data_quality_alerts",
    "emergency_call_999",
    "emergency_call_111",
    "AlertType",
    
    # Clinical Charts
    "render_risk_score_chart",
    "render_symptoms_timeline",
    "render_vitals_chart",
    "render_vital_card",
    
    # Patient Summary
    "render_patient_summary_card",
    "render_no_patient_data_card",
    "render_session_metadata",
    "render_demographic_item",
    "render_medical_alerts",
    
    # Document Uploader (UPDATED: with actual function names)
    "render_medical_document_uploader",
    "handle_document_upload",
    "simulate_upload_progress",
    "view_file_action",
    "remove_file_action",
    "render_upload_progress",
    "render_uploaded_files_section",
    "render_file_item", 
    "render_compliance_notice",
    "DocumentUploaderState",
    
    # Chat Components
    "render_chat_bubble",
    "render_chat_header",
    "render_chat_input"
]
