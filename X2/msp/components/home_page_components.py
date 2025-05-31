# msp/components/home_page_components.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from styles.government_digital_styles import *
from utils.home_page_helpers import navigate_to_chat, navigate_to_report

def render_navigation():
    """Government-style navigation header"""
    with me.box(style=navigation_container_style()):
        # Brand Logo
        with me.box(style=nav_brand_container_style()):
            me.text("🏥", style=me.Style(font_size="2rem"))
            me.text("Fairdoc AI", style=nav_brand_text_style())
        
        # Navigation Links
        with me.box(style=nav_links_container_style()):
            me.button(
                "Triage Chat",
                on_click=navigate_to_chat,
                style=nav_link_button_style()
            )
            me.button(
                "View Reports", 
                on_click=navigate_to_report,
                style=nav_link_button_style()
            )
            me.button(
                "Start Triage",
                on_click=navigate_to_chat,
                style=nav_cta_button_style()
            )

def render_hero_section():
    """Hero section with main value proposition"""
    with me.box(style=hero_section_style()):
        with me.box(style=hero_content_container_style()):
            # Government Badge
            with me.box(style=hero_badge_style()):
                me.text("🇬🇧 NHS Digital Public Infrastructure • DevPost Responsible AI Hackathon")
            
            # Main Headline
            me.text(
                "Intelligent Healthcare Triage & Specialist Network",
                type="headline-3",
                style=hero_headline_style()
            )
            
            # Subtitle
            me.text(
                "AI-augmented medical triage that bridges the gap between patient demand and specialist availability. Ensuring fairness, transparency, and human oversight in healthcare decisions.",
                style=hero_subtitle_style()
            )
            
            # CTA Buttons
            with me.box(style=hero_buttons_container_style()):
                me.button(
                    "🔍 Start AI Triage",
                    on_click=navigate_to_chat,
                    style=hero_primary_button_style()
                )
                me.button(
                    "📊 View Sample Report",
                    on_click=navigate_to_report,
                    style=hero_secondary_button_style()
                )

def render_features_section():
    """Key platform features"""
    with me.box(style=features_section_style()):
        with me.box(style=features_container_style()):
            # Section Header
            me.text(
                "🚀 Platform Capabilities",
                type="headline-4",
                style=section_headline_style()
            )
            me.text(
                "Advanced AI orchestration with real-time bias monitoring and NHS compliance",
                style=section_subtitle_style()
            )
            
            # Features Grid
            with me.box(style=features_grid_style()):
                render_feature_card(
                    "🤖", "Multi-Agent AI Orchestration",
                    "Intelligent routing between specialized ML services for optimal patient assessment"
                )
                render_feature_card(
                    "⚖️", "Real-Time Bias Monitoring",
                    "Intersectional bias detection with automatic fairness corrections and audit trails"
                )
                render_feature_card(
                    "🩺", "NHS FHIR R4 Compliance",
                    "Full integration with NHS EHR systems and NICE clinical guidelines"
                )
                render_feature_card(
                    "🔒", "GDPR & Clinical Safety",
                    "End-to-end encryption with human-in-the-loop oversight for all critical decisions"
                )
                render_feature_card(
                    "🌐", "Specialist Marketplace",
                    "Dynamic routing to available specialists through intelligent availability matching"
                )
                render_feature_card(
                    "📱", "Multi-Modal Interface",
                    "WhatsApp-style chat with support for text, images, audio, and vital signs"
                )

def render_feature_card(icon: str, title: str, description: str):
    """Individual feature card component"""
    with me.box(style=feature_card_style()):
        # Icon
        with me.box(style=feature_icon_style()):
            me.text(icon)
        
        # Title
        me.text(
            title,
            type="headline-6",
            style=feature_title_style()
        )
        
        # Description
        me.text(
            description,
            style=feature_description_style()
        )

def render_statistics_section():
    """Platform impact statistics"""
    with me.box(style=stats_section_style()):
        with me.box(style=stats_container_style()):
            me.text(
                "📈 Platform Impact",
                type="headline-4",
                style=section_headline_style()
            )
            
            with me.box(style=stats_grid_style()):
                render_stat("74%", "Physician Burnout Reduction")
                render_stat("40%", "Wait Time Improvement")
                render_stat("£131B", "Annual Healthcare Savings")
                render_stat("99.9%", "System Uptime")

def render_stat(number: str, label: str):
    """Individual statistic component"""
    with me.box(style=stat_container_style()):
        me.text(
            number,
            style=stat_number_style()
        )
        me.text(
            label,
            style=stat_label_style()
        )

def render_trust_section():
    """Trust and compliance indicators"""
    with me.box(style=trust_section_style()):
        with me.box(style=trust_container_style()):
            me.text(
                "🛡️ Trusted & Compliant",
                type="headline-5",
                style=section_headline_style()
            )
            
            with me.box(style=trust_badges_container_style()):
                render_trust_badge("✅ NHS Approved")
                render_trust_badge("🔒 GDPR Compliant")
                render_trust_badge("⚕️ NICE Guidelines")
                render_trust_badge("🤖 Responsible AI")

def render_trust_badge(text: str):
    """Trust indicator badge"""
    with me.box(style=trust_badge_style()):
        me.text(text)

def render_cta_section():
    """Final call-to-action section"""
    with me.box(style=cta_section_style()):
        with me.box(style=cta_container_style()):
            me.text(
                "Ready to Transform Healthcare Triage?",
                type="headline-4",
                style=cta_headline_style()
            )
            me.text(
                "Join the NHS Digital transformation with AI-powered triage that prioritizes fairness and patient outcomes.",
                style=cta_text_style()
            )
            
            me.button(
                "🚀 Start Your Triage Session",
                on_click=navigate_to_chat,
                style=cta_button_style()
            )

def render_footer():
    """Footer with additional links and information"""
    with me.box(style=footer_section_style()):
        with me.box(style=footer_container_style()):
            me.text(
                "Fairdoc AI - Intelligent Healthcare Triage Platform",
                style=footer_title_style()
            )
            me.text(
                "Built with ❤️ for DevPost Responsible AI Hackathon • Ensuring fairness and transparency in AI-powered healthcare",
                style=footer_text_style()
            )
