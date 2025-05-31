# msp/components/report_tabs.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, List, Any
from styles.government_digital_styles import GOVERNMENT_COLORS

@me.stateclass
class ReportTabsState:
    active_tab: str = "overview"
    tabs_data: Dict[str, Dict] = None

def render_report_tabs(report_data: Dict[str, Any]):
    """Render the 3-tab report interface"""
    
    state = me.state(ReportTabsState)
    
    # Tab navigation
    with me.box(style=me.Style(
        display="flex",
        border_bottom=f"2px solid {GOVERNMENT_COLORS['primary']}",
        margin=me.Margin(bottom=24)
    )):
        render_tab_button("overview", "📊 Overview", state.active_tab)
        render_tab_button("analysis", "🔍 Deep Analysis", state.active_tab) 
        render_tab_button("transcript", "💬 Clinical Transcript", state.active_tab)
    
    # Tab content
    if state.active_tab == "overview":
        from components.report_overview import render_overview_tab
        render_overview_tab(report_data)
    elif state.active_tab == "analysis":
        from components.report_analysis import render_analysis_tab
        render_analysis_tab(report_data)
    elif state.active_tab == "transcript":
        from components.report_transcript import render_transcript_tab
        render_transcript_tab(report_data)

def render_tab_button(tab_id: str, label: str, active_tab: str):
    """Render individual tab button"""
    
    is_active = tab_id == active_tab
    
    me.button(
        label,
        on_click=lambda e: switch_tab(e, tab_id),
        style=me.Style(
            background=GOVERNMENT_COLORS["primary"] if is_active else "transparent",
            color="white" if is_active else GOVERNMENT_COLORS["primary"],
            border=me.Border.all(me.BorderSide(width=2, style="solid", color=GOVERNMENT_COLORS["primary"])),
            padding=me.Padding.symmetric(horizontal=20, vertical=12),
            margin=me.Margin(right=8),
            font_weight="600",
            cursor="pointer"
        )
    )

def switch_tab(e: me.ClickEvent, tab_id: str):
    """Switch active tab"""
    state = me.state(ReportTabsState)
    state.active_tab = tab_id
