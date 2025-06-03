# msp/state/ui_states.py

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ReportTabsState:
    """State for report tab management"""
    active_tab: str = "overview"
    tabs_expanded: Dict[str, bool] = field(default_factory=lambda: {"ehr_overview": True, "nice_protocol": True})
