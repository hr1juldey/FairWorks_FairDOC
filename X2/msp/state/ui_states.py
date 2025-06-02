# msp/state/ui_states.py

import mesop as me
from typing import Dict

@me.stateclass
class ReportTabsState:
    """State for report tab management"""
    active_tab: str = "overview"
    tabs_expanded: Dict[str, bool] = None
    
    def __post_init__(self):
        """Initialize None dict fields"""
        if self.tabs_expanded is None:
            self.tabs_expanded = {"ehr_overview": True, "nice_protocol": True}
