# msp/main.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from utils.extensions.webcomponents import get_webcomponent_security_policy

# Import page modules to ensure their @me.page decorators run and register the paths
import pages.chat_page
import pages.report_page
from pages.home_page import render_home_page
from state.state_manager import initialize_app_state, AppState

def on_load_global(e: me.LoadEvent):
    me.set_theme_mode("dark")
    state = me.state(AppState)
    if not state.session_id:
        state.session_id = "global_default_session"
    initialize_app_state(state.session_id)

@me.page(
    path="/",
    title="Fairdoc AI - Intelligent Healthcare Triage Platform",
    on_load=on_load_global,
    security_policy=get_webcomponent_security_policy(),
    stylesheets=["/static/government_digital.css"]
)
def home_page_router():
    """
    Main home page showcasing Fairdoc AI platform capabilities
    """
    render_home_page()

if __name__ == "__main__":
    print("Starting Mesop app. Ensure you run with `mesop main.py`")
