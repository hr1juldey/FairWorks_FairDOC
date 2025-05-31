# msp/main.py
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()
import mesop as me
# Import page modules to ensure their @me.page decorators run and register the paths
import pages.chat_page
import pages.report_page
from state.state_manager import initialize_app_state, AppState  # AppState might not be needed here
from styles.whatsapp_dark_theme import app_container_style  # Might not be needed here if pages handle it

# Global on_load, could set a default session_id or theme
def on_load_global(e: me.LoadEvent):
    me.set_theme_mode("dark")
    state = me.state(AppState)  # Get state to access/set session_id
    # Initialize state with a session_id. In a real app, this might come from URL or auth.
    # If session_id is already set (e.g. by a specific page's on_load), this might just confirm it.
    if not state.session_id:  # Only set if not already set by a more specific page load
        state.session_id = "global_default_session"
    initialize_app_state(state.session_id)

@me.page(
    path="/",  # Default page
    title="NHS AI Triage - Home",  # Title for the root page
    on_load=on_load_global
)
def home_page_router():
    # The root path can now simply navigate to the default chat page
    # Or display a welcome/home screen that then links to chat or report
    # For simplicity, let's navigate to /chat by default.
    # Mesop will then load the /chat page which has its own full-page layout.
    me.navigate("/chat")
    
    # Alternatively, render a simple home page here:
    with me.box(style=app_container_style()):
        me.text("Welcome to NHS AI Triage", type="headline-4")
        with me.button("Go to Chat", on_click=lambda e: me.navigate("/chat")):
            pass
        with me.button("View Last Report", on_click=lambda e: me.navigate("/report")):
            pass


if __name__ == "__main__":
    print("Starting Mesop app. Ensure you run with `mesop main.py` or `mesop run main.py`")
