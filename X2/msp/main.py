import mesop as me
import utils.path_setup # CRITICAL: This must be the first Fairdoc/msp import

# Import pages (after path setup)
from pages import chat_page  # Import the module
from pages import report_page # Import the module
from state.state_manager import initialize_app_state, AppState
from styles.whatsapp_dark_theme import app_container_style

# Global on_load for all pages, or define per page if needed
def on_load_main(e: me.LoadEvent):
    me.set_theme_mode("dark") 
    # Initialize state with a session_id. In a real app, this might come from URL or auth.
    initialize_app_state(session_id="user_session_123") 

@me.page(
    path="/", # Default page, will render chat_page content
    title="NHS AI Triage",
    on_load=on_load_main # Apply the on_load handler
)
def main_app_page():
    # This page can act as a router or directly render the default page.
    # For simplicity, we'll call the chat_page's rendering function directly.
    # Ensure chat_page.chat_page_content is the function that builds the UI.
    chat_page.chat_page_content()


if __name__ == "__main__":
    print("Starting Mesop app. Ensure you run with `mesop main.py` or `mesop run main.py`")
    # To run directly with `python main.py`, you'd typically use Mesop's CLI execution.
    # For development, `mesop main.py` is standard.