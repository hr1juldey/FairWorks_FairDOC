import mesop as me
import utils.path_setup  # CRITICAL: This must be the first Fairdoc/msp import

# Import pages (after path setup)
import pages.chat_page
import pages.report_page
from state.state_manager import initialize_app_state, AppState
from styles.whatsapp_dark_theme import app_container_style

def on_load(e: me.LoadEvent):
    me.set_theme_mode("dark")  # Force dark mode
    initialize_app_state()    # Initialize state on first load or reload

@me.page(
    path="/",  # Default page
    title="NHS AI Triage",
    on_load=on_load
)
def main_app_container():
    state = me.state(AppState)
    with me.box(style=app_container_style()):
        if state.current_page == "chat":
            pages.chat_page.chat_page()  # Render the chat page content
        elif state.current_page == "report":
            pages.report_page.report_page()  # Render the report page content
        # Add routing for other pages here if needed
        
        # Simple navigation example (can be improved with a proper sidebar/header)
        # with me.box(style=me.Style(position="fixed", bottom=10, left=10)):
        #     me.button("Chat", on_click=lambda e: setattr(state, 'current_page', 'chat'))
        #     me.button("Report", on_click=lambda e: setattr(state, 'current_page', 'report'))

if __name__ == "__main__":
    # This will automatically discover and run pages if using Mesop CLI
    # For direct python main.py execution, you might need to specify the app.
    # However, Mesop's primary run method is `mesop main.py`
    print("Starting Mesop app. Ensure you run with `mesop main.py` or `mesop run main.py`")
    # If you must run with `python main.py`, use:
    # import mesop.cli.execute_module
    # mesop.cli.execute_module.execute_module()
