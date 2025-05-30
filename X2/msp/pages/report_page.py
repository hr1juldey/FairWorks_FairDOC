import mesop as me
import json
import asyncio  # Required for me.effects to call async functions
import time    # For synchronous sleep during progress simulation
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()  # SET UP PATH

# Import AppState and the helper for updating state + DB
from state.state_manager import AppState, update_report_state_in_app_and_db
from styles.whatsapp_dark_theme import WHATSAPP_DARK_COLORS

# Assuming these components are correctly defined and imported if they are separate files
# For this fix, they are assumed to be correctly imported or defined elsewhere.
# If components.report_display doesn't exist or functions are elsewhere, adjust.
from components.report_display import (
    render_report_header,
    render_generation_progress,
    render_complete_report,
    render_error_state
)
# This import is crucial for the async call
from components.report_generator import deepseek_generator


@me.page(path="/report", title="Medical Report")
def report_page_content():
    state = me.state(AppState)

    # Ensure state fields related to report are initialized
    if state.report_data_json is None: state.report_data_json = ""
    if state.report_error_message is None: state.report_error_message = ""
    if state.report_generation_progress is None: state.report_generation_progress = ""
    # current_page should be managed by main_app_page or navigation logic

    with me.box(style=me.Style(padding=me.Padding.all(20), background=WHATSAPP_DARK_COLORS["app_bg"], height="100vh", color=WHATSAPP_DARK_COLORS["primary_text"], overflow_y="auto")):
        render_report_header()

        parsed_report_data = None
        if state.report_data_json and state.report_generation_complete and not state.report_error_message:
            try:
                parsed_report_data = json.loads(state.report_data_json)
            except json.JSONDecodeError:
                # This update should also persist if it's a critical error.
                # For simplicity, we'll let the UI reflect the error from state.
                if not state.report_error_message: # Avoid overwriting specific LLM errors
                    state.report_error_message = "Error: Could not display the report (invalid local format)."


        # Logic to determine what to display
        if not state.report_generation_complete and not state.report_is_generating:
            # If landed here and no report generation has started or completed.
            # This might happen if navigated directly to /report.
            # Typically, generation is triggered from chat_page.
            # Add a button for manual trigger or show a message.
            if not state.report_data_json and not state.report_error_message :
                 me.button("Generate Full Medical Report", on_click=trigger_manual_report_generation, style=me.Style(margin=me.Margin(top=20)))
                 me.text("Or complete a chat session to generate a report automatically.", style=me.Style(font_size="12px", color=WHATSAPP_DARK_COLORS["secondary_text"], margin=me.Margin(top=5)))
            elif state.report_error_message: # If there was a previous error
                 render_error_state(state.report_error_message)

        elif state.report_is_generating:
            render_generation_progress(state.report_generation_progress)
        elif parsed_report_data:
            render_complete_report(parsed_report_data)
        elif state.report_error_message:
            render_error_state(state.report_error_message)
        else: # Default, e.g. complete=true, but no data and no error (should be rare)
            me.text("Report data is not available at this moment.", style=me.Style(color=WHATSAPP_DARK_COLORS["secondary_text"]))

        with me.button("Back to Chat", on_click=lambda e: me.navigate("/chat"), style=me.Style(margin=me.Margin(top=20))):
            pass

def trigger_manual_report_generation(e: me.ClickEvent):
    # This is an effect generator, so it's fine to yield from another effect generator
    yield from start_report_generation_effect()

# This is the new async helper function that will be called by me.effects
async def _async_generate_report_from_deepseek(case_data: dict, app_state_snapshot: AppState) -> dict:
    """
    Asynchronous helper to call the deepseek_generator.
    Receives necessary data to avoid relying on live Mesop state within async context.
    """
    # The actual async call to your LLM
    report_dict = await deepseek_generator.generate_comprehensive_report(case_data, app_state_snapshot)
    return report_dict

def start_report_generation_effect():
    """Effect to start asynchronous report generation and update AppState."""
    state = me.state(AppState) # Get current state
    
    # Update AppState and DB to reflect "generating" status
    update_report_state_in_app_and_db(
        is_generating=True,
        generation_complete=False,
        error_message=None, # Clear previous errors
        report_data_json=None, # Clear previous report data
        generation_progress="Initializing NHS assessment..."
    )
    yield # UI update: Show "Initializing"

    try:
        # Prepare case_data from the current state for the async task.
        # It's often better to pass a snapshot of data rather than the whole state object
        # if the async function doesn't need to interact with Mesop's live state.
        case_data_for_llm = {
            "case_id": state.session_id,
            "chat_history_summary": [msg.content for msg in state.chat_history[-10:]], # Example: last 10 messages
            # Add any other relevant fields from AppState that deepseek_generator.generate_comprehensive_report expects
        }
        
        # Simulate progress updates using synchronous time.sleep
        progress_messages = [
            "🔍 Analyzing patient symptoms...",
            "📋 Applying NICE clinical guidelines...",
            "🧠 DeepSeek-R1:14b preparing assessment context..."
        ]
        
        for progress_msg in progress_messages:
            update_report_state_in_app_and_db(generation_progress=progress_msg)
            yield # UI update: Show current progress_msg
            time.sleep(0.5) # Synchronous sleep for UI effect

        update_report_state_in_app_and_db(generation_progress="🤖 DeepSeek AI is now generating the full report...")
        yield # UI update: Show "DeepSeek AI is now generating..."

        # CORRECTED: Call the async function using me.effects
        # The result (report_data_dict) will be passed to the next yield of this generator
        report_data_dict = yield from me.effects(
            # Lambda calls our new async helper
            lambda: _async_generate_report_from_deepseek(case_data_for_llm, state), 
            () # No arguments needed for the lambda itself
        )
        
        # Update AppState and DB with the generated report (as JSON string)
        update_report_state_in_app_and_db(
            report_data_dict=report_data_dict, # This should be a dictionary
            generation_progress="Report generated successfully!",
            generation_complete=True,
            is_generating=False # Set generating to false
        )
        
    except Exception as e:
        error_msg = f"Report generation process failed: {str(e)}"
        update_report_state_in_app_and_db(
            error_message=error_msg,
            generation_complete=True, # Still "complete" but with an error
            is_generating=False
        )
    yield # Final UI update with result or error
