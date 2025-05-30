# In msp/components/report_display.py
# ... (imports including asyncio, json, AppState, deepseek_generator)
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()
import mesop as me
import time
import json
import asyncio  # For me.effects

# Ensures paths are set up so 'from state.state_manager import AppState' works
import utils.path_setup

# CORRECTED: Import AppState and the state update helper
from state.state_manager import AppState, update_report_state_in_app_and_db
# Assuming deepseek_generator is correctly defined in report_generator.py
from components.report_generator import deepseek_generator
# Assuming these rendering components are also in this file or correctly imported
# For this example, I'll add placeholder definitions if they aren't typically here.
# from styles.whatsapp_dark_theme import WHATSAPP_DARK_COLORS # If needed by render functions below

# Placeholder render functions if not imported from elsewhere
# In a real setup, these would likely be full components.
@me.component
def render_report_header():
    me.text("Medical Report Header", type="headline-5")

@me.component
def render_generation_progress(progress_text: str):
    me.progress_spinner(style=me.Style(margin=me.Margin(bottom=10)))
    me.text(f"Progress: {progress_text}")

@me.component
def render_complete_report(parsed_report_data: dict):
    with me.text_to_text(title="Generated Report"): # Example of displaying JSON
        me.text(json.dumps(parsed_report_data, indent=2))

@me.component
def render_error_state(error_message: str):
    me.text(f"Error: {error_message}", style=me.Style(color="red"))


async def _actual_report_generation_task(case_data: dict, app_state_snapshot: AppState) -> dict:
    """
    The core async task for generating the report data.
    This function will be called by me.effects.
    """
    # Call the deepseek_generator.generate_comprehensive_report
    # This function should return a dictionary.
    report_dict = await deepseek_generator.generate_comprehensive_report(case_data, app_state_snapshot)
    return report_dict


# This is the effect generator function that will be called
def start_report_generation_effect(): # Renamed to avoid conflict if imported directly
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
    yield # Allow UI to update to "generating" state

    try:
        # Prepare case_data from the current state for the async task.
        case_data_for_llm = {
            "case_id": state.session_id,
            "chat_history_summary": [msg.content for msg in state.chat_history[-10:] if msg.role == 'user'], # Example
            # Add other relevant fields from AppState that deepseek_generator needs
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
        yield # UI update

        # Call the async function using me.effects
        report_data_dict_result = yield from me.effects(
            lambda: _actual_report_generation_task(case_data_for_llm, state), 
            ()
        )
        
        update_report_state_in_app_and_db(
            report_data_dict=report_data_dict_result,
            generation_progress="Report generated successfully!",
            generation_complete=True,
            is_generating=False
        )
        
    except Exception as e:
        error_msg = f"Report generation process failed: {str(e)}"
        update_report_state_in_app_and_db(
            error_message=error_msg,
            generation_complete=True,
            is_generating=False
        )
    yield # Final UI update
