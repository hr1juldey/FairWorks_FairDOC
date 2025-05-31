import mesop as me
import json
import asyncio
import time
from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()  # SET UP PATH

from state.state_manager import AppState, update_report_state_in_app_and_db, initialize_app_state
# Import app_container_style as well
from styles.whatsapp_dark_theme import WHATSAPP_DARK_COLORS, app_container_style
from components.report_display import (
    render_report_header,
    render_generation_progress,
    render_complete_report,
    render_error_state
)
from components.report_generator import deepseek_generator

# This on_load is specific to this page if navigated to directly
def on_report_page_load(e: me.LoadEvent):
    state = me.state(AppState)
    me.set_theme_mode("dark") # Ensure theme is set
    # Initialize AppState, crucial if user lands here directly.
    # Use existing session_id if available (e.g., from query param or persisted state)
    # or create/use a default for direct access.
    initialize_app_state(state.session_id if state.session_id else "direct_report_session")
    
    # Optionally, trigger report generation if no data and not already generating
    if not state.report_data_json and not state.report_is_generating and not state.report_error_message:
        # This needs to be an effect. For direct load, it's tricky.
        # Better to show a button or message if report data is missing.
        # For now, we'll let the main render logic handle showing the "Generate" button.
        pass


@me.page(
    path="/report", 
    title="Medical Report",
    on_load=on_report_page_load # Add on_load here
)
def report_page_content(): 
    state = me.state(AppState)

    # WRAP THE ENTIRE PAGE CONTENT
    with me.box(style=app_container_style()): # APPLY THE FULL PAGE CONTAINER STYLE
        # The rest of your report page content from the previous version
        # goes here, inside this app_container_style box.
        # ... (render_report_header, logic for displaying progress/report/error, back button) ...
        # For brevity, I'm not repeating the entire report display logic here.
        # Ensure it's correctly placed within this new top-level box.
        
        # Copied from your previous report_page.py (ensure imports for render_ functions are correct)
        render_report_header()

        parsed_report_data = None
        if state.report_data_json and state.report_generation_complete and not state.report_error_message:
            try:
                parsed_report_data = json.loads(state.report_data_json)
            except json.JSONDecodeError:
                if not state.report_error_message:
                    state.report_error_message = "Error: Could not display report (format error)."

        if not state.report_generation_complete and not state.report_is_generating:
            if not state.report_data_json and not state.report_error_message :
                 me.button("Generate Full Medical Report", on_click=trigger_manual_report_generation, style=me.Style(margin=me.Margin(top=20)))
                 me.text("Or complete a chat session to generate a report automatically.", style=me.Style(font_size="12px", color=WHATSAPP_DARK_COLORS["secondary_text"], margin=me.Margin(top=5)))
            elif state.report_error_message:
                 render_error_state(state.report_error_message)
        elif state.report_is_generating:
            render_generation_progress(state.report_generation_progress)
        elif parsed_report_data:
            render_complete_report(parsed_report_data)
        elif state.report_error_message:
            render_error_state(state.report_error_message)
        else: 
            me.text("Report data is not available at this moment.", style=me.Style(color=WHATSAPP_DARK_COLORS["secondary_text"]))

        with me.button("Back to Chat", on_click=lambda e: me.navigate("/chat"), style=me.Style(margin=me.Margin(top=20))):
            pass

# The effect functions (trigger_manual_report_generation, _async_generate_report_from_deepseek, 
# start_report_generation_effect) remain the same as the previous corrected version.
# Make sure they are defined in this file or correctly imported if they are in report_display.py

def trigger_manual_report_generation(e: me.ClickEvent):
    yield from start_report_generation_effect()

async def _async_generate_report_from_deepseek(case_data: dict, app_state_snapshot: AppState) -> dict:
    report_dict = await deepseek_generator.generate_comprehensive_report(case_data, app_state_snapshot)
    return report_dict

def start_report_generation_effect():
    state = me.state(AppState)
    update_report_state_in_app_and_db(
        is_generating=True,
        generation_complete=False,
        error_message=None,
        report_data_json=None,
        generation_progress="Initializing NHS assessment..."
    )
    yield
    try:
        case_data_for_llm = {
            "case_id": state.session_id,
            "chat_history_summary": [msg.content for msg in state.chat_history[-10:]],
        }
        progress_messages = [
            "🔍 Analyzing patient symptoms...",
            "📋 Applying NICE clinical guidelines...",
            "🧠 DeepSeek-R1:14b preparing assessment context..."
        ]
        for progress_msg in progress_messages:
            update_report_state_in_app_and_db(generation_progress=progress_msg)
            yield
            time.sleep(0.5)
        update_report_state_in_app_and_db(generation_progress="🤖 DeepSeek AI is now generating the full report...")
        yield
        report_data_dict = yield from me.effects(
            lambda: _async_generate_report_from_deepseek(case_data_for_llm, state), 
            ()
        )
        update_report_state_in_app_and_db(
            report_data_dict=report_data_dict,
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
    yield
