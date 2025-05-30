import mesop as me
import utils.path_setup 
from state.state_manager import AppState, ReportContent
from styles.whatsapp_dark_theme import WHATSAPP_DARK_COLORS

@me.page(path="/report", title="Medical Report")
def report_page_content(): # Renamed
    state = me.state(AppState)
    if state.report_content is None: 
        state.report_content = ReportContent(
            sections=[{"header": "Placeholder", "text": "Report generation not yet fully implemented."}]
        )

    with me.box(style=me.Style(padding=me.Padding.all(20), background=WHATSAPP_DARK_COLORS["app_bg"], height="100vh", color=WHATSAPP_DARK_COLORS["primary_text"])):
        me.text(state.report_content.title, type="headline-4", style=me.Style(margin=me.Margin(bottom=20)))

        if state.report_content.is_generating:
            me.progress_spinner()
            me.text("Generating your detailed medical report...", style=me.Style(margin=me.Margin(top=10)))
        elif state.report_content.error:
            me.text(f"Error: {state.report_content.error}", style=me.Style(color=WHATSAPP_DARK_COLORS["error_text"]))
        elif state.report_content.sections:
            for section in state.report_content.sections:
                me.text(section.get("header", "Section"), type="headline-5", style=me.Style(margin=me.Margin(top=15, bottom=5)))
                me.markdown(section.get("text", "No content for this section."))
        else:
            me.text("No report data available. Please complete a chat session first.")

        with me.button("Back to Chat", on_click=lambda e: me.navigate("/chat")): # Navigate to /chat
            pass