"""
WhatsApp-style Chat Interface for Medical Triage
Material Design 3 compliant with NHS branding
FIXED: Absolute imports only, explicit style imports
"""

# === SMART IMPORT SETUP - ABSOLUTE IMPORTS ONLY ===
import sys
import os
from pathlib import Path

# Setup paths once to prevent double imports
if not hasattr(sys, '_fairdoc_paths_setup'):
    current_dir = Path(__file__).parent
    msp_dir = current_dir.parent
    frontend_dir = msp_dir.parent
    project_root = frontend_dir.parent
    
    paths_to_add = [
        str(project_root),
        str(frontend_dir),
        str(msp_dir),
        str(msp_dir / "components"),
        str(msp_dir / "styles"),
        str(msp_dir / "utils"),
        str(msp_dir / "mock_backend")
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    sys._fairdoc_paths_setup = True

# Standard imports first
import mesop as me

# Explicit style imports - NO WILDCARDS
from styles.material_theme import (
    MD3_COLORS, MD3_TYPOGRAPHY,
    header_style, message_sent_style, message_received_style,
    input_container_style, input_field_style, send_button_style,
    chat_container_style
)

# Utility imports
from utils.state_manager import get_state, add_message, set_loading

# Mock backend imports
from mock_backend.api_mock import mock_api
from mock_backend.nhs_data import NHS_CHEST_PAIN_QUESTIONS

# === END SMART IMPORT SETUP ===

def render_header():
    """Render chat header"""
    with me.box(style=header_style()):
        with me.box(style=me.Style(display="flex", align_items="center", gap=12)):
            # NHS Logo placeholder
            me.text("🏥", style=me.Style(font_size="24px"))
            with me.box():
                me.text("NHS Digital Triage", style=me.Style(font_weight="500", font_size="18px"))
                me.text("Powered by Fairdoc AI", style=me.Style(font_size="12px", opacity=0.8))

def render_message(message):
    """Render individual chat message"""
    style = message_sent_style() if message.is_user else message_received_style()
    
    with me.box(style=style):
        # Message content
        me.text(message.content, style=MD3_TYPOGRAPHY["body_medium"])
        
        # Timestamp
        timestamp_style = me.Style(
            font_size="11px",
            opacity=0.7,
            margin=me.Margin(top=4),
            text_align="right" if message.is_user else "left"
        )
        me.text(
            message.timestamp.strftime("%H:%M"),
            style=timestamp_style
        )
        
        # Special message types
        if message.message_type == "file":
            with me.box(style=me.Style(margin=me.Margin(top=8))):
                me.text(f"📎 {message.file_url}", style=me.Style(
                    font_size="12px",
                    color=MD3_COLORS["primary"],
                    text_decoration="underline"
                ))

def render_chat_messages():
    """Render all chat messages"""
    state = get_state()
    
    with me.box(style=me.Style(
        flex="1",
        overflow_y="auto",
        padding=me.Padding.all(8),
        display="flex",
        flex_direction="column"
    )):
        for message in state.messages:
            render_message(message)

def handle_send_message(e: me.InputEvent):
    """Handle sending user message"""
    state = get_state()
    user_input = e.value.strip()
    
    if not user_input:
        return
    
    # Add user message
    add_message(user_input, is_user=True)
    
    # Clear input
    state.current_input = ""
    
    # Process response
    process_user_response(user_input)

def process_user_response(user_input: str):
    """Process user response and generate bot reply"""
    state = get_state()
    set_loading(True)
    
    # Simulate bot typing delay
    me.navigate_to_async(simulate_bot_response(user_input))

async def simulate_bot_response(user_input: str):
    """Simulate bot response with NHS questions"""
    state = get_state()
    
    # Check if we're in initial greeting
    if len(state.messages) == 1:  # First user message
        response = """Hello! I'm your NHS Digital Triage assistant. I'll ask you some questions to assess your condition.

Let's start with some basic information."""
        add_message(response, is_user=False)
        
        # Ask first question
        first_question = NHS_CHEST_PAIN_QUESTIONS[0]
        add_message(first_question["question"], is_user=False)
        state.current_question_index = 0
    
    elif not state.questions_completed:
        # Process current answer and move to next question
        await process_question_response(user_input)
    
    set_loading(False)

async def process_question_response(user_input: str):
    """Process answer to current question"""
    state = get_state()
    current_q_index = state.current_question_index
    
    # Store answer (in real app, this would go to case report)
    current_question = NHS_CHEST_PAIN_QUESTIONS[current_q_index]
    
    # Check if this triggers any red flags
    if current_question.get("red_flag") and check_red_flag_response(user_input, current_question):
        add_message("⚠️ Based on your response, this may require urgent medical attention.", is_user=False)
    
    # Move to next question
    state.current_question_index += 1
    
    if state.current_question_index < len(NHS_CHEST_PAIN_QUESTIONS):
        next_question = NHS_CHEST_PAIN_QUESTIONS[state.current_question_index]
        
        # Add acknowledgment
        add_message("Thank you for that information.", is_user=False)
        
        # Ask next question
        add_message(next_question["question"], is_user=False)
        
        # Special handling for file upload question
        if next_question["type"] == "file_upload":
            state.awaiting_file_upload = True
            add_message("You can upload files using the attachment button below.", is_user=False)
    else:
        # Questions completed
        state.questions_completed = True
        add_message("Thank you for providing all the information. I'm now analyzing your responses to create your case report...", is_user=False)
        
        # Trigger case report generation
        await generate_case_report()

def check_red_flag_response(response: str, question: dict) -> bool:
    """Check if response triggers red flag"""
    response_lower = response.lower()
    
    # Simple red flag detection
    if question["type"] == "yes_no":
        return "yes" in response_lower
    elif question["type"] == "text_with_scale":
        # Extract number from response
        try:
            numbers = [int(s) for s in response.split() if s.isdigit()]
            if numbers and max(numbers) >= question.get("red_flag_threshold", 8):
                return True
        except Exception:
            pass
    
    return False

async def generate_case_report():
    """Generate final case report"""
    state = get_state()
    
    add_message("🔄 Generating your case report...", is_user=False)
    
    # Mock API call
    try:
        # Simulate answers collection (in real app, this would be stored properly)
        mock_answers = {
            "age": 45,
            "gender": "Male",
            "pain_severity": 7,
            "shortness_of_breath": "yes",
            "coughing_blood": "no",
            "pain_spreading": "yes"
        }
        
        report_result = await mock_api.generate_case_report(mock_answers)
        
        if report_result["status"] == "success":
            report = report_result["report"]
            
            # Display report summary
            summary = f"""📋 **Case Report Generated**

**Risk Assessment:**
- Urgency: {report['ai_analysis']['coordinates']['urgency']:.1%}
- Importance: {report['ai_analysis']['coordinates']['importance']:.1%}

**Recommendation:** {report['recommendations']['immediate_action']}

**AI Analysis:** {report['ai_analysis']['reasoning']}"""

            add_message(summary, is_user=False, message_type="report")
            
            # Check if emergency
            if report["status"] == "emergency":
                trigger_emergency_response(report)
            else:
                add_message("Your case report has been generated. You can view the full PDF report below.", is_user=False)
        
    except Exception as e:
        add_message("Sorry, there was an error generating your report. Please try again.", f"\n error:{e}", is_user=False)

def trigger_emergency_response(report):
    """Trigger emergency response flow"""
    state = get_state()
    
    add_message("🚨 **URGENT MEDICAL ATTENTION REQUIRED** 🚨", is_user=False, message_type="alert")
    add_message("Based on your symptoms, you need immediate medical care. Emergency services are being contacted.", is_user=False)
    
    # Trigger emergency alert in UI
    state.show_emergency_alert = True
    state.case_report.status = "emergency"

def render_input_area():
    """Render message input area"""
    state = get_state()
    
    with me.box(style=input_container_style()):
        # File upload button
        me.button(
            "📎",
            on_click=handle_file_upload,
            style=me.Style(
                background="transparent",
                border="none",
                font_size="20px",
                cursor="pointer"
            )
        )
        
        # Text input
        me.input(
            value=state.current_input,
            placeholder="Type your message...",
            on_input=handle_input_change,
            on_enter=handle_send_message,
            style=input_field_style()
        )
        
        # Send button
        me.button(
            "Send",
            on_click=handle_send_message_click,
            style=send_button_style(),
            disabled=state.is_loading
        )

def handle_input_change(e: me.InputEvent):
    """Handle input text change"""
    state = get_state()
    state.current_input = e.value

def handle_send_message_click(e: me.ClickEvent):
    """Handle send button click"""
    state = get_state()
    if state.current_input.strip():
        handle_send_message(me.InputEvent(value=state.current_input))

def handle_file_upload(e: me.ClickEvent):
    """Handle file upload"""
    state = get_state()
    
    if state.awaiting_file_upload:
        add_message("📎 File uploaded: chest_xray.jpg", is_user=True, message_type="file")
        add_message("Thank you for uploading the file. It will be analyzed as part of your case report.", is_user=False)
        state.awaiting_file_upload = False
    else:
        add_message("File upload is available when prompted during the assessment.", is_user=False)

def render_chat_interface():
    """Main chat interface component"""
    with me.box(style=chat_container_style()):
        render_header()
        render_chat_messages()
        
        # Loading indicator
        state = get_state()
        if state.is_loading:
            with me.box(style=me.Style(padding=me.Padding.all(16), text_align="center")):
                me.text("⚡ Processing...", style=me.Style(
                    color=MD3_COLORS["primary"],
                    font_style="italic"
                ))
        
        render_input_area()
