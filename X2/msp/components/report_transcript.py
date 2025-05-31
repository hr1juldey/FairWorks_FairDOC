# msp/components/report_transcript.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import Dict, List, Any

from styles.government_digital_styles import GOVERNMENT_COLORS
from utils.report_helpers import extract_key_quotes, create_citation_map
from utils.transcript_helpers import (
    group_phrases_by_category,
    check_message_contains_flags,
    get_role_style_config,
    get_severity_color_mapping,
    get_category_icon_mapping,
    highlight_flagged_phrases,
    extract_flagged_phrase_list,
    validate_message_data,
    validate_quote_data,
    calculate_citation_number,
    create_conversation_statistics,
    format_category_assessment_text,
    get_default_icon_for_category
)

def render_transcript_tab(report_data: Dict[str, Any]):
    """Render Tab 3: Clinical Transcript with Perplexity-style citations"""
    
    chat_history = report_data.get("chat_history", [])
    urgency_data = report_data.get("urgency_analysis", {})
    flagged_phrases = urgency_data.get("flagged_phrases", [])
    
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=24)):
        render_transcript_overview(chat_history, urgency_data)
        render_key_quotes_section(chat_history, flagged_phrases)
        render_full_transcript(chat_history, flagged_phrases)
        render_clinical_inference_summary(flagged_phrases)

def render_transcript_overview(chat_history: List[Dict], urgency_data: Dict[str, Any]):
    """Render conversation overview statistics"""
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="12px",
        padding=me.Padding.all(24),
        box_shadow="0 2px 8px rgba(0,0,0,0.1)"
    )):
        me.text("💬 Conversation Overview", type="headline-5", style=me.Style(margin=me.Margin(bottom=16)))
        
        with me.box(style=me.Style(
            display="grid",
            grid_template_columns="repeat(auto-fit, minmax(200px, 1fr))",
            gap=20
        )):
            # Use utility function to get statistics
            stats = create_conversation_statistics(chat_history, urgency_data)
            for label, value in stats:
                render_stat_card(label, value)

def render_stat_card(label: str, value: str):
    """Render individual statistic card"""
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        padding=me.Padding.all(16),
        border_radius="8px",
        text_align="center"
    )):
        me.text(value, style=me.Style(
            font_size="1.5rem",
            font_weight="700",
            color=GOVERNMENT_COLORS["primary"],
            margin=me.Margin(bottom=4)
        ))
        me.text(label, style=me.Style(
            color=GOVERNMENT_COLORS["text_secondary"],
            font_weight="500"
        ))

def render_key_quotes_section(chat_history: List[Dict], flagged_phrases: List[Dict]):
    """Render key clinical quotes with Perplexity-style citations"""
    
    quotes = extract_key_quotes(chat_history, flagged_phrases)
    citation_map = create_citation_map(quotes)
    
    with me.accordion():
        with me.expansion_panel(
            key="key_quotes",
            title="🔍 Key Clinical Statements",
            description="Critical patient statements with clinical significance",
            expanded=True
        ):
            if quotes:
                for quote in quotes:
                    citation_number = calculate_citation_number(citation_map, quote["flagged_phrase"])
                    render_quoted_statement(quote, citation_number)
            else:
                render_no_quotes_message()

def render_no_quotes_message():
    """Render message when no quotes are available"""
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        padding=me.Padding.all(20),
        border_radius="8px",
        text_align="center"
    )):
        me.text("💬 No critical clinical statements identified in this conversation", 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"]))

def render_quoted_statement(quote: Dict[str, Any], citation_number: int):
    """Render individual quoted statement with clinical inference"""
    
    quote_data = validate_quote_data(quote)
    severity_colors = get_severity_color_mapping()
    severity_color = severity_colors.get(quote_data["severity"], GOVERNMENT_COLORS["medium_grey"])
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        render_quote_header(citation_number, quote_data["severity"], severity_color)
        render_quote_content(quote_data["quote"])
        render_clinical_inference(quote_data["inference"])
        render_quote_footer(quote_data["clinical_category"], quote_data["flagged_phrase"], citation_number)

def render_quote_header(citation_number: int, severity: str, severity_color: str):
    """Render quote header with citation number and severity"""
    with me.box(style=me.Style(display="flex", align_items="center", margin=me.Margin(bottom=12))):
        with me.box(style=me.Style(
            background=GOVERNMENT_COLORS["primary"],
            color="white",
            width="24px",
            height="24px",
            border_radius="50%",
            display="flex",
            align_items="center",
            justify_content="center",
            font_size="0.8rem",
            font_weight="600",
            margin=me.Margin(right=12)
        )):
            me.text(str(citation_number))
        
        with me.box(style=me.Style(
            background=severity_color,
            color="white",
            padding=me.Padding.symmetric(horizontal=8, vertical=4),
            border_radius="12px",
            font_size="0.7rem",
            font_weight="600"
        )):
            me.text(severity.upper())

def render_quote_content(quote: str):
    """Render the actual quote content"""
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        padding=me.Padding.all(12),
        border_radius="6px",
        border_left=f"4px solid {GOVERNMENT_COLORS['primary']}",
        margin=me.Margin(bottom=12)
    )):
        me.text(f'"{quote}"', style=me.Style(
            font_style="italic",
            line_height="1.5",
            color=GOVERNMENT_COLORS["text_primary"]
        ))

def render_clinical_inference(inference: str):
    """Render clinical inference section"""
    me.text("Clinical Significance:", style=me.Style(
        font_weight="600",
        color=GOVERNMENT_COLORS["text_primary"],
        margin=me.Margin(bottom=4)
    ))
    me.text(inference, style=me.Style(
        color=GOVERNMENT_COLORS["text_secondary"],
        line_height="1.4"
    ))

def render_quote_footer(category: str, flagged_phrase: str, citation_number: int):
    """Render quote footer with category and citation"""
    with me.box(style=me.Style(
        display="flex",
        justify_content="space-between",
        align_items="center",
        margin=me.Margin(top=8),
        padding=me.Padding(top=8),
        border_top=f"1px solid {GOVERNMENT_COLORS['light_grey']}"
    )):
        me.text(f"Category: {category.title()}", 
               style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["text_muted"]))
        
        with me.box(style=me.Style(display="flex", align_items="center", gap=8)):
            me.text(f"Key phrase: '{flagged_phrase}'", 
                   style=me.Style(font_size="0.9rem", color=GOVERNMENT_COLORS["primary"], font_weight="500"))
            
            with me.box(style=me.Style(
                background=GOVERNMENT_COLORS["primary"],
                color="white",
                padding=me.Padding.symmetric(horizontal=6, vertical=2),
                border_radius="4px",
                font_size="0.7rem",
                font_weight="600"
            )):
                me.text(f"[{citation_number}]")

def render_full_transcript(chat_history: List[Dict], flagged_phrases: List[Dict]):
    """Render complete conversation transcript with highlighting"""
    
    flagged_phrase_list = extract_flagged_phrase_list(flagged_phrases)
    
    with me.accordion():
        with me.expansion_panel(
            key="full_transcript",
            title="📝 Complete Conversation Transcript",
            description="Full conversation with clinical annotations",
            expanded=False
        ):
            with me.box(style=me.Style(
                max_height="500px",
                overflow_y="auto",
                background=GOVERNMENT_COLORS["bg_secondary"],
                border_radius="8px",
                padding=me.Padding.all(16)
            )):
                for i, message in enumerate(chat_history):
                    render_transcript_message(message, i + 1, flagged_phrase_list)

def render_transcript_message(message: Dict[str, Any], message_number: int, flagged_phrases: List[str]):
    """Render individual message in transcript"""
    
    role, content, timestamp = validate_message_data(message)
    contains_flags = check_message_contains_flags(content, flagged_phrases)
    role_info = get_role_style_config(role)
    
    with me.box(style=me.Style(
        background=role_info["bg"],
        border=me.Border.all(me.BorderSide(
            width=2 if contains_flags else 1,
            style="solid",
            color=GOVERNMENT_COLORS["error"] if contains_flags else GOVERNMENT_COLORS["medium_grey"]
        )),
        border_radius="8px",
        padding=me.Padding.all(12),
        margin=me.Margin(bottom=12)
    )):
        render_message_header(role_info, message_number, contains_flags, timestamp)
        render_message_content(content, flagged_phrases)

def render_message_header(role_info: Dict, message_number: int, contains_flags: bool, timestamp: str):
    """Render message header with role, number, flags, and timestamp"""
    with me.box(style=me.Style(
        display="flex",
        justify_content="space-between",
        align_items="center",
        margin=me.Margin(bottom=8)
    )):
        with me.box(style=me.Style(display="flex", align_items="center")):
            me.text(role_info["icon"], style=me.Style(margin=me.Margin(right=8)))
            me.text(f"{role_info['name']} #{message_number}", style=me.Style(
                font_weight="600",
                color=GOVERNMENT_COLORS["text_primary"]
            ))
            
            if contains_flags:
                with me.box(style=me.Style(
                    background=GOVERNMENT_COLORS["error"],
                    color="white",
                    padding=me.Padding.symmetric(horizontal=6, vertical=2),
                    border_radius="4px",
                    font_size="0.7rem",
                    margin=me.Margin(left=8)
                )):
                    me.text("FLAGGED")
        
        if timestamp:
            me.text(timestamp, style=me.Style(
                font_size="0.8rem",
                color=GOVERNMENT_COLORS["text_muted"]
            ))

def render_message_content(content: str, flagged_phrases: List[str]):
    """Render message content with phrase highlighting"""
    highlighted_content = highlight_flagged_phrases(content, flagged_phrases)
    me.text(highlighted_content, style=me.Style(
        line_height="1.5",
        color=GOVERNMENT_COLORS["text_primary"]
    ))

def render_clinical_inference_summary(flagged_phrases: List[Dict]):
    """Render summary of clinical inferences"""
    
    with me.accordion():
        with me.expansion_panel(
            key="clinical_summary",
            title="🧠 Clinical Inference Summary",
            description="AI-generated clinical insights from conversation",
            expanded=False
        ):
            if flagged_phrases:
                categories = group_phrases_by_category(flagged_phrases)
                for category, phrases in categories.items():
                    render_category_summary(category, phrases)
            else:
                render_no_clinical_concerns()

def render_no_clinical_concerns():
    """Render message when no clinical concerns are identified"""
    with me.box(style=me.Style(
        background=GOVERNMENT_COLORS["bg_secondary"],
        padding=me.Padding.all(20),
        border_radius="8px",
        text_align="center"
    )):
        me.text("✅ No significant clinical concerns identified in conversation patterns", 
               style=me.Style(color=GOVERNMENT_COLORS["success"]))

def render_category_summary(category: str, phrases: List[Dict]):
    """Render summary for specific clinical category"""
    
    icon = get_default_icon_for_category(category)
    
    with me.box(style=me.Style(
        background="white",
        border=me.Border.all(me.BorderSide(width=1, style="solid", color=GOVERNMENT_COLORS["medium_grey"])),
        border_radius="8px",
        padding=me.Padding.all(16),
        margin=me.Margin(bottom=16)
    )):
        me.text(f"{icon} {category.title()} Assessment", type="headline-6", 
               style=me.Style(margin=me.Margin(bottom=12)))
        
        me.text(format_category_assessment_text(category, len(phrases)), 
               style=me.Style(color=GOVERNMENT_COLORS["text_secondary"], margin=me.Margin(bottom=8)))
        
        for phrase in phrases:
            with me.box(style=me.Style(margin=me.Margin(left=16, bottom=4))):
                me.text(f"• {phrase.get('phrase', '')}", style=me.Style(
                    color=GOVERNMENT_COLORS["text_primary"],
                    font_weight="500"
                ))
