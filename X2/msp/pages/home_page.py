# msp/pages/home_page.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from styles.government_digital_styles import base_page_style
from components.home_page_components import (
    render_navigation,
    render_hero_section,
    render_features_section,
    render_statistics_section,
    render_trust_section,
    render_cta_section,
    render_footer
)

def render_home_page():
    """
    Renders the main Fairdoc AI home page with Government Digital Infrastructure aesthetic
    """
    with me.box(style=base_page_style()):
        render_navigation()
        render_hero_section()
        render_features_section()
        render_statistics_section()
        render_trust_section()
        render_cta_section()
        render_footer()
