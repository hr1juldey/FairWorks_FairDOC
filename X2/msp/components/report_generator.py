# msp/components/report_generator.py

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
# Ensures paths are set up correctly for imports from other project modules
import utils.path_setup

from utils.path_setup import setup_project_paths  # Ensures paths are set up
setup_project_paths()
# Import models and state if needed by the generator logic directly
from state.state_manager import AppState # Assuming AppState might be used for context
# from models import MedicalReport # If you have a Pydantic model for the report structure from a 'models.py'

logger = logging.getLogger(__name__)
# Ensure logging is configured, typically in main.py or a central logging setup
# If not, basicConfig can be called here for this module if run independently for testing.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
    logger.info("Ollama client imported successfully for report_generator.")
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not found in report_generator. Report generation will use mock/fallback data.")


class DeepSeekReportGenerator:
    """
    Handles the generation of comprehensive medical reports using a DeepSeek LLM model
    via Ollama.
    """
    
    def __init__(self, model_name: str = "deepseek-coder:1.3b-instruct"): # Default, adjust to your actual DeepSeek model
        self.model_name = model_name # e.g., "deepseek-r1:14b" if that's what you have and intend to use
        if OLLAMA_AVAILABLE:
            self.client = AsyncClient()
        else:
            self.client = None
    
    async def generate_comprehensive_report(
        self, 
        case_data: Dict[str, Any], 
        app_state_snapshot: AppState # A snapshot of the app state for context
    ) -> Dict[str, Any]: # Returns a dictionary representing the structured report
        """
        Generates a comprehensive NHS-compliant medical report using the configured DeepSeek model.
        
        Args:
            case_data: Dictionary containing necessary information for report generation,
                       e.g., {"case_id": "...", "chat_history_summary": [...]}.
            app_state_snapshot: A snapshot of the application state for broader context if needed.
                                It's a good practice to pass specific data rather than the whole state
                                to keep dependencies clear.

        Returns:
            A dictionary representing the structured medical report.
        """
        
        if not self.client:
            logger.warning(f"Ollama client unavailable. Using fallback report for case: {case_data.get('case_id')}")
            return self._fallback_report(case_data, app_state_snapshot)

        prompt = self._create_nice_assessment_prompt(case_data, app_state_snapshot)
        
        try:
            logger.info(f"Generating report with {self.model_name} for case: {case_data.get('case_id', 'N/A')}. Prompt length: {len(prompt)} chars.")
            
            response = await self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format="json", # Request JSON output from Ollama if the model supports it well
                stream=False  # Get the full response at once for easier JSON parsing
            )
            
            raw_response_content = response['message']['content']
            logger.info(f"Raw LLM response for report (first 200 chars): {raw_response_content[:200]}...")
            
            report_data_dict = self._parse_deepseek_response(raw_response_content)
            
            # You can add post-processing or enhancement steps here if needed
            # report_data_dict = self._enhance_report(report_data_dict, case_data)
            
            logger.info(f"Successfully generated and parsed report for case: {case_data.get('case_id', 'N/A')}")
            return report_data_dict
            
        except Exception as e:
            logger.error(f"DeepSeek report generation failed for case {case_data.get('case_id', 'N/A')}: {e}", exc_info=True)
            return self._fallback_report(case_data, app_state_snapshot) # Return a fallback dict on error
    
    def _create_nice_assessment_prompt(self, case_data: Dict[str, Any], app_state_snapshot: AppState) -> str:
        """
        Creates a detailed, NICE-compliant assessment prompt for the DeepSeek LLM.
        This prompt should guide the LLM to output a structured JSON report.
        """
        # Example: Extract chat history summary from case_data
        chat_summary = "\n".join(case_data.get("chat_history_summary", ["No chat summary provided."]))
        session_id = case_data.get("case_id", "UNKNOWN_SESSION")

        # This prompt needs to be carefully engineered.
        # It should clearly define the expected JSON structure.
        prompt = f"""
        Objective: Generate a comprehensive NHS Clinical Assessment Report based on the provided patient interaction summary.
        Adherence: Strictly follow NICE (National Institute for Health and Care Excellence) clinical guidelines relevant to the presented symptoms.
        Output Format: Provide the report exclusively in JSON format, adhering to the structure defined below. Do not include any conversational text, preambles, or apologies.

        Patient Session ID: {session_id}
        Date of Assessment: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Patient Interaction Summary:
        ---
        {chat_summary}
        ---

        JSON Report Structure Required:
        {{
          "report_metadata": {{
            "report_id": "AUTO_GENERATED_UUID",
            "case_id": "{session_id}",
            "generation_timestamp": "{datetime.now().isoformat()}",
            "generating_model": "{self.model_name}",
            "nice_guideline_references": ["e.g., NICE CG95", "NICE NG185"]
          }},
          "patient_summary": {{
            "presenting_complaint": "Concise summary of the patient's main reason for consultation.",
            "history_of_presenting_complaint": "Detailed chronological account of symptoms, including onset, duration, severity, character, associated symptoms, and relieving/exacerbating factors.",
            "relevant_medical_history": "List significant past medical conditions, surgeries, allergies, and current medications."
          }},
          "clinical_findings_and_assessment": {{
            "symptoms_analysis": [
              {{"symptom": "Name of symptom", "details": "Description, severity, etc."}}
            ],
            "red_flag_assessment": {{
              "flags_identified": ["List any red flags identified during triage"],
              "implications": "Clinical implications of identified red flags."
            }},
            "differential_diagnoses": [
              {{"diagnosis": "Potential diagnosis 1", "likelihood": "High/Medium/Low", "reasoning": "Clinical basis"}},
              {{"diagnosis": "Potential diagnosis 2", "likelihood": "High/Medium/Low", "reasoning": "Clinical basis"}}
            ],
            "provisional_diagnosis": "Most likely diagnosis based on available information."
          }},
          "risk_stratification": {{
            "overall_risk_level": "Low/Medium/High/Critical",
            "specific_risk_scores": {{"e.g., HEART_score": "Calculated score if applicable"}},
            "contributing_factors": ["List factors increasing risk"],
            "mitigating_factors": ["List factors decreasing risk"]
          }},
          "management_plan_and_recommendations": {{
            "immediate_actions_required": ["Urgent steps to be taken, if any."],
            "recommended_investigations": ["e.g., ECG, Blood tests (specify)"],
            "treatment_suggestions": ["Pharmacological and non-pharmacological, if appropriate for triage stage."],
            "referral_pathway": "Recommended referral (e.g., GP, A&E, Specialist Clinic).",
            "safety_netting_advice": "Clear instructions on when to seek further help or what to do if symptoms worsen.",
            "patient_education": "Brief advice or information for the patient."
          }},
          "disposition": {{
            "recommended_outcome": "e.g., Attend A&E within 1 hour, See GP within 24 hours, Self-care",
            "urgency_of_follow_up": "Specific timeframe for the recommended action."
          }}
        }}

        Instructions for LLM:
        - Fill all fields in the JSON structure. If information is unavailable from the summary, state "Not assessed" or "Information not available".
        - Be clinically precise and use appropriate medical terminology.
        - Prioritize patient safety in all recommendations.
        - Base risk stratification and disposition on NICE guidelines.
        - Ensure the entire output is a single, valid JSON object.
        """
        return prompt.strip()

    def _parse_deepseek_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the JSON string response from the DeepSeek LLM.
        Handles potential errors in JSON formatting.
        """
        try:
            # Attempt to find and parse JSON within the response text
            # LLMs sometimes add leading/trailing text or code block markers
            json_start_index = response_text.find('{')
            json_end_index = response_text.rfind('}') + 1

            if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                json_str = response_text[json_start_index:json_end_index]
                parsed_json = json.loads(json_str)
                logger.info("Successfully parsed JSON response from DeepSeek.")
                return parsed_json
            else:
                logger.warning("No valid JSON object found in DeepSeek response.")
                return {"error": "LLM did not return valid JSON", "raw_response": response_text}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from DeepSeek response: {e}. Raw response: {response_text[:500]}...")
            return {"error": f"JSON parsing failed: {e}", "raw_response": response_text}
        except Exception as ex:
            logger.error(f"An unexpected error occurred during DeepSeek response parsing: {ex}")
            return {"error": f"Unexpected parsing error: {ex}", "raw_response": response_text}

    def _fallback_report(self, case_data: Dict[str, Any], app_state_snapshot: AppState) -> Dict[str, Any]:
        """
        Generates a fallback report when the DeepSeek LLM fails or is unavailable.
        This should still be a dictionary matching the expected report structure.
        """
        session_id = case_data.get("case_id", "N/A")
        logger.warning(f"Generating fallback report for case: {session_id}")
        
        # Try to get some basic info from the AppState snapshot if available
        presenting_complaint_fallback = "Patient interaction summary not fully processed due to system error."
        if app_state_snapshot and app_state_snapshot.chat_history:
            user_messages = [m.content for m in app_state_snapshot.chat_history if m.role == "user"]
            if user_messages:
                presenting_complaint_fallback = f"Initial concerns included: {'; '.join(user_messages[:2])}"
        
        return {
          "report_metadata": {
            "report_id": str(uuid.uuid4()), # Generate a unique ID for the fallback
            "case_id": session_id,
            "generation_timestamp": datetime.now().isoformat(),
            "generating_model": "FallbackSystemReport",
            "nice_guideline_references": ["N/A - Fallback Report"]
          },
          "patient_summary": {
            "presenting_complaint": presenting_complaint_fallback,
            "history_of_presenting_complaint": "Details unavailable due to report generation error. Clinical review required.",
            "relevant_medical_history": "Not assessed due to error."
          },
          "clinical_findings_and_assessment": {
            "symptoms_analysis": [{"symptom": "Unknown", "details": "System error prevented full symptom analysis."}],
            "red_flag_assessment": {
              "flags_identified": ["System Error - Unable to assess red flags"],
              "implications": "Urgent manual clinical review is necessary."
            },
            "differential_diagnoses": [],
            "provisional_diagnosis": "Requires Manual Clinical Assessment"
          },
          "risk_stratification": {
            "overall_risk_level": "High", # Default to high risk on error for safety
            "specific_risk_scores": {},
            "contributing_factors": ["System error during assessment"],
            "mitigating_factors": []
          },
          "management_plan_and_recommendations": {
            "immediate_actions_required": ["Urgent manual clinical review by a healthcare professional."],
            "recommended_investigations": ["As determined by clinical review."],
            "treatment_suggestions": ["As determined by clinical review."],
            "referral_pathway": "Urgent Clinical Review",
            "safety_netting_advice": "If symptoms worsen or new critical symptoms appear, seek immediate emergency care (e.g., A&E or call 999/111 as appropriate).",
            "patient_education": "A system error occurred during your AI assessment. Please contact a healthcare professional for advice."
          },
          "disposition": {
            "recommended_outcome": "Urgent Clinical Review Required",
            "urgency_of_follow_up": "Immediate"
          },
          "system_error_notes": {
              "error_type": "ReportGenerationFailure",
              "message": "The AI report generator encountered an issue. This is a system-generated fallback. Please ensure this case is manually reviewed by a clinician."
          }
        }

# Global instance of the report generator
deepseek_generator = DeepSeekReportGenerator()
