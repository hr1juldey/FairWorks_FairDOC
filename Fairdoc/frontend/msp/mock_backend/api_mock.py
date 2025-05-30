# File: Fairdoc\frontend\msp\mock_backend\api_mock.py

"""
Mock Backend API for Fairdoc AI
Simulates FastAPI backend responses
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Any
from .nhs_data import NHS_CHEST_PAIN_QUESTIONS, calculate_risk_score

class MockAPI:
    """Mock API client to simulate backend calls"""
    
    @staticmethod
    async def process_answer(question_id: int, answer: str) -> Dict[str, Any]:
        """Process user answer and return next question or result"""
        # Simulate API delay
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        current_question_index = question_id - 1
        
        # Check if this is the last question
        if current_question_index >= len(NHS_CHEST_PAIN_QUESTIONS) - 1:
            return {
                "status": "complete",
                "message": "Assessment complete. Analyzing your responses...",
                "next_action": "generate_report"
            }
        
        # Return next question
        next_question = NHS_CHEST_PAIN_QUESTIONS[current_question_index + 1]
        return {
            "status": "continue",
            "next_question": next_question,
            "message": "Thank you. Next question:"
        }
    
    @staticmethod
    async def upload_file(file_data: bytes, filename: str) -> Dict[str, Any]:
        """Mock file upload to MinIO"""
        # Simulate upload delay
        await asyncio.sleep(random.uniform(2, 4))
        
        mock_url = f"https://mock-minio.fairdoc.ai/files/{filename}"
        
        return {
            "status": "success",
            "file_url": mock_url,
            "file_id": f"file_{random.randint(1000, 9999)}",
            "message": f"Successfully uploaded {filename}"
        }
    
    @staticmethod
    async def generate_case_report(answers: Dict[str, Any]) -> Dict[str, Any]:
        """Generate case report with AI analysis"""
        # Simulate AI processing
        await asyncio.sleep(random.uniform(3, 6))
        
        # Calculate risk assessment
        risk_analysis = calculate_risk_score(answers)
        
        # Generate mock report
        report = {
            "case_id": f"CASE_{random.randint(10000, 99999)}",
            "patient_summary": {
                "age": answers.get("age", "Not provided"),
                "gender": answers.get("gender", "Not provided"),
                "chief_complaint": "Left chest pain with coughing and wheezing"
            },
            "risk_assessment": risk_analysis,
            "recommendations": {
                "immediate_action": risk_analysis["recommendation"],
                "follow_up": "Monitor symptoms and seek medical attention if worsening",
                "general_advice": "Rest, avoid strenuous activity, stay hydrated"
            },
            "ai_analysis": {
                "coordinates": {
                    "urgency": risk_analysis["urgency"],
                    "importance": risk_analysis["importance"]
                },
                "reasoning": risk_analysis["reasoning"],
                "confidence": random.uniform(0.8, 0.95)
            },
            "generated_at": datetime.now().isoformat(),
            "status": "emergency" if risk_analysis["urgency"] > 0.8 else "completed"
        }
        
        return {
            "status": "success",
            "report": report,
            "pdf_url": f"https://mock-reports.fairdoc.ai/report_{report['case_id']}.pdf"
        }
    
    @staticmethod
    async def notify_emergency_services(case_id: str) -> Dict[str, Any]:
        """Mock emergency service notification"""
        await asyncio.sleep(1)
        
        return {
            "status": "success",
            "message": "Emergency services have been notified",
            "reference_number": f"EMG_{random.randint(100000, 999999)}",
            "estimated_arrival": "8-12 minutes"
        }

# Global mock API instance
mock_api = MockAPI()
