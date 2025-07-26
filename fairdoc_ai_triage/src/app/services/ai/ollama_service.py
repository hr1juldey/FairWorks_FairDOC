"""
Fairdoc AI Ollama Integration Service with Thinking Process Support
"""

import asyncio
import json
from typing import Dict, Any, Optional

import httpx
import structlog

from src.app.core.config import settings
from src.app.models.schemas.context import ConversationContext
from src.app.services.ai.thinking_processor import ThinkingProcessor

logger = structlog.get_logger(__name__)

thinking_processor = ThinkingProcessor()
class OllamaService:
    """Service for integrating with Ollama LLM with thinking process extraction"""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL
        self.client: Optional[httpx.AsyncClient] = None
        self.thinking_processor = ThinkingProcessor()
        
    async def initialize(self):
        """Initialize Ollama service"""
        self.client = httpx.AsyncClient(timeout=120.0)  # Increased timeout for thinking models
        
        # Test connection
        try:
            await self._test_connection()
            logger.info("✅ Ollama service initialized", 
                       base_url=self.base_url, 
                       model=self.model_name)
        except Exception as e:
            logger.error("❌ Failed to initialize Ollama service", error=str(e))
            raise
    
    async def process_message(
        self, 
        message: str, 
        context: ConversationContext,
        user_id: str
    ) -> Dict[str, Any]:
        """Process user message and generate AI response with thinking extraction"""
        
        try:
            # Build context-aware prompt
            system_prompt = self._build_system_prompt(context)
            user_prompt = self._build_user_prompt(message, context)
            
            # Call Ollama API
            raw_response = await self._call_ollama(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Extract thinking process and clean response
            clean_response, thinking_data = self.thinking_processor.extract_thinking_process(
                raw_response.get("response", "")
            )
            
            # Build enhanced AI response
            ai_response = {
                "text": clean_response or "I'm here to help with your healthcare questions.",
                "raw_response": raw_response.get("response", ""),  # Keep raw for debugging
                "intent": self._extract_intent(message),
                "intent_confidence": 0.7,
                "model": self.model_name,
                "context_used": len(context.messages),
                "thinking_process": thinking_data,
                "safety_summary": self.thinking_processor.generate_safety_summary(thinking_data)
            }
            
            logger.info("AI response generated", 
                       user_id=user_id,
                       response_length=len(clean_response) if clean_response else 0,
                       intent=ai_response["intent"],
                       has_thinking=thinking_data is not None,
                       safety_level=ai_response["safety_summary"].get("overall_safety_level"))
            
            return ai_response
            
        except Exception as e:
            logger.error("Error processing message with Ollama", 
                        error=str(e), user_id=user_id)
            
            
            
            error_safety_summary = thinking_processor.generate_safety_summary(None)
            # Return fallback response
            return {
                "text": "I apologize, but I'm experiencing technical difficulties. Please try again shortly.",
                "intent": "error",
                "intent_confidence": 0.0,
                "model": "fallback",
                "error": str(e),
                "thinking_process": None,
                "safety_summary":  error_safety_summary     # {"status": "error"}
            }
    
    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make API call to Ollama"""
        
        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2000  # Allow longer responses for thinking
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
    
    def _build_system_prompt(self, context: ConversationContext) -> str:
        """Build system prompt with context"""
        
        base_prompt = """You are a helpful healthcare AI assistant for Fairdoc AI Triage System.

    Your role is to:
    - Provide helpful, accurate healthcare information
    - Assess symptoms and guide users appropriately  
    - Route urgent cases to healthcare professionals
    - Never provide specific medical diagnoses
    - Always recommend consulting healthcare providers for serious concerns

    Guidelines:
    - Be empathetic and professional
    - Ask clarifying questions when needed
    - Prioritize patient safety
    - Keep responses concise and helpful
    - Think through your reasoning before responding"""

        # Add conversation history if available
        if context.messages:
            recent_context = context.messages[-3:]  # Last 3 interactions
            context_str = "\n\nRecent conversation context:\n"
            for msg in recent_context:
                # FIX: Access dictionary keys, not object attributes
                user_text = msg["user_message"].get("text", "") if isinstance(msg, dict) else ""
                ai_text = msg["ai_response"].get("text", "") if isinstance(msg, dict) else ""
                context_str += f"User: {user_text}\nAssistant: {ai_text}\n"
            
            base_prompt += context_str
        
        return base_prompt
    
    def _build_user_prompt(self, message: str, context: ConversationContext) -> str:
        """Build user prompt with current message"""
        return message
    
    def _extract_intent(self, message: str) -> str:
        """Simple rule-based intent extraction"""
        
        message_lower = message.lower()
        
        # Emergency/urgent keywords
        if any(word in message_lower for word in ['emergency', 'urgent', 'severe pain', 'chest pain', 'difficulty breathing']):
            return 'emergency'
        
        # Symptom assessment
        elif any(word in message_lower for word in ['pain', 'symptoms', 'feeling', 'headache', 'fever']):
            return 'symptom_assessment'
        
        # Appointment scheduling
        elif any(word in message_lower for word in ['appointment', 'schedule', 'book', 'availability']):
            return 'appointment_booking'
        
        # General information
        elif any(word in message_lower for word in ['what is', 'how to', 'information', 'explain']):
            return 'information_request'
        
        else:
            return 'general_query'
    
    async def _test_connection(self):
        """Test Ollama connection"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            logger.info("Ollama connection test successful")
        except Exception as e:
            logger.error("Ollama connection test failed", error=str(e))
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
        logger.info("🧹 Ollama service cleanup completed")
