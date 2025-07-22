"""
Fairdoc AI Raven Chat Integration Service
"""

import json
from typing import Dict, Any, Optional

import httpx
import structlog

from src.app.core.config import settings

logger = structlog.get_logger(__name__)


class RavenChatService:
    """Service for integrating with Raven Chat"""
    
    def __init__(self):
        self.webhook_url = settings.RAVEN_WEBHOOK_URL
        self.api_key = settings.RAVEN_API_KEY
        self.secret = settings.RAVEN_SECRET
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self):
        """Initialize Raven Chat service"""
        self.client = httpx.AsyncClient(timeout=10.0)
        logger.info("✅ Raven Chat service initialized")
    
    async def send_message(
        self, 
        channel_id: str, 
        message: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """Send message to Raven Chat"""
        
        try:
            payload = {
                "channel_id": channel_id,
                "message": message,
                "user_id": user_id,
                "message_type": "text"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                self.webhook_url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            
            logger.info("Message sent to Raven Chat", 
                       channel_id=channel_id, user_id=user_id)
            
            return response.json()
            
        except Exception as e:
            logger.error("Failed to send message to Raven Chat", 
                        error=str(e), channel_id=channel_id)
            raise
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook from Raven Chat"""
        
        try:
            # Extract message data
            message = webhook_data.get("message", "")
            user_id = webhook_data.get("user_id", "")
            channel_id = webhook_data.get("channel_id", "")
            
            # Process the message (this will be connected to main chat processing)
            processed_data = {
                "user_id": user_id,
                "message": message,
                "channel_id": channel_id,
                "source": "raven_chat",
                "processed": True
            }
            
            logger.info("Webhook processed from Raven Chat", 
                       user_id=user_id, channel_id=channel_id)
            
            return processed_data
            
        except Exception as e:
            logger.error("Failed to process Raven Chat webhook", error=str(e))
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
        logger.info("🧹 Raven Chat service cleanup completed")
