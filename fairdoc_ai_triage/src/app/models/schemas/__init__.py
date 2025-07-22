"""
Fairdoc AI Schema Models
"""

from .context import (
    ConversationContext,
    ConversationMessage,
    UserProfile,
    SessionState,
    StakeholderRoute
)

from .chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    HealthCheckResponse
)

__all__ = [
    "ConversationContext",
    "ConversationMessage", 
    "UserProfile",
    "SessionState",
    "StakeholderRoute",
    "ChatMessageRequest",
    "ChatMessageResponse",
    "HealthCheckResponse"
]
