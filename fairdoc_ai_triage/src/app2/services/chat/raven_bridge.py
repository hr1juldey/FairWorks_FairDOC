"""
Raven Chat Bridge (V2)
Thin async wrapper to post conversation events to the external Raven
micro-service.  Keeps ≤200 LOC and degrades gracefully when Raven is
unavailable so the core triage flow is never blocked.
"""
from __future__ import annotations

import os
from typing import Dict, Any, Optional

import httpx
import structlog

from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)


class RavenBridge:  # pylint: disable=too-few-public-methods
    """Send messages to the Raven chat service.

    The real Raven stack runs in a separate Docker container and exposes a
    simple webhook (default ``POST /api/method/raven.api.raven_message``)
    that accepts JSON payloads.  We *never* await the response in the
    hot API path—errors are logged but not raised so triage continues
    even if Raven is down.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or settings_v2.RAVEN_WEBHOOK_URL
        self.api_key = api_key or settings_v2.RAVEN_API_KEY
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def send_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fire-and-forget message to Raven.

        Parameters
        ----------
        conversation_id : str
            Current Fairdoc conversation ID so Raven can thread messages.
        user_id : str
            Originating user (patient) ID; echoed back in Raven UI.
        content : str
            Plain-text message to deliver.
        metadata : dict | None
            Additional structured data (e.g.
            ``{"medical_outcome": "emergency"}``).
        """
        payload = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "content": content,
            "metadata": metadata or {},
        }

        headers = {"X-API-KEY": self.api_key}

        try:
            client = await self._get_client()
            resp = await client.post(self.base_url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.error(
                    "❌ Raven webhook error",
                    status=resp.status_code,
                    text=resp.text,
                    url=self.base_url,
                )
            else:
                logger.debug(
                    "📤 Raven message dispatched",
                    conversation_id=conversation_id,
                    status=resp.status_code,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "⚠️ Raven unreachable — continuing without external chat",
                error=str(exc),
                url=self.base_url,
            )

    async def close(self) -> None:
        """Close the underlying HTTP client (used during shutdown)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Convenience singleton (mirrors pattern used elsewhere)
raven_bridge = RavenBridge()
