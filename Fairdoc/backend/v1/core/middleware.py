"""
V1 Middleware Configuration
CORS, security, logging, and performance middleware
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM MIDDLEWARE CLASSES
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging for medical compliance"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"📥 {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"📤 {response.status_code} {request.method} {request.url.path} "
            f"({process_time:.3f}s)"
        )
        
        # Add headers for debugging
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = str(id(request))
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Medical-grade security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers for medical data protection
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-API-Version": "v1",
            "X-Service": "Fairdoc-AI-Triage"
        })
        
        return response

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring and optimization"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{process_time:.3f}s"
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                f"⚠️ Slow request: {request.method} {request.url.path} "
                f"took {process_time:.3f}s"
            )
        
        return response

# =============================================================================
# MIDDLEWARE CONFIGURATION FUNCTION
# =============================================================================

def configure_middleware(app: FastAPI, settings) -> None:
    """Configure all middleware for the V1 API"""
    
    logger.info("🔧 Configuring V1 middleware...")
    
    # 1. CORS Middleware (must be first for preflight requests)
    cors_origins = settings.get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=[
            "X-Process-Time",
            "X-API-Version",
            "X-Total-Count",
            "X-Request-ID",
            "X-Response-Time"
        ]
    )
    logger.info(f"✅ CORS configured with {len(cors_origins)} origins")
    
    # 2. Trusted Host Middleware for security
    if not settings.is_development:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["fairdoc.ai", "*.fairdoc.ai", "localhost", "127.0.0.1"]
        )
        logger.info("✅ Trusted host middleware configured")
    
    # 3. Session Middleware for frontend integration
    app.add_middleware(
        SessionMiddleware,
        secret_key=f"fairdoc_v1_{settings.APP_VERSION}",
        max_age=1800,  # 30 minutes
        same_site="lax",
        https_only=not settings.is_development
    )
    logger.info("✅ Session middleware configured")
    
    # 4. GZip Middleware for performance
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=6
    )
    logger.info("✅ GZip compression configured")
    
    # 5. Custom middleware stack
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(PerformanceMiddleware)
    
    logger.info("✅ Custom middleware stack configured")
    
    # Log final configuration
    logger.info(f"🎯 Middleware configuration complete for {settings.ENVIRONMENT}")

# =============================================================================
# EXPORT
# =============================================================================


__all__ = [
    "configure_middleware",
    "RequestLoggingMiddleware",
    "SecurityHeadersMiddleware",
    "PerformanceMiddleware"
]
