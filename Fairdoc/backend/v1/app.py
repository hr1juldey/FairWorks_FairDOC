"""
FastAPI main application for Fairdoc Medical AI Backend.
Environment-aware FastAPI app with authentication, WebSocket support, and health monitoring.
"""
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List
import uvicorn
import asyncio
import logging
from datetime import datetime

# Import core modules
from core.config import settings, db_manager, current_environment
from core.websocket_manager import ConnectionManager  # type: ignore
from datamodels.auth_models import UserLogin, TokenResponse, UserCreate
from services.auth_service import AuthService  # type: ignore

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with environment-specific settings
app = FastAPI(
    title=settings.APP_NAME,
    description="Fairdoc Medical AI Backend - Bias-aware chest pain triage system",
    version="1.0.0",
    debug=settings.DEBUG,
    docs_url="/docs" if not settings.is_production else None,  # Disable docs in production
    redoc_url="/redoc" if not settings.is_production else None
)

# Add security middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )

# Add CORS middleware with environment-specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize global services
auth_service = AuthService()
websocket_manager = ConnectionManager()
security = HTTPBearer()

@app.on_event("startup")
async def startup_event():
    """Initialize all services on application startup."""
    logger.info(f"🚀 Starting Fairdoc backend ({current_environment} environment)")
    
    try:
        # Initialize database connections
        if not await db_manager.initialize():
            raise Exception("Database initialization failed")
        
        # Initialize authentication service
        if not await auth_service.initialize():
            raise Exception("Auth service initialization failed")
        
        # TODO: Initialize RAG service (services/rag_service.py)
        # TODO: Initialize NLP service (services/nlp_service.py)
        # TODO: Initialize ML classifiers (MLmodels/classifier.py)
        # TODO: Initialize image diagnosis service (services/image_diagnosis.py)
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("👋 Shutting down Fairdoc backend...")
    
    try:
        # Disconnect all WebSocket connections
        await websocket_manager.disconnect_all()
        
        # Close database connections
        await db_manager.close()
        
        # TODO: Cleanup ML model resources
        # TODO: Cleanup RAG service resources
        
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception: {exc}")
    
    if settings.DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# ============================================================================
# AUTHENTICATION REST ENDPOINTS
# ============================================================================

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(user_credentials: UserLogin):
    """Authenticate user and return JWT tokens."""
    try:
        tokens = await auth_service.authenticate_user(
            user_credentials.username,
            user_credentials.password
        )
        logger.info(f"User {user_credentials.username} logged in successfully")
        return TokenResponse(**tokens)
        
    except Exception as e:
        logger.warning(f"Login failed for {user_credentials.username}: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Refresh JWT access token using refresh token."""
    try:
        new_tokens = await auth_service.refresh_access_token(credentials.credentials)
        return TokenResponse(**new_tokens)
        
    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.post("/auth/register", response_model=dict, tags=["Authentication"])
async def register_user(user_data: UserCreate):
    """Register new user account."""
    try:
        user_id = await auth_service.create_user(user_data)
        logger.info(f"New user registered: {user_data.username}")
        return {"message": "User created successfully", "user_id": user_id}
        
    except Exception as e:
        logger.warning(f"Registration failed for {user_data.username}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/logout", tags=["Authentication"])
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user and invalidate token."""
    try:
        await auth_service.invalidate_token(credentials.credentials)
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.warning(f"Logout failed: {e}")
        raise HTTPException(status_code=400, detail="Logout failed")

# ============================================================================
# WEBSOCKET ENDPOINTS FOR REAL-TIME MESSAGING
# ============================================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time medical triage chat.
    Handles patient-doctor communication and AI-assisted triage.
    """
    await websocket_manager.connect(websocket, client_id)
    logger.info(f"WebSocket client {client_id} connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # TODO: Process message through medical AI pipeline
            # TODO: Integrate with RAG service for medical knowledge retrieval
            # TODO: Use NLP service for symptom extraction and analysis
            # TODO: Apply ML classifiers for triage priority routing
            # TODO: Check for bias in AI recommendations
            
            processed_message = await process_medical_message(data, client_id)
            
            # Send response back to client
            await websocket_manager.send_personal_message(processed_message, client_id)
            
            # Broadcast urgent alerts to healthcare providers if needed
            if processed_message.get("urgency_level") == "critical":
                await websocket_manager.broadcast_urgent_alert(processed_message)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)

async def process_medical_message(data: Dict, client_id: str) -> Dict:
    """
    Process incoming medical message through AI pipeline.
    TODO: This will integrate with all the medical AI services.
    """
    # Placeholder processing - will be replaced with actual AI pipeline
    data.get("type", "chat")
    content = data.get("content", "")
    
    # TODO: Integrate with services/rag_service.py for medical knowledge retrieval
    # TODO: Integrate with services/nlp_service.py for symptom analysis
    # TODO: Integrate with MLmodels/classifier.py for risk assessment
    # TODO: Integrate with services/image_diagnosis.py for medical images
    
    response = {
        "type": "ai_response",
        "client_id": client_id,
        "content": f"Processed: {content}",
        "timestamp": datetime.utcnow().isoformat(),
        "urgency_level": "low",  # Will be determined by AI
        "confidence_score": 0.8,  # Will be calculated by AI
        "bias_score": 0.05  # Will be calculated by bias detection
    }
    
    return response

# ============================================================================
# HEALTH CHECK AND MONITORING ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Comprehensive health check for all system components."""
    try:
        # Check database health
        db_status = await db_manager.health_check()
        
        # Check authentication service health
        auth_status = await auth_service.health_check()
        
        # Check WebSocket connections
        ws_stats = websocket_manager.get_connection_stats()
        
        # TODO: Check RAG service health
        # TODO: Check NLP service health
        # TODO: Check ML model health
        # TODO: Check Ollama service health
        
        health_data = {
            "status": "healthy",
            "environment": current_environment,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": db_status,
                "authentication": auth_status,
                "websocket": ws_stats
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system metrics for monitoring."""
    # TODO: Implement comprehensive metrics collection
    # TODO: Include bias detection metrics
    # TODO: Include ML model performance metrics
    
    return {
        "active_connections": websocket_manager.active_connections_count(),
        "environment": current_environment,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# API ROUTE INCLUDES
# ============================================================================

# TODO: Include API routes from api/ package
# from api.medical_routes import router as medical_router
# from api.admin_routes import router as admin_router
# app.include_router(medical_router, prefix="/api/medical", tags=["Medical"])
# app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    # Configure uvicorn based on environment
    uvicorn_config = {
        "app": "app:app",
        "host": settings.API_HOST,
        "port": settings.API_PORT,
        "reload": settings.DEBUG and settings.is_development,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": settings.DEBUG
    }
    
    # Production-specific settings
    if settings.is_production:
        uvicorn_config.update({
            "workers": 1,  # Single worker for GPU model sharing
            "reload": False,
            "access_log": False
        })
    
    logger.info(f"🚀 Starting Fairdoc backend on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(**uvicorn_config)
