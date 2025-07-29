"""
Death Note Terminal - Main FastAPI Application

Modern FastAPI server with WebSocket support for real-time terminal updates.
Uses lifespan management (no deprecated @app.on_event decorators).

File: main.py
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import config
from server_manager import server_manager
from test_runner import test_runner, test_discovery
from ollama_client import ollama_client
import json
import time

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Connection management for WebSockets
class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

# Global connection manager
manager = ConnectionManager()

# Request/Response Models
class ServerRequest(BaseModel):
    server_type: str = Field(..., description="Server type: v1_only, v2_only, combined")
    custom_port: Optional[int] = Field(None, description="Optional custom port")

class TestExecutionRequest(BaseModel):
    test_files: List[str] = Field(..., description="List of test files to execute")
    pytest_args: Optional[List[str]] = Field(None, description="Custom pytest arguments")

class AnalysisRequest(BaseModel):
    content: str = Field(..., description="Log content to analyze")
    analysis_type: str = Field("general_summary", description="Type of analysis")

# FastAPI Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan management"""
    
    # Startup
    logger.info("🔥 Death Note Terminal starting...")
    logger.info(f"📁 Root directory: {config.ROOT_DIR}")
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        logger.warning(f"Configuration issues found: {issues}")
    
    # Check Ollama connection
    ollama_connected, ollama_msg = await ollama_client.check_connection()
    if ollama_connected:
        logger.info(f"🤖 {ollama_msg}")
    else:
        logger.warning(f"⚠️  Ollama: {ollama_msg}")
    
    # Discover tests
    try:
        tests = test_discovery.discover_all_tests()
        test_count = len(tests.get('all', []))
        logger.info(f"🧪 Discovered {test_count} tests")
    except Exception as e:
        logger.error(f"Test discovery failed: {e}")
    
    logger.info("✅ Death Note Terminal ready - http://localhost:8999")
    
    try:
        yield  # Application running
    finally:
        # Shutdown
        logger.info("🧹 Shutting down Death Note Terminal...")
        
        # Stop all servers
        stopped_servers = server_manager.stop_all_servers()
        if stopped_servers:
            logger.info(f"🛑 Stopped servers: {', '.join(stopped_servers)}")
        
        # Cancel running tests
        for session_id in list(test_runner.sessions.keys()):
            test_runner.cancel_session(session_id)
        
        logger.info("👋 Death Note Terminal shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description="Death Note themed terminal for Fairdoc AI development",
    version=config.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS + ["http://localhost:8999"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (for CSS, JS, images)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    logger.warning("Static directory not found - static files unavailable")

# HTML Routes
@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_main_page():
    """Serve the main Death Note terminal interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Main interface not found. Ensure index.html exists."
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket connection for real-time terminal updates"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, client_id)
            elif message.get("type") == "subscribe":
                # Subscribe to specific events (server logs, test output, etc.)
                pass
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Server Management API
@app.get("/api/servers/status", tags=["servers"])
async def get_server_status():
    """Get status of all configured servers"""
    status_data = server_manager.get_all_status()
    available_ports = server_manager.get_available_ports()
    
    return {
        "servers": status_data,
        "available_ports": available_ports,
        "timestamp": time.time()
    }

@app.post("/api/servers/start", tags=["servers"])
async def start_server(request: ServerRequest, background_tasks: BackgroundTasks):
    """Start a server with optional custom port"""
    
    success, message = server_manager.start_server(request.server_type, request.custom_port)
    
    if success:
        # Broadcast server status update
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "server_started",
                "server": request.server_type,
                "port": request.custom_port or config.get_server_config(request.server_type).get("port"),
                "message": message
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "message": message}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )

@app.post("/api/servers/stop/{server_type}", tags=["servers"])
async def stop_server(server_type: str, background_tasks: BackgroundTasks):
    """Stop a specific server"""
    
    success, message = server_manager.stop_server(server_type)
    
    if success:
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "server_stopped",
                "server": server_type,
                "message": message
            }
        )
        
        return {"success": True, "message": message}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )

# Test Management API
@app.get("/api/tests/discover", tags=["testing"])
async def discover_tests():
    """Discover all available tests"""
    try:
        tests = test_discovery.discover_all_tests()
        return {
            "tests": tests,
            "summary": {
                "total": len(tests.get('all', [])),
                "unit": len(tests.get('unit', [])),
                "integration": len(tests.get('integration', [])),
                "e2e": len(tests.get('e2e', []))
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test discovery failed: {str(e)}"
        )

@app.post("/api/tests/run", tags=["testing"])
async def run_tests(request: TestExecutionRequest, background_tasks: BackgroundTasks):
    """Execute selected tests"""
    
    if not request.test_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No test files specified"
        )
    
    # Create test session
    session_id = test_runner.create_session(request.test_files, request.pytest_args)
    
    # Start test execution in background
    background_tasks.add_task(run_test_session, session_id)
    
    return {
        "session_id": session_id,
        "test_count": len(request.test_files),
        "message": "Test execution started"
    }

async def run_test_session(session_id: str):
    """Background task to run tests and broadcast updates"""
    
    # Notify start
    await manager.broadcast({
        "type": "test_started",
        "session_id": session_id
    })
    
    # Run tests
    success = await test_runner.run_tests(session_id)
    
    # Notify completion
    session = test_runner.get_session(session_id)
    if session:
        await manager.broadcast({
            "type": "test_completed",
            "session_id": session_id,
            "status": session.status,
            "duration": session.duration,
            "return_code": session.return_code
        })

@app.get("/api/tests/sessions/{session_id}/output", tags=["testing"])
async def get_test_output(session_id: str, from_line: int = 0):
    """Get test session output"""
    
    output_lines, is_complete = test_runner.get_session_output(session_id, from_line)
    
    return {
        "session_id": session_id,
        "output": output_lines,
        "from_line": from_line,
        "is_complete": is_complete,
        "timestamp": time.time()
    }

@app.get("/api/tests/sessions", tags=["testing"])
async def list_test_sessions():
    """List all test sessions"""
    return {
        "sessions": test_runner.list_sessions(),
        "timestamp": time.time()
    }

# Ollama Integration API
@app.post("/api/analyze", tags=["ai"])
async def analyze_content(request: AnalysisRequest):
    """Analyze content with Ollama AI"""
    
    try:
        result = ollama_client.sync_analyze_logs(request.content, request.analysis_type)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/ollama/status", tags=["ai"])
async def ollama_status():
    """Check Ollama connection status"""
    connected, message = await ollama_client.check_connection()
    return {
        "connected": connected,
        "message": message,
        "model": config.OLLAMA_MODEL,
        "base_url": config.OLLAMA_BASE_URL
    }

# System Health API
@app.get("/api/health", tags=["system"])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": config.VERSION,
        "active_connections": len(manager.active_connections),
        "active_test_sessions": len([s for s in test_runner.sessions.values() if s.is_running]),
        "running_servers": len([s for s in server_manager.processes.values() if s.is_running])
    }

# Development server
if __name__ == "__main__":
    print(f"\n🔥 {config.APP_NAME}")
    print(f"🌐 Interface: http://localhost:{config.PORT}")
    print("🕷️ Death Note Terminal is starting...\n")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )