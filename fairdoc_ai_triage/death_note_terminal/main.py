"""
Death Note Terminal - FastAPI Server

Provides web interface for server management, testing, and AI analysis.
The terminal uses a dark theme inspired by Death Note anime.

Features:
- Server lifecycle management (V1, V2, Combined)
- Test execution (unit, integration, e2e)
- Real-time terminal output via WebSocket
- AI-powered log analysis with Ollama
- Dark/Light theme toggle

File: ./main.py
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from ollama_client import OllamaClient
from server_manager import ServerManager
from terminal_handler import TerminalManager as TerminalHandler
from test_runner import TestRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
server_manager: Optional[ServerManager] = None
terminal_handler: Optional[TerminalHandler] = None
test_runner: Optional[TestRunner] = None
ollama_client: Optional[OllamaClient] = None

# Get current directory
current_dir = Path(__file__).parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global server_manager, terminal_handler, test_runner, ollama_client
    
    logger.info("🔥 Death Note Terminal starting...")
    logger.info(f"📁 Root directory: {str(config.ROOT_DIR)}")  # FIXED: Added str()
    
    # Initialize components
    server_manager = ServerManager()
    terminal_handler = TerminalHandler()
    test_runner = TestRunner()
    ollama_client = OllamaClient()
    
    # Test Ollama connection
    try:
        await ollama_client.test_connection()  # Now this method exists
        logger.info(f"🤖 Connected to Ollama with {config.OLLAMA_MODEL}")
    except Exception as e:
        logger.warning(f"⚠️ Ollama connection failed: {e}")
    
    # Discover tests
    try:
        tests = test_runner.discover_tests()  # Now this method exists
        total_tests = sum(len(tests[category]) for category in tests)
        logger.info(f"🧪 Discovered {total_tests} tests")
    except Exception as e:
        logger.warning(f"⚠️ Test discovery failed: {e}")
    
    logger.info("✅ Death Note Terminal ready - http://localhost:8999")
    
    yield
    
    # Shutdown
    logger.info("🧹 Shutting down Death Note Terminal...")
    
    if server_manager:
        server_manager.stop_all_servers()
    
    if terminal_handler:
        await terminal_handler.cleanup()
    
    logger.info("👋 Death Note Terminal shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="🔥 Death Note Terminal",
    description="Server management and testing interface",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] + ["http://localhost:8999"],  # FIXED: Simplified CORS origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception:
                self.disconnect(client_id)

    async def broadcast(self, message: str):
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

manager = ConnectionManager()

# Pydantic models
class ServerRequest(BaseModel):
    server_type: str
    port: Optional[int] = None

class TestRequest(BaseModel):
    test_types: List[str]
    specific_tests: Optional[List[str]] = None
    pytest_args: Optional[str] = None

class AnalysisRequest(BaseModel):
    content: str
    analysis_type: str = "general"

# FIXED: Individual static file routes
@app.get("/styles.css")
async def get_styles():
    """Serve styles.css"""
    styles_path = current_dir / "styles.css"
    if styles_path.exists():
        return FileResponse(styles_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/app.js")
async def get_app_js():
    """Serve app.js"""
    app_js_path = current_dir / "app.js"
    if app_js_path.exists():
        return FileResponse(app_js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")

@app.get("/terminal.js")
async def get_terminal_js():
    """Serve terminal.js"""
    terminal_js_path = current_dir / "terminal.js"
    if terminal_js_path.exists():
        return FileResponse(terminal_js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="terminal.js not found")

# Log static file availability
if (current_dir / "styles.css").exists():
    logger.info("📄 Static files found and mounted")
else:
    logger.warning("📄 Static files not found")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main Death Note Terminal interface"""
    index_path = current_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Index file not found")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - can be extended for interactive terminal
            await manager.send_message(f"Echo: {data}", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# API Routes
@app.get("/api/servers/status")
async def get_servers_status():
    """Get status of all configured servers"""
    if not server_manager:
        raise HTTPException(status_code=503, detail="Server manager not initialized")
    
    return server_manager.get_all_status()

@app.post("/api/servers/start")
async def start_server(request: ServerRequest):
    """Start a server"""
    if not server_manager:
        raise HTTPException(status_code=503, detail="Server manager not initialized")
    
    try:
        result = server_manager.start_server(request.server_type, request.port)
        await manager.broadcast(json.dumps({
            "type": "server_started",
            "server_type": request.server_type,
            "result": result
        }))
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/servers/stop/{server_type}")
async def stop_server(server_type: str):
    """Stop a server"""
    if not server_manager:
        raise HTTPException(status_code=503, detail="Server manager not initialized")
    
    try:
        result = server_manager.stop_server(server_type)
        await manager.broadcast(json.dumps({
            "type": "server_stopped",
            "server_type": server_type,
            "result": result
        }))
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tests/discover")
async def discover_tests():
    """Discover available tests"""
    if not test_runner:
        raise HTTPException(status_code=503, detail="Test runner not initialized")
    
    try:
        return test_runner.discover_tests()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tests/run")
async def run_tests(request: TestRequest):
    """Run selected tests"""
    if not test_runner:
        raise HTTPException(status_code=503, detail="Test runner not initialized")
    
    try:
        session_id = str(uuid.uuid4())
        
        # Start test execution in background
        asyncio.create_task(
            test_runner.run_tests_async(
                test_types=request.test_types,
                specific_tests=request.specific_tests,
                pytest_args=request.pytest_args,
                session_id=session_id,
                callback=lambda msg: asyncio.create_task(manager.broadcast(msg))
            )
        )
        
        return {"session_id": session_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tests/sessions/{session_id}/output")
async def get_test_output(session_id: str):
    """Get test session output"""
    if not test_runner:
        raise HTTPException(status_code=503, detail="Test runner not initialized")
    
    return test_runner.get_session_output(session_id)

@app.get("/api/tests/sessions")
async def get_test_sessions():
    """Get all test sessions"""
    if not test_runner:
        raise HTTPException(status_code=503, detail="Test runner not initialized")
    
    return test_runner.get_all_sessions()

@app.post("/api/analyze")
async def analyze_content(request: AnalysisRequest):
    """Analyze content with Ollama AI"""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    try:
        result = await ollama_client.analyze(request.content, request.analysis_type)
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ollama/status")
async def get_ollama_status():
    """Check Ollama connection status"""
    if not ollama_client:
        return {"status": "not_initialized"}
    
    try:
        await ollama_client.test_connection()
        return {"status": "connected", "model": config.OLLAMA_MODEL}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}

if __name__ == "__main__":
    print("🔥 Death Note Terminal")
    print("🌐 Interface: http://localhost:8999")
    print("🕷️ Death Note Terminal is starting...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8999,
        reload=True,
        log_level="info"
    )
