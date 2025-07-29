#!/usr/bin/env python3
"""
Fairdoc AI Control Panel - Modern FastAPI Server

Web-based interface for managing V1/V2 servers and running tests.
Uses modern FastAPI lifespan API (no deprecated @app.on_event).

Single responsibility: Development server control and test execution interface
File: ./runserver.py
"""

from __future__ import annotations

import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from run_ctrl import ProcessManager, TestDiscovery, TestRunner, ServerConfig

# ─────────────────────────── Global State ────────────────────────────
ROOT = Path(__file__).parent
process_manager = ProcessManager()
test_discovery = TestDiscovery(ROOT)
test_runner = TestRunner(ROOT)

SERVER_CONFIGS: Dict[str, ServerConfig] = {
    "v1": ServerConfig("src.app.main:app", 8000, "V1 API - Legacy System"),
    "v2": ServerConfig("src.app2.main_v2:app", 8000, "V2 API - Modern Architecture"),
    "both": ServerConfig("src.app.main:app", 8000, "V1 + V2 Mounted Combined"),
}

# ─────────────────────── Modern FastAPI Lifespan ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan management
    Replaces deprecated @app.on_event decorators
    """
    # Enhanced startup sequence
    print("🚀 Fairdoc Control Panel starting...")
    print(f"📁 Project root: {ROOT}")
    print(f"🔧 Available servers: {list(SERVER_CONFIGS.keys())}")
    
    # Test discovery with detailed breakdown
    try:
        all_tests = test_discovery.discover_all_tests()
        total_tests = len(all_tests.get('all', []))
        print(f"🔍 Test discovery complete: {total_tests} tests found")
        print(f"📊 Breakdown: Unit={len(all_tests.get('unit', []))}, "
              f"Integration={len(all_tests.get('integration', []))}, "
              f"E2E={len(all_tests.get('e2e', []))}")
    except Exception as e:
        print(f"⚠️ Test discovery warning: {e}")
    
    print("✅ Control Panel ready - http://localhost:8999")
    
    try:
        yield  # ── Application is now running ──
    finally:
        # Enhanced shutdown sequence
        print("🧹 Shutting down Fairdoc Control Panel...")
        
        # Stop all running servers
        stopped_servers = process_manager.stop_all_servers()
        if stopped_servers:
            print(f"🛑 Stopped {len(stopped_servers)} server(s): {', '.join(stopped_servers)}")
        
        # Clean up test runners
        active_runs = test_runner.list_runs()
        if active_runs:
            print(f"🧪 Cleaned up {len(active_runs)} test run(s)")
        
        print("👋 Control Panel shutdown complete")

# ─────────────────────── FastAPI Application ─────────────────────

app = FastAPI(
    title="Fairdoc AI Control Panel",
    description="Modern web interface for Fairdoc AI development, testing, and server management",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if __name__ == "__main__" else None,  # Only in dev mode
    redoc_url="/redoc" if __name__ == "__main__" else None,
)

# ────────────────────────── Middleware Stack ───────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ───────────────────────── Request/Response Models ───────────────────────

class ServerRequest(BaseModel):
    """Server management request model"""
    server: str = Field(..., description="Server to manage: v1, v2, both, or stop_all")

class TestRunRequest(BaseModel):
    """Test execution request model"""
    test_files: List[str] = Field(..., description="List of test files to execute")
    live_mode: bool = Field(False, description="Run against live server")

class ServerStatus(BaseModel):
    """Server status response model"""
    running: bool
    port: int
    description: str
    pid: Optional[int] = None

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    message: str
    data: Any = None
    timestamp: float = Field(default_factory=time.time)

# ────────────────────────── HTML Frontend ────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_control_panel() -> str:
    """Serve the main HTML5 control panel interface"""
    ui_file = ROOT / "controller_ui.html"
    
    if not ui_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Control panel UI not found. Ensure controller_ui.html exists."
        )
    
    return ui_file.read_text(encoding="utf-8")

# ──────────────────── Server Management Endpoints ────────────────────

@app.get("/api/status", response_model=Dict[str, Dict[str, Any]], tags=["server"])
async def get_server_status():
    """Get status of all configured servers"""
    servers_status = {}
    
    for name, config in SERVER_CONFIGS.items():
        if name != "both":  # Skip composite server in status
            servers_status[name] = {
                "running": process_manager.is_running(name),
                "port": config.port,
                "description": config.description,
                "pid": process_manager.get_pid(name) if process_manager.is_running(name) else None
            }
    
    return {
        "servers": servers_status,
        "timestamp": time.time(),
        "control_panel_version": "2.0.0"
    }

@app.post("/api/server", response_model=ApiResponse, tags=["server"])
async def manage_server(request: ServerRequest):
    """Start, stop, or restart servers"""
    server_name = request.server
    
    # Handle stop all servers
    if server_name == "stop_all":
        stopped = process_manager.stop_all_servers()
        return ApiResponse(
            success=True,
            message=f"Stopped {len(stopped)} server(s)",
            data={"stopped_servers": stopped}
        )
    
    # Validate server name
    if server_name not in SERVER_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown server '{server_name}'. Available: {list(SERVER_CONFIGS.keys())}"
        )
    
    # Stop all servers before starting new one
    process_manager.stop_all_servers()
    
    # Start requested server
    config = SERVER_CONFIGS[server_name]
    success = process_manager.start_server(server_name, config)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start server '{server_name}'"
        )
    
    return ApiResponse(
        success=True,
        message=f"Server '{server_name}' started successfully",
        data={"server": server_name, "port": config.port}
    )

# ────────────────────── Test Management Endpoints ─────────────────────

@app.get("/api/tests", tags=["testing"])
async def discover_tests():
    """Discover and categorize all available tests"""
    try:
        tests = test_discovery.discover_all_tests()
        return {
            "tests": tests,
            "summary": {
                "total": len(tests.get('all', [])),
                "unit": len(tests.get('unit', [])),
                "integration": len(tests.get('integration', [])),
                "e2e": len(tests.get('e2e', []))
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test discovery failed: {str(e)}"
        )

@app.post("/api/run_tests", response_model=ApiResponse, tags=["testing"])
async def execute_tests(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Execute selected tests in background"""
    if not request.test_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No test files specified"
        )
    
    # Create unique run ID
    run_id = test_runner.create_run_id()
    
    # Start test execution in background
    background_tasks.add_task(
        test_runner.execute_tests,
        run_id,
        request.test_files,
        request.live_mode
    )
    
    return ApiResponse(
        success=True,
        message=f"Test execution started with {len(request.test_files)} test(s)",
        data={
            "run_id": run_id,
            "test_count": len(request.test_files),
            "live_mode": request.live_mode
        }
    )

@app.get("/api/test_output/{run_id}", tags=["testing"])
async def get_test_output(run_id: str):
    """Get real-time test execution output"""
    return {
        "run_id": run_id,
        "status": test_runner.get_status(run_id),
        "output": test_runner.get_output(run_id),
        "timestamp": time.time()
    }

@app.get("/api/test_runs", tags=["testing"])
async def list_test_runs():
    """List all test runs (active and completed)"""
    return {
        "runs": test_runner.list_runs(),
        "timestamp": time.time()
    }

# ─────────────────────── Health & System Info ──────────────────────

@app.get("/api/health", tags=["system"])
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "uptime": time.time(),
        "servers_available": list(SERVER_CONFIGS.keys()),
        "timestamp": time.time()
    }

@app.get("/api/info", tags=["system"])
async def system_info():
    """Get system information and capabilities"""
    return {
        "name": "Fairdoc AI Control Panel",
        "version": "2.0.0",
        "root_directory": str(ROOT),
        "servers": {name: config.description for name, config in SERVER_CONFIGS.items()},
        "test_categories": ["unit", "integration", "e2e"],
        "features": [
            "Server Management",
            "Test Execution",
            "Real-time Output",
            "Background Processing"
        ],
        "endpoints": {
            "frontend": "/",
            "server_status": "/api/status",
            "test_discovery": "/api/tests",
            "health": "/api/health"
        }
    }

# ────────────────────────── Development Server ─────────────────────────

if __name__ == "__main__":
    print("\n🌐 Fairdoc Control Panel")
    print("🔗 Web Interface: http://localhost:8999")
    print("📚 API Docs: http://localhost:8999/docs")
    print("🔧 Starting development server...\n")
    
    uvicorn.run(
        "runserver:app",
        host="0.0.0.0",
        port=8999,
        reload=True,
        log_level="info",
        access_log=True
    )
