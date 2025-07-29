"""
Fairdoc AI Control Panel - FastAPI Server

Web-based interface for managing V1/V2 servers and running tests
Serves HTML5 UI and provides REST API endpoints

File: ./runserver.py
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from run_ctrl import (
    ProcessManager,
    TestDiscovery,
    TestRunner,
    ServerConfig
)

# Initialize FastAPI app
app = FastAPI(
    title="Fairdoc AI Control Panel",
    description="Web interface for Fairdoc AI development and testing",
    version="1.0.0"
)

# Global state
ROOT = Path(__file__).parent
process_manager = ProcessManager()
test_discovery = TestDiscovery(ROOT)
test_runner = TestRunner(ROOT)

# Server configurations
SERVER_CONFIGS = {
    "v1": ServerConfig("src.app.main:app", 8000, "V1 API"),
    "v2": ServerConfig("src.app2.main_v2:app", 8000, "V2 API"),
    "both": ServerConfig("src.app.main:app", 8000, "V1+V2 Combined")
}

# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class ServerRequest(BaseModel):
    server: str  # v1, v2, both, stop_all

class TestRunRequest(BaseModel):
    test_files: list[str]
    live_mode: bool = False

# ---------------------------------------------------------------------------
# HTML UI Endpoint
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main HTML5 control panel interface"""
    ui_file = ROOT / "controller_ui.html"
    if ui_file.exists():
        return ui_file.read_text()
    return "<h1>UI file not found. Please create controller_ui.html</h1>"

# ---------------------------------------------------------------------------
# Server Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Get current status of all servers with port/description"""
    server_status = {}
    for name, config in SERVER_CONFIGS.items():
        if name == "both":
            continue
        server_status[name] = {
            "running": process_manager.is_running(name),
            "port": config.port,
            "description": config.description
        }
    return {
        "servers": server_status,
        "timestamp": time.time()
    }

@app.post("/api/server")
async def manage_server(request: ServerRequest):
    """Start, stop, or restart servers"""
    server_name = request.server
    
    if server_name == "stop_all":
        stopped = process_manager.stop_all_servers()
        return {
            "action": "stop_all",
            "stopped_servers": stopped,
            "message": f"Stopped {len(stopped)} servers"
        }
    
    if server_name not in SERVER_CONFIGS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown server: {server_name}"}
        )
    
    config = SERVER_CONFIGS[server_name]
    
    # Stop any running servers first
    process_manager.stop_all_servers()
    
    # Start the requested server
    success = process_manager.start_server(server_name, config)
    
    if success:
        return {
            "action": "start",
            "server": server_name,
            "message": f"{config.description} started on port {config.port}"
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start {server_name}"}
        )

# ---------------------------------------------------------------------------
# Test Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/tests")
async def list_tests():
    """List all available test files organized by type"""
    return test_discovery.discover_all_tests()

@app.post("/api/run_tests")
async def run_tests(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Execute selected tests in background"""
    if not request.test_files:
        return JSONResponse(
            status_code=400,
            content={"error": "No test files specified"}
        )
    
    run_id = test_runner.create_run_id()
    
    # Start test execution in background
    background_tasks.add_task(
        test_runner.execute_tests,
        run_id,
        request.test_files,
        request.live_mode
    )
    
    return {
        "run_id": run_id,
        "test_count": len(request.test_files),
        "live_mode": request.live_mode,
        "message": "Tests started"
    }

@app.get("/api/test_output/{run_id}")
async def get_test_output(run_id: str):
    """Get test execution output and status"""
    output = test_runner.get_output(run_id)
    status = test_runner.get_status(run_id)
    
    return {
        "run_id": run_id,
        "status": status,
        "output": output,
        "timestamp": time.time()
    }

@app.get("/api/test_runs")
async def list_test_runs():
    """List all recent test runs"""
    return {
        "runs": test_runner.list_runs(),
        "timestamp": time.time()
    }

# ---------------------------------------------------------------------------
# Health and Info Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fairdoc Control Panel",
        "version": "1.0.0",
        "uptime": time.time()
    }

@app.get("/api/info")
async def get_info():
    """Get system information"""
    return {
        "available_servers": list(SERVER_CONFIGS.keys()),
        "test_types": ["unit", "integration", "e2e"],
        "project_root": str(ROOT),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "features": {
            "server_management": True,
            "test_execution": True,
            "live_mode": True,
            "background_tasks": True
        }
    }

# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting Fairdoc Control Panel...")
    print(f"📁 Project root: {ROOT}")
    print(f"🔍 Available tests: {len(test_discovery.discover_all_tests().get('all', []))}")
    print("✅ Control panel ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🧹 Shutting down control panel...")
    stopped = process_manager.stop_all_servers()
    if stopped:
        print(f"🛑 Stopped {len(stopped)} running servers")
    print("👋 Control panel shutdown complete")

# ---------------------------------------------------------------------------
# Development Server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("  Fairdoc AI Control Panel - Web Interface")
    print("=" * 60)
    print("  🌐 http://localhost:8999")
    print("  📖 API docs: http://localhost:8999/docs")
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        "runserver:app",
        host="0.0.0.0",
        port=8999,
        reload=True,
        log_level="info"
    )
