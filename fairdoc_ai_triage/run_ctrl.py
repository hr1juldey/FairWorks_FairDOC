"""
Fairdoc AI Control Helpers

Process management, test discovery, and execution utilities
for the web-based control panel

File: ./run_ctrl.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Configuration for a server instance"""
    module: str
    port: int
    description: str
    reload: bool = True

@dataclass
class TestRun:
    """Information about a test execution"""
    run_id: str
    test_files: List[str]
    live_mode: bool
    status: str  # running, completed, failed
    output: str
    start_time: float
    end_time: Optional[float] = None

# ---------------------------------------------------------------------------
# Process Management
# ---------------------------------------------------------------------------

class ProcessManager:
    """Manages server processes (uvicorn instances)"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
    
    def start_server(self, name: str, config: ServerConfig) -> bool:
        """Start a server process"""
        try:
            # Build uvicorn command
            cmd = [
                sys.executable, "-m", "uvicorn",
                config.module,
                "--host", "0.0.0.0",
                "--port", str(config.port)
            ]
            
            if config.reload:
                cmd.append("--reload")
            
            # Start process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[name] = proc
            print(f"🚀 Started {name}: {config.description} (PID: {proc.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def stop_server(self, name: str) -> bool:
        """Stop a specific server"""
        if name not in self.processes:
            return False
        
        proc = self.processes[name]
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"🛑 Stopped {name}")
            del self.processes[name]
            return True
        return False
    
    def stop_all_servers(self) -> List[str]:
        """Stop all running servers"""
        stopped = []
        for name in list(self.processes.keys()):
            if self.stop_server(name):
                stopped.append(name)
        return stopped
    
    def is_running(self, name: str) -> bool:
        """Check if a server is running"""
        if name not in self.processes:
            return False
        proc = self.processes[name]
        return proc is not None and proc.poll() is None
    
    def get_server_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all servers"""
        info = {}
        for name, proc in self.processes.items():
            if proc:
                info[name] = {
                    "pid": proc.pid,
                    "running": proc.poll() is None,
                    "returncode": proc.returncode
                }
        return info

# ---------------------------------------------------------------------------
# Test Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    """Discovers and categorizes test files"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.tests_path = root_path / "src" / "tests"
        self.test_types = ["unit", "integration", "e2e"]
    
    def discover_all_tests(self) -> Dict[str, List[Dict[str, str]]]:
        """Discover all test files organized by type"""
        all_tests = {"all": []}
        
        for test_type in self.test_types:
            type_tests = self._discover_tests_by_type(test_type)
            all_tests[test_type] = type_tests
            all_tests["all"].extend(type_tests)
        
        return all_tests
    
    def _discover_tests_by_type(self, test_type: str) -> List[Dict[str, str]]:
        """Discover tests for a specific type"""
        type_path = self.tests_path / test_type
        if not type_path.exists():
            return []
        
        tests = []
        for test_file in sorted(type_path.glob("test_*.py")):
            tests.append({
                "name": test_file.name,
                "path": str(test_file.relative_to(self.root)),
                "full_path": str(test_file),
                "type": test_type,
                "size": test_file.stat().st_size if test_file.exists() else 0
            })
        
        return tests
    
    def get_test_info(self, test_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific test"""
        full_path = self.root / test_path
        if not full_path.exists():
            return None
        
        return {
            "path": test_path,
            "exists": True,
            "size": full_path.stat().st_size,
            "modified": full_path.stat().st_mtime,
            "type": self._get_test_type(test_path)
        }
    
    def _get_test_type(self, test_path: str) -> str:
        """Determine test type from path"""
        for test_type in self.test_types:
            if f"/{test_type}/" in test_path:
                return test_type
        return "unknown"

# ---------------------------------------------------------------------------
# Test Execution
# ---------------------------------------------------------------------------

class TestRunner:
    """Executes tests and manages results"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.runs: Dict[str, TestRun] = {}
        self.max_runs = 50  # Keep last 50 runs
    
    def create_run_id(self) -> str:
        """Create a unique run ID"""
        return f"run_{int(time.time())}_{len(self.runs)}"
    
    def execute_tests(self, run_id: str, test_files: List[str], live_mode: bool):
        """Execute tests in background"""
        # Create test run record
        test_run = TestRun(
            run_id=run_id,
            test_files=test_files,
            live_mode=live_mode,
            status="running",
            output="",
            start_time=time.time()
        )
        self.runs[run_id] = test_run
        
        try:
            # Set environment
            env = os.environ.copy()
            env["FAIRDOC_AI_LIVE"] = "true" if live_mode else "false"
            
            # Build pytest command
            full_paths = [str(self.root / path) for path in test_files]
            cmd = [sys.executable, "-m", "pytest"] + full_paths + [
                "-v", "--tb=short", "--no-header"
            ]
            
            # Execute tests
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(self.root)
            )
            
            # Capture output
            output_lines = []
            for line in iter(proc.stdout.readline, ''):
                if line:
                    output_lines.append(line)
                    test_run.output = ''.join(output_lines)
            
            proc.stdout.close()
            return_code = proc.wait()
            
            # Update run status
            test_run.status = "completed" if return_code == 0 else "failed"
            test_run.end_time = time.time()
            
            print(f"✅ Test run {run_id} completed with return code {return_code}")
            
        except Exception as e:
            test_run.status = "failed"
            test_run.output += f"\n\nError: {str(e)}"
            test_run.end_time = time.time()
            print(f"❌ Test run {run_id} failed: {e}")
        
        # Cleanup old runs
        self._cleanup_old_runs()
    
    def get_output(self, run_id: str) -> str:
        """Get output for a test run"""
        if run_id in self.runs:
            return self.runs[run_id].output
        return f"Run {run_id} not found"
    
    def get_status(self, run_id: str) -> str:
        """Get status for a test run"""
        if run_id in self.runs:
            return self.runs[run_id].status
        return "not_found"
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all test runs with summary info"""
        runs = []
        for run_id, test_run in sorted(self.runs.items(), 
                                      key=lambda x: x[1].start_time, 
                                      reverse=True):
            runs.append({
                "run_id": run_id,
                "test_count": len(test_run.test_files),
                "status": test_run.status,
                "live_mode": test_run.live_mode,
                "start_time": test_run.start_time,
                "end_time": test_run.end_time,
                "duration": (test_run.end_time - test_run.start_time) 
                           if test_run.end_time else None
            })
        return runs
    
    def _cleanup_old_runs(self):
        """Remove old test runs to prevent memory buildup"""
        if len(self.runs) > self.max_runs:
            # Keep most recent runs
            sorted_runs = sorted(self.runs.items(), 
                               key=lambda x: x[1].start_time, 
                               reverse=True)
            
            # Remove oldest runs
            for run_id, _ in sorted_runs[self.max_runs:]:
                del self.runs[run_id]
