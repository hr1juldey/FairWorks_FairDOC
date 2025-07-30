"""
Test Runner - Test Discovery and Execution

Manages test discovery, execution, and result collection
for unit, integration, and e2e tests.

Single responsibility: Test management and execution
File: ./test_runner.py
"""

import asyncio
import glob
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import uuid

import config

logger = logging.getLogger(__name__)

class TestRunner:
    """Manages test discovery and execution"""
    
    def __init__(self):
        self.root_dir = config.ROOT_DIR
        self.test_sessions: Dict[str, Dict[str, Any]] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        
    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover available tests"""
        try:
            return {
                "unit": self._discover_unit_tests(),
                "integration": self._discover_integration_tests(), 
                "e2e": self._discover_e2e_tests()
            }
        except Exception as e:
            logger.error(f"❌ Test discovery failed: {e}")
            return {"unit": [], "integration": [], "e2e": []}

    def _discover_unit_tests(self) -> List[str]:
        """Helper to find unit tests"""
        pattern = str(self.root_dir / "src" / "tests" / "unit" / "test_*.py")
        tests = glob.glob(pattern)
        return [Path(test).name for test in tests]

    def _discover_integration_tests(self) -> List[str]:
        """Helper to find integration tests"""  
        pattern = str(self.root_dir / "src" / "tests" / "integration" / "test_*.py")
        tests = glob.glob(pattern)
        return [Path(test).name for test in tests]

    def _discover_e2e_tests(self) -> List[str]:
        """Helper to find e2e tests"""
        pattern = str(self.root_dir / "src" / "tests" / "e2e" / "test_*.py")
        tests = glob.glob(pattern)
        return [Path(test).name for test in tests]
    
    def create_session(self) -> str:
        """Create a new test session"""
        session_id = str(uuid.uuid4())
        self.test_sessions[session_id] = {
            "id": session_id,
            "status": "created",
            "start_time": time.time(),
            "end_time": None,
            "output": [],
            "results": {},
            "error": None
        }
        return session_id
    
    def run_tests_sync(
        self,
        test_types: List[str],
        specific_tests: Optional[List[str]] = None,
        pytest_args: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run tests synchronously"""
        
        if not session_id:
            session_id = self.create_session()
        
        session = self.test_sessions[session_id]
        session["status"] = "running"
        
        try:
            # Build pytest command
            cmd = ["python", "-m", "pytest", "-v"]
            
            # Add test paths
            for test_type in test_types:
                test_path = self.root_dir / "src" / "tests" / test_type
                if test_path.exists():
                    if specific_tests:
                        for test in specific_tests:
                            specific_path = test_path / test
                            if specific_path.exists():
                                cmd.append(str(specific_path))
                    else:
                        cmd.append(str(test_path))
            
            # Add custom pytest args
            if pytest_args:
                cmd.extend(pytest_args.split())
            
            # Add output formatting
            cmd.extend(["--tb=short", "--no-header"])
            
            session["command"] = " ".join(cmd)
            logger.info(f"🧪 Running tests: {session['command']}")
            
            # Execute tests
            process = subprocess.Popen(
                cmd,
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.active_processes[session_id] = process
            
            # Collect output
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    session["output"].append({
                        "timestamp": time.time(),
                        "line": line
                    })
            
            # Wait for completion
            return_code = process.wait()
            
            # Parse results
            session["results"] = self._parse_pytest_output(output_lines)
            session["status"] = "completed" if return_code == 0 else "failed"
            session["return_code"] = return_code
            session["end_time"] = time.time()
            
            # Clean up
            if session_id in self.active_processes:
                del self.active_processes[session_id]
            
            logger.info(f"✅ Tests completed: {session['status']}")
            return session
            
        except Exception as e:
            session["status"] = "error"
            session["error"] = str(e)
            session["end_time"] = time.time()
            logger.error(f"❌ Test execution failed: {e}")
            return session
    
    async def run_tests_async(
        self,
        test_types: List[str],
        specific_tests: Optional[List[str]] = None,
        pytest_args: Optional[str] = None,
        session_id: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run tests asynchronously"""
        
        def run_in_thread():
            result = self.run_tests_sync(test_types, specific_tests, pytest_args, session_id)
            if callback:
                asyncio.create_task(callback(json.dumps({
                    "type": "test_completed",
                    "session_id": session_id,
                    "result": result
                })))
            return result
        
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_in_thread)
    
    def _parse_pytest_output(self, output_lines: List[str]) -> Dict[str, Any]:
        """Parse pytest output to extract results"""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "failed_tests": [],
            "duration": 0
        }
        
        for line in output_lines:
            # Parse summary line (e.g., "5 passed, 2 failed in 10.23s")
            if " passed" in line or " failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            status = parts[i + 1]
                            if "passed" in status:
                                results["passed"] = count
                            elif "failed" in status:
                                results["failed"] = count
                            elif "skipped" in status:
                                results["skipped"] = count
                            elif "error" in status:
                                results["errors"] = count
            
            # Parse duration
            if " in " in line and "s" in line:
                try:
                    duration_str = line.split(" in ")[-1].replace("s", "")
                    results["duration"] = float(duration_str)
                except Exception:
                    pass
            
            # Collect failed tests
            if "FAILED" in line:
                test_name = line.split("FAILED")[0].strip()
                results["failed_tests"].append(test_name)
        
        results["total"] = results["passed"] + results["failed"] + results["skipped"] + results["errors"]
        return results
    
    def get_session_output(self, session_id: str) -> Dict[str, Any]:
        """Get output for a test session"""
        if session_id not in self.test_sessions:
            return {"error": "Session not found"}
        
        return self.test_sessions[session_id]
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all test sessions"""
        return list(self.test_sessions.values())
    
    def stop_session(self, session_id: str) -> bool:
        """Stop a running test session"""
        if session_id in self.active_processes:
            try:
                process = self.active_processes[session_id]
                process.terminate()
                process.wait(timeout=5)
                return True
            except Exception:
                try:
                    process.kill()
                    return True
                except Exception:
                    return False
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old test sessions"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for session_id, session in self.test_sessions.items():
            session_age = current_time - session["start_time"]
            if session_age > max_age_seconds and session["status"] in ["completed", "failed", "error"]:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.test_sessions[session_id]
            logger.info(f"🧹 Cleaned up old test session: {session_id}")
    
    def get_test_stats(self) -> Dict[str, Any]:
        """Get overall test statistics"""
        total_sessions = len(self.test_sessions)
        completed = sum(1 for s in self.test_sessions.values() if s["status"] == "completed")
        failed = sum(1 for s in self.test_sessions.values() if s["status"] == "failed")
        running = sum(1 for s in self.test_sessions.values() if s["status"] == "running")
        
        return {
            "total_sessions": total_sessions,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": (completed / total_sessions * 100) if total_sessions > 0 else 0
        }
