"""
Death Note Terminal - Test Discovery & Execution

Scans test folders and executes pytest with configurable parameters.
Single responsibility: Test management and execution.

File: test_runner.py
"""

import subprocess
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from pathlib import Path
from config import config
import json
import uuid

logger = logging.getLogger(__name__)

class TestSession:
    """Represents a running test session"""
    
    def __init__(self, session_id: str, test_files: List[str], pytest_args: List[str]):
        self.session_id = session_id
        self.test_files = test_files
        self.pytest_args = pytest_args
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status = "running"  # running, completed, failed, cancelled
        self.output_lines: List[str] = []
        self.process: Optional[subprocess.Popen] = None
        self.return_code: Optional[int] = None
    
    @property
    def duration(self) -> float:
        """Get test duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def is_running(self) -> bool:
        """Check if test is still running"""
        return self.status == "running" and self.process and self.process.poll() is None
    
    def add_output(self, line: str):
        """Add output line with length limit"""
        self.output_lines.append(line)
        if len(self.output_lines) > config.MAX_TEST_OUTPUT_LINES:
            self.output_lines.pop(0)  # Remove oldest line
    
    def get_summary(self) -> Dict:
        """Get test session summary"""
        return {
            "session_id": self.session_id,
            "test_files": self.test_files,
            "pytest_args": self.pytest_args,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "return_code": self.return_code,
            "output_lines": len(self.output_lines),
            "test_count": len(self.test_files)
        }

class TestDiscovery:
    """Test file discovery and categorization"""
    
    def __init__(self):
        self.tests_dir = config.TESTS_DIR
        self.categories = config.TEST_CATEGORIES
    
    def discover_all_tests(self) -> Dict[str, List[Dict]]:
        """Discover all test files organized by category"""
        discovered = {"all": []}
        
        for category in self.categories:
            category_tests = self.discover_category_tests(category)
            discovered[category] = category_tests
            discovered["all"].extend(category_tests)
        
        return discovered
    
    def discover_category_tests(self, category: str) -> List[Dict]:
        """Discover tests in specific category folder"""
        category_path = config.get_test_path(category)
        
        if not category_path.exists():
            logger.warning(f"Test category path does not exist: {category_path}")
            return []
        
        tests = []
        for test_file in sorted(category_path.glob("test_*.py")):
            test_info = {
                "name": test_file.name,
                "path": str(test_file.relative_to(config.ROOT_DIR)),
                "full_path": str(test_file),
                "category": category,
                "size": test_file.stat().st_size,
                "modified": test_file.stat().st_mtime,
                "functions": self.extract_test_functions(test_file)
            }
            tests.append(test_info)
        
        return tests
    
    def extract_test_functions(self, test_file: Path) -> List[str]:
        """Extract test function names from file"""
        functions = []
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple regex to find test functions
            import re
            pattern = r'^def (test_\w+)\s*\('
            matches = re.findall(pattern, content, re.MULTILINE)
            functions = matches
            
        except Exception as e:
            logger.warning(f"Could not extract functions from {test_file}: {e}")
        
        return functions
    
    def get_test_file_info(self, test_path: str) -> Optional[Dict]:
        """Get detailed info for specific test file"""
        full_path = config.ROOT_DIR / test_path
        
        if not full_path.exists():
            return None
        
        # Determine category
        category = "unknown"
        for cat in self.categories:
            if f"/{cat}/" in test_path:
                category = cat
                break
        
        return {
            "name": full_path.name,
            "path": test_path,
            "full_path": str(full_path),
            "category": category,
            "size": full_path.stat().st_size,
            "modified": full_path.stat().st_mtime,
            "functions": self.extract_test_functions(full_path),
            "exists": True
        }

class TestRunner:
    """Pytest execution and management"""
    
    def __init__(self):
        self.sessions: Dict[str, TestSession] = {}
        self.discovery = TestDiscovery()
        self.max_sessions = 10  # Limit concurrent test sessions
    
    def create_session(self, test_files: List[str], pytest_args: Optional[List[str]] = None) -> str:
        """Create new test session"""
        session_id = str(uuid.uuid4())[:8]
        
        # Use default args if none provided
        if pytest_args is None:
            pytest_args = config.PYTEST_DEFAULT_ARGS.copy()
        
        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            self.cleanup_old_sessions()
        
        session = TestSession(session_id, test_files, pytest_args)
        self.sessions[session_id] = session
        
        return session_id
    
    async def run_tests(self, session_id: str) -> bool:
        """Run tests for a session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            # Build pytest command
            cmd = ["python", "-m", "pytest"] + session.pytest_args + session.test_files
            
            # Start process
            session.process = subprocess.Popen(
                cmd,
                cwd=str(config.ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            async for line in self.stream_process_output(session.process):
                session.add_output(line.rstrip())
            
            # Wait for completion
            session.return_code = session.process.wait()
            session.end_time = time.time()
            
            # Update status
            if session.return_code == 0:
                session.status = "completed"
            else:
                session.status = "failed"
            
            logger.info(f"Test session {session_id} finished with code {session.return_code}")
            return True
            
        except Exception as e:
            session.status = "failed"
            session.end_time = time.time()
            session.add_output(f"ERROR: {str(e)}")
            logger.error(f"Test session {session_id} failed: {e}")
            return False
    
    async def stream_process_output(self, process: subprocess.Popen) -> AsyncGenerator[str, None]:
        """Stream process output line by line"""
        while True:
            line = process.stdout.readline()
            if not line:
                break
            yield line
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel running test session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if session.is_running and session.process:
            try:
                session.process.terminate()
                session.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                session.process.kill()
            
            session.status = "cancelled"
            session.end_time = time.time()
            session.add_output("Test session cancelled by user")
            return True
        
        return False
    
    def get_session(self, session_id: str) -> Optional[TestSession]:
        """Get test session by ID"""
        return self.sessions.get(session_id)
    
    def get_session_output(self, session_id: str, from_line: int = 0) -> Tuple[List[str], bool]:
        """Get session output from specific line"""
        if session_id not in self.sessions:
            return [], False
        
        session = self.sessions[session_id]
        output_slice = session.output_lines[from_line:]
        is_complete = not session.is_running
        
        return output_slice, is_complete
    
    def list_sessions(self) -> List[Dict]:
        """List all test sessions"""
        return [session.get_summary() for session in self.sessions.values()]
    
    def cleanup_old_sessions(self, max_age: int = 3600):
        """Clean up old completed sessions"""
        current_time = time.time()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            if (not session.is_running and 
                session.end_time and 
                (current_time - session.end_time) > max_age):
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
    
    def get_pytest_help(self) -> List[str]:
        """Get pytest help options"""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.split('\n')
        except Exception as e:
            logger.error(f"Failed to get pytest help: {e}")
            return ["Pytest help not available"]

# Global instances
test_discovery = TestDiscovery()
test_runner = TestRunner()