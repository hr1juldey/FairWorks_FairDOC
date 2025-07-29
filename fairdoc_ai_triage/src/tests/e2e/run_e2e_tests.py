#!/usr/bin/env python3
"""
Comprehensive E2E Test Runner for Fairdoc AI V2 System

Executes all E2E tests with detailed reporting, performance analysis,
and debugging information for both offline and live environments.

Usage:
    python src/tests/e2e/run_e2e_tests.py [--live] [--debug] [--report-file output.json]

File: src/tests/e2e/run_e2e_tests.py
"""

import asyncio
import argparse
import json
import time
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import psutil
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def utcnow():
    """
    Return timezone-aware UTC datetime
    
    This function replaces the deprecated datetime.utcnow() method
    with the recommended timezone-aware approach using UTC timezone.
    
    Returns:
        datetime: Current UTC time with timezone information
    """
    return datetime.now(timezone.utc)


class E2ETestRunner:
    """Comprehensive E2E test runner with detailed reporting"""
    
    def __init__(self, live_mode: bool = False, debug: bool = False):
        self.live_mode = live_mode
        self.debug = debug
        self.test_session_id = f"e2e_run_{int(time.time())}"
        self.results = {
            "session_id": self.test_session_id,
            "start_time": utcnow().isoformat(),
            "environment": "live" if live_mode else "test",
            "system_info": self._get_system_info(),
            "test_results": {},
            "performance_summary": {},
            "errors": [],
            "recommendations": []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "timestamp": utcnow().isoformat()
        }
    
    async def check_prerequisites(self) -> bool:
        """Check system prerequisites before running tests"""
        print("🔍 Checking E2E test prerequisites...")
        
        prerequisites = {
            "python_packages": ["pytest", "httpx", "asyncio", "fastapi"],
            "environment_vars": ["ENVIRONMENT", "DATABASE_URL", "REDIS_URL"],
            "system_resources": {"min_memory_gb": 2, "min_disk_space_gb": 1}
        }
        
        checks_passed = True
        
        # Check Python packages
        for package in prerequisites["python_packages"]:
            try:
                __import__(package)
                print(f"  ✅ {package} installed")
            except ImportError:
                print(f"  ❌ {package} not found")
                checks_passed = False
        
        # Check environment variables
        for env_var in prerequisites["environment_vars"]:
            if os.getenv(env_var):
                print(f"  ✅ {env_var} configured")
            else:
                print(f"  ⚠️ {env_var} not set (using defaults)")
        
        # Check system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        
        if memory_gb >= prerequisites["system_resources"]["min_memory_gb"]:
            print(f"  ✅ Memory: {memory_gb:.1f}GB available")
        else:
            print(f"  ⚠️ Low memory: {memory_gb:.1f}GB (recommended: 2GB+)")
        
        if disk_gb >= prerequisites["system_resources"]["min_disk_space_gb"]:
            print(f"  ✅ Disk space: {disk_gb:.1f}GB available")
        else:
            print(f"  ⚠️ Low disk space: {disk_gb:.1f}GB")
        
        return checks_passed
    
    async def run_test_suite(self, test_file: str, test_name: str) -> Dict[str, Any]:
        """Run a specific test suite and collect results"""
        print(f"\n{'=' * 60}")
        print(f"🧪 Running {test_name}")
        print(f"{'=' * 60}")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--asyncio-mode=auto",
            "--tb=short",
            "--json-report",
            f"--json-report-file=/tmp/{test_name}_report.json"
        ]
        
        if self.debug:
            cmd.extend(["-s", "--capture=no"])
        
        try:
            # Run test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test suite
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_result = {
                "test_name": test_name,
                "file": test_file,
                "execution_time": execution_time,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": result.returncode == 0
            }
            
            # Try to load JSON report
            try:
                with open(f"/tmp/{test_name}_report.json", "r") as f:
                    json_report = json.load(f)
                    test_result["detailed_results"] = json_report
            except Exception:
                pass
            
            if test_result["passed"]:
                print(f"✅ {test_name} passed in {execution_time:.2f}s")
            else:
                print(f"❌ {test_name} failed in {execution_time:.2f}s")
                if self.debug:
                    print(f"STDERR: {result.stderr}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} timed out after 5 minutes")
            return {
                "test_name": test_name,
                "file": test_file,
                "execution_time": 300,
                "exit_code": -1,
                "error": "Test timed out",
                "passed": False
            }
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            return {
                "test_name": test_name,
                "file": test_file,
                "execution_time": time.time() - start_time,
                "exit_code": -1,
                "error": str(e),
                "passed": False
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all E2E tests in sequence"""
        print(f"🚀 Starting E2E Test Suite - Session: {self.test_session_id}")
        print(f"Environment: {'LIVE' if self.live_mode else 'TEST'}")
        print(f"Debug Mode: {'ON' if self.debug else 'OFF'}")
        
        # Check prerequisites
        if not await self.check_prerequisites():
            print("⚠️ Some prerequisites not met, but continuing...")
        
        # Define test suites
        test_suites = [
            {
                "file": "src/tests/e2e/test_e2e_emergency_detection.py",
                "name": "emergency_detection",
                "description": "Emergency detection and response flow"
            },
            {
                "file": "src/tests/e2e/test_e2e_multiturn_conversation.py", 
                "name": "multiturn_conversation",
                "description": "Multi-turn conversation workflow"
            },
            {
                "file": "src/tests/e2e/test_e2e_system_health.py",
                "name": "system_health",
                "description": "System health and monitoring"
            }
        ]
        
        total_start_time = time.time()
        
        # Run each test suite
        for suite in test_suites:
            result = await self.run_test_suite(suite["file"], suite["name"])
            self.results["test_results"][suite["name"]] = result
        
        total_execution_time = time.time() - total_start_time
        self.results["total_execution_time"] = total_execution_time
        self.results["end_time"] = utcnow().isoformat()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate test execution summary"""
        total_tests = len(self.results["test_results"])
        passed_tests = sum(1 for r in self.results["test_results"].values() if r["passed"])
        failed_tests = total_tests - passed_tests
        
        avg_execution_time = sum(
            r["execution_time"] for r in self.results["test_results"].values()
        ) / total_tests if total_tests > 0 else 0
        
        self.results["performance_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "average_execution_time": avg_execution_time,
            "total_execution_time": self.results["total_execution_time"]
        }
        
        # Generate recommendations
        if failed_tests > 0:
            self.results["recommendations"].append(
                "Some tests failed - check logs for debugging information"
            )
        
        if avg_execution_time > 30:
            self.results["recommendations"].append(
                "Tests running slower than expected - check system resources"
            )
        
        if self.results["performance_summary"]["success_rate"] < 0.8:
            self.results["recommendations"].append(
                "Low success rate - system may need debugging or optimization"
            )
    
    def print_detailed_report(self):
        """Print comprehensive test results"""
        print(f"\n{'=' * 80}")
        print("FAIRDOC AI V2 E2E TEST SUITE RESULTS")
        print(f"{'=' * 80}")
        print(f"Session ID: {self.results['session_id']}")
        print(f"Environment: {self.results['environment'].upper()}")
        print(f"Execution Time: {self.results['total_execution_time']:.2f}s")
        
        # Summary
        summary = self.results["performance_summary"]
        print("\n📊 SUMMARY:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']} ✅")
        print(f"  Failed: {summary['failed_tests']} ❌")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Average Time: {summary['average_execution_time']:.2f}s")
        
        # Individual test results
        print("\n🧪 INDIVIDUAL TEST RESULTS:")
        for test_name, result in self.results["test_results"].items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"  {test_name}: {status} ({result['execution_time']:.2f}s)")
            if not result["passed"] and "error" in result:
                print(f"    Error: {result['error']}")
        
        # System info
        sys_info = self.results["system_info"]
        print("\n💻 SYSTEM INFO:")
        print(f"  Platform: {sys_info['platform']}")
        print(f"  Python: {sys_info['python_version'].split()[0]}")
        print(f"  CPU Cores: {sys_info['cpu_count']}")
        print(f"  Memory: {sys_info['memory_gb']}GB")
        print(f"  Disk Usage: {sys_info['disk_usage_percent']:.1f}%")
        
        # Recommendations
        if self.results["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                print(f"  • {rec}")
        
        print(f"{'=' * 80}")
    
    def save_report(self, filename: str):
        """Save detailed report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Detailed report saved to: {filename}")

async def main():
    """Main entry point for E2E test runner"""
    parser = argparse.ArgumentParser(description="Fairdoc AI V2 E2E Test Runner")
    parser.add_argument("--live", action="store_true", 
                       help="Run against live system (default: test mode)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--report-file", default="e2e_test_report.json",
                       help="Output file for detailed report")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = E2ETestRunner(live_mode=args.live, debug=args.debug)
    
    try:
        # Run all tests
        results = await runner.run_all_tests()
        
        # Print results
        runner.print_detailed_report()
        
        # Save detailed report
        runner.save_report(args.report_file)
        
        # Exit with appropriate code
        success_rate = results["performance_summary"]["success_rate"]
        if success_rate >= 0.8:
            print(f"\n🎉 E2E Tests completed successfully!  (success rate: {success_rate:.1%})")
            sys.exit(0)
        else:
            print(f"\n⚠️ E2E Tests completed with issues (success rate: {success_rate:.1%})")
            sys.exit(1)
    
    except KeyboardInterrupt as e:
        print(f"\n🛑 E2E Tests interrupted by user {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 E2E Test runner crashed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())
