"""
E2E Test: System Health and Monitoring

Tests system health endpoints, monitoring capabilities, resilience under load,
error handling, and overall system stability during various operational scenarios.

File: src/tests/e2e/test_e2e_system_health.py
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timezone
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import psutil
import concurrent.futures

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider


def utcnow():
    """
    Return timezone-aware UTC datetime
    
    This function replaces the deprecated datetime.utcnow() method
    with the recommended timezone-aware approach using UTC timezone.
    
    Returns:
        datetime: Current UTC time with timezone information
    """
    return datetime.now(timezone.utc)

class TestSystemHealthE2E:
    """System health, monitoring, and resilience testing"""
    
    @pytest.fixture(autouse=True)
    async def setup_monitoring_environment(self):
        """Setup comprehensive system monitoring"""
        self.test_session_id = f"health_test_{int(time.time())}"
        self.system_metrics = {
            "test_start": utcnow().isoformat(),
            "health_checks": [],
            "performance_metrics": [],
            "error_scenarios": [],
            "load_test_results": {},
            "resource_usage": []
        }
        
        # Start system resource monitoring
        self.monitor_resources = True
        self.resource_monitor_task = asyncio.create_task(self._monitor_system_resources())
        
        yield
        
        # Stop monitoring
        self.monitor_resources = False
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await self.resource_monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage during tests"""
        while self.monitor_resources:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                self.system_metrics["resource_usage"].append({
                    "timestamp": utcnow().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024)
                })
                
                await asyncio.sleep(1)
            except Exception:
                break
    
    @pytest.mark.asyncio
    async def test_health_endpoints_comprehensive(self):
        """Test all health and monitoring endpoints"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            health_endpoints = [
                ("/api/v2/health", "main_health"),
                ("/api/v2/health/ready", "readiness_probe"),
                ("/api/v2/health/live", "liveness_probe"),
                ("/api/v2/info", "api_info")
            ]
            
            for endpoint, endpoint_type in health_endpoints:
                start_time = time.time()
                response = await client.get(endpoint)
                response_time = time.time() - start_time
                
                health_check = {
                    "endpoint": endpoint,
                    "type": endpoint_type,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "timestamp": utcnow().isoformat()
                }
                
                try:
                    health_check["response_data"] = response.json()
                except Exception:
                    health_check["response_data"] = {"raw": response.text}
                
                self.system_metrics["health_checks"].append(health_check)
                
                # Validate health endpoint responses
                if endpoint_type == "liveness_probe":
                    assert response.status_code == 200, "Liveness probe should always return 200"
                    data = response.json()
                    assert data["status"] == "alive"
                    assert "version" in data
                
                elif endpoint_type == "api_info":
                    assert response.status_code == 200, "API info should always be available"
                    data = response.json()
                    assert data["name"] == "Fairdoc AI Triage System V2"
                    assert data["version"] == "v2.6-stable"
                    assert "features" in data
                
                else:
                    # Health and readiness can be degraded but should respond
                    assert response.status_code in [200, 503], f"{endpoint} should respond"
                
                print(f"✅ {endpoint}: {response.status_code} ({response_time:.3f}s)")
    
    @pytest.mark.asyncio
    async def test_system_load_performance(self):
        """Test system performance under various load conditions"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # === Light Load Test (5 concurrent requests) ===
            light_load_results = await self._load_test(client, 5, "light_load")
            
            # === Medium Load Test (15 concurrent requests) ===
            medium_load_results = await self._load_test(client, 15, "medium_load")
            
            # === Heavy Load Test (30 concurrent requests) ===
            heavy_load_results = await self._load_test(client, 30, "heavy_load")
            
            self.system_metrics["load_test_results"] = {
                "light_load": light_load_results,
                "medium_load": medium_load_results,
                "heavy_load": heavy_load_results
            }
            
            # Performance assertions
            assert light_load_results["success_rate"] >= 0.9, "90% success rate under light load"
            assert medium_load_results["avg_response_time"] < 5.0, "Medium load avg response < 5s"
            assert heavy_load_results["success_rate"] >= 0.7, "70% success rate under heavy load"
            
            print("Load Test Results:")
            print(f"  Light (5): {light_load_results['success_rate']:.1%} success, {light_load_results['avg_response_time']:.2f}s avg")
            print(f"  Medium (15): {medium_load_results['success_rate']:.1%} success, {medium_load_results['avg_response_time']:.2f}s avg")
            print(f"  Heavy (30): {heavy_load_results['success_rate']:.1%} success, {heavy_load_results['avg_response_time']:.2f}s avg")
    
    async def _load_test(self, client, concurrent_requests, test_name):
        """Execute load test with specified concurrency"""
        test_messages = [
            "I have a headache",
            "Chest pain and shortness of breath",
            "Feeling nauseous and dizzy",
            "Severe abdominal pain",
            "Back pain for several days"
        ]
        
        tasks = []
        for i in range(concurrent_requests):
            message = test_messages[i % len(test_messages)]
            payload = {
                "user_message": f"{message} (load test {i})",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"{self.test_session_id}_load_{test_name}_{i}",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            task = client.post("/api/v2/medical/chat", json=payload)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = 0
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif hasattr(result, 'status_code'):
                if result.status_code == 200:
                    successful_requests += 1
                else:
                    errors.append(f"HTTP {result.status_code}")
            
        success_rate = successful_requests / concurrent_requests
        avg_response_time = total_time / concurrent_requests
        
        return {
            "concurrent_requests": concurrent_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "errors": errors
        }
    
    @pytest.mark.asyncio
    async def test_error_handling_resilience(self):
        """Test system resilience under various error conditions"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            error_scenarios = [
                {
                    "name": "empty_message",
                    "payload": {
                        "user_message": "",
                        "stakeholder_role": StakeholderRole.PATIENT.value,
                        "stakeholder_id": f"{self.test_session_id}_error_empty"
                    },
                    "expected_status": [400, 422]
                },
                {
                    "name": "invalid_role",
                    "payload": {
                        "user_message": "Test message",
                        "stakeholder_role": "invalid_role",
                        "stakeholder_id": f"{self.test_session_id}_error_role"
                    },
                    "expected_status": [400, 422]
                },
                {
                    "name": "malformed_json",
                    "payload": '{"incomplete": json}',
                    "expected_status": [400, 422],
                    "send_raw": True
                },
                {
                    "name": "nonexistent_conversation",
                    "endpoint": f"/api/v2/medical/chat/{self.test_session_id}_fake/state",
                    "expected_status": [404]
                }
            ]
            
            for scenario in error_scenarios:
                try:
                    if scenario.get("endpoint"):
                        # Test GET endpoint
                        response = await client.get(scenario["endpoint"])
                    elif scenario.get("send_raw"):
                        # Test malformed JSON
                        response = await client.post(
                            "/api/v2/medical/chat",
                            content=scenario["payload"],
                            headers={"content-type": "application/json"}
                        )
                    else:
                        # Test POST with payload
                        response = await client.post("/api/v2/medical/chat", json=scenario["payload"])
                    
                    error_result = {
                        "scenario": scenario["name"],
                        "status_code": response.status_code,
                        "expected_status": scenario["expected_status"],
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0,
                        "handled_correctly": response.status_code in scenario["expected_status"]
                    }
                    
                    self.system_metrics["error_scenarios"].append(error_result)
                    
                    assert response.status_code in scenario["expected_status"], \
                        f"Scenario {scenario['name']} returned {response.status_code}, expected {scenario['expected_status']}"
                    
                    print(f"✅ Error scenario '{scenario['name']}': {response.status_code}")
                    
                except Exception as e:
                    print(f"❌ Error scenario '{scenario['name']}' failed: {e}")
                    raise
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_errors(self):
        """Test system recovery after encountering errors"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Trigger multiple error conditions
            error_requests = []
            for i in range(5):
                payload = {
                    "user_message": "",  # Invalid empty message
                    "stakeholder_role": "invalid",
                    "stakeholder_id": f"error_test_{i}"
                }
                error_requests.append(client.post("/api/v2/medical/chat", json=payload))
            
            # Execute error requests
            await asyncio.gather(*error_requests, return_exceptions=True)
            
            # Test normal request after errors
            normal_payload = {
                "user_message": "I have a headache after multiple system errors",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"{self.test_session_id}_recovery_test",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            recovery_response = await client.post("/api/v2/medical/chat", json=normal_payload)
            
            # System should recover and process normal request
            assert recovery_response.status_code == 200, "System should recover after errors"
            
            recovery_data = recovery_response.json()
            assert "conversation_id" in recovery_data
            assert "medical_outcome" in recovery_data
            
            print("✅ System successfully recovered after error conditions")
            
            # Verify health endpoints still work
            health_response = await client.get("/api/v2/health")
            assert health_response.status_code in [200, 503], "Health endpoint should remain responsive"
    
    @pytest.mark.asyncio
    async def test_system_monitoring_during_operation(self):
        """Test system monitoring capabilities during normal operation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Start background monitoring
            monitoring_task = asyncio.create_task(self._continuous_health_monitoring(client))
            
            # Simulate normal operation with various requests
            normal_operations = [
                "I have a mild headache",
                "Chest discomfort after exercise",
                "Feeling tired and rundown",
                "Minor cut on finger",
                "Stomach ache after eating"
            ]
            
            for i, message in enumerate(normal_operations):
                payload = {
                    "user_message": message,
                    "stakeholder_role": StakeholderRole.PATIENT.value,
                    "stakeholder_id": f"{self.test_session_id}_monitoring_{i}",
                    "chat_provider": ChatProvider.API_DIRECT.value
                }
                
                await client.post("/api/v2/medical/chat", json=payload)
                await asyncio.sleep(0.5)  # Simulate realistic request timing
            
            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            print(f"✅ System monitoring completed during {len(normal_operations)} operations")
    
    async def _continuous_health_monitoring(self, client):
        """Continuously monitor health endpoints during operation"""
        while True:
            try:
                health_response = await client.get("/api/v2/health")
                
                health_metric = {
                    "timestamp": utcnow().isoformat(),
                    "endpoint": "/api/v2/health",
                    "status_code": health_response.status_code,
                    "response_time": 0  # Simplified for this test
                }
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    health_metric["system_status"] = health_data.get("status")
                    health_metric["services"] = health_data.get("services", {})
                
                self.system_metrics["performance_metrics"].append(health_metric)
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except asyncio.CancelledError:
                break
            except Exception:
                break
    
    def teardown_method(self):
        """Print comprehensive system health analysis"""
        self.system_metrics["test_end"] = utcnow().isoformat()
        
        print(f"\n{'=' * 80}")
        print("SYSTEM HEALTH & MONITORING E2E TEST REPORT")
        print(f"{'=' * 80}")
        print(f"Session ID: {self.test_session_id}")
        
        # Health checks summary
        if self.system_metrics["health_checks"]:
            print(f"\nHEALTH ENDPOINTS TESTED: {len(self.system_metrics['health_checks'])}")
            for check in self.system_metrics["health_checks"]:
                status_indicator = "✅" if check["status_code"] in [200, 503] else "❌"
                print(f"  {status_indicator} {check['endpoint']}: {check['status_code']} ({check['response_time']:.3f}s)")
        
        # Load test summary
        if self.system_metrics["load_test_results"]:
            print("\nLOAD TEST PERFORMANCE:")
            for test_name, results in self.system_metrics["load_test_results"].items():
                print(f"  {test_name.title()}: {results['success_rate']:.1%} success, {results['avg_response_time']:.2f}s avg")
        
        # Error handling summary
        if self.system_metrics["error_scenarios"]:
            handled_correctly = sum(1 for e in self.system_metrics["error_scenarios"] if e["handled_correctly"])
            total_scenarios = len(self.system_metrics["error_scenarios"])
            print(f"\nERROR HANDLING: {handled_correctly}/{total_scenarios} scenarios handled correctly")
        
        # Resource usage summary
        if self.system_metrics["resource_usage"]:
            cpu_usage = [r["cpu_percent"] for r in self.system_metrics["resource_usage"]]
            memory_usage = [r["memory_percent"] for r in self.system_metrics["resource_usage"]]
            
            if cpu_usage and memory_usage:
                print("\nRESOURCE USAGE:")
                print(f"  CPU: {min(cpu_usage):.1f}% - {max(cpu_usage):.1f}% (avg: {sum(cpu_usage) / len(cpu_usage):.1f}%)")
                print(f"  Memory: {min(memory_usage):.1f}% - {max(memory_usage):.1f}% (avg: {sum(memory_usage) / len(memory_usage):.1f}%)")
        
        print(f"{'=' * 80}\n")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
