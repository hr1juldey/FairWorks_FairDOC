"""
DSPy LiteLLM Call Monitoring - Correct Layer Interception
Monitors actual LiteLLM HTTP calls to Ollama via requests/aiohttp
"""
import pytest
import asyncio
import time
from unittest.mock import patch, Mock, AsyncMock
import requests
import aiohttp
import structlog
from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

# Test environment setup
with patch.dict('os.environ', {
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'FAIRDOC_V2_DSPy_MODEL': settings_v2.DSPY_MODEL_NAME,
}):
    from src.app2.services.dspy.medical_agent import MedicalTriageAgent

class LiteLLMCallMonitor:
    """Monitor LiteLLM HTTP calls via requests and aiohttp"""
    
    def __init__(self):
        self.api_calls = []
        self.original_requests_post = None
        self.original_aiohttp_post = None
    
    def setup_monitoring(self):
        """Setup monitoring for both requests and aiohttp"""
        # Monitor requests (sync calls)
        self.original_requests_post = requests.post
        requests.post = self.mock_requests_post
        
        # Monitor aiohttp (async calls)  
        self.original_aiohttp_post = aiohttp.ClientSession._request
        aiohttp.ClientSession._request = self.mock_aiohttp_request
    
    def cleanup_monitoring(self):
        """Restore original HTTP clients"""
        if self.original_requests_post:
            requests.post = self.original_requests_post
        if self.original_aiohttp_post:
            aiohttp.ClientSession._request = self.original_aiohttp_post
    
    def mock_requests_post(self, url, *args, **kwargs):
        """Monitor requests.post calls"""
        call_info = {
            'url': str(url),
            'method': 'POST',
            'client': 'requests',
            'timestamp': time.time(),
            'is_ollama': 'localhost:11434' in str(url) or '11434' in str(url)
        }
        self.api_calls.append(call_info)
        
        if call_info['is_ollama']:
            logger.info("🌐 LiteLLM requests call intercepted", url=str(url)[:50])
        
        # Make real call
        return self.original_requests_post(url, *args, **kwargs)
    
    async def mock_aiohttp_request(self, method, url, *args, **kwargs):
        """Monitor aiohttp ClientSession calls"""
        call_info = {
            'url': str(url),
            'method': method.upper(),
            'client': 'aiohttp',
            'timestamp': time.time(), 
            'is_ollama': 'localhost:11434' in str(url) or '11434' in str(url)
        }
        self.api_calls.append(call_info)
        
        if call_info['is_ollama']:
            logger.info("🌐 LiteLLM aiohttp call intercepted", url=str(url)[:50])
        
        # Make real call
        return await self.original_aiohttp_post(method, url, *args, **kwargs)

class TestDSPyLiteLLMIntegration:
    """Test DSPy with proper LiteLLM HTTP monitoring"""
    
    @pytest.fixture(autouse=True)
    def setup_litellm_monitoring(self):
        """Setup LiteLLM call monitoring"""
        self.call_monitor = LiteLLMCallMonitor()
        self.call_monitor.setup_monitoring()
        self.medical_agent = MedicalTriageAgent()
        
        logger.info("🔍 LiteLLM call monitoring setup complete")
        yield
        
        self.call_monitor.cleanup_monitoring()
        logger.info("🔄 LiteLLM call monitoring cleaned up")
    
    @pytest.mark.asyncio
    async def test_verify_litellm_calls_to_ollama(self):
        """Verify LiteLLM makes calls to Ollama through requests/aiohttp"""
        
        # Clear previous calls
        self.call_monitor.api_calls.clear()
        
        # Test with emergency symptoms
        emergency_symptoms = "severe crushing chest pain radiating to left arm"
        nice_context = "CG95 Chest Pain: Emergency assessment"
        
        logger.info("🚨 Testing LiteLLM → Ollama integration")
        start_time = time.time()
        
        result = await self.medical_agent.process_turn(
            symptoms=emergency_symptoms,
            nice_context=nice_context
        )
        
        processing_time = time.time() - start_time
        
        # Analyze API calls
        all_calls = self.call_monitor.api_calls
        ollama_calls = [call for call in all_calls if call['is_ollama']]
        
        logger.info("📊 LiteLLM API Call Analysis",
                   total_calls=len(all_calls),
                   ollama_calls=len(ollama_calls),
                   processing_time=f"{processing_time:.2f}s",
                   clients_used=[call['client'] for call in ollama_calls])
        
        # Verify real LLM integration through LiteLLM
        assert len(ollama_calls) > 0, f"No Ollama calls detected via LiteLLM! Found {len(all_calls)} total calls"
        assert processing_time > 1.0, "Response too fast - likely not real LLM inference"
        
        # Verify response quality
        assert result['outcome'] in ['emergency', 'routine', 'inconclusive']
        assert 0 <= result['confidence'] <= 100
        assert isinstance(result['red_flags'], list)
        
        logger.info("✅ LiteLLM → Ollama integration verified",
                   outcome=result['outcome'],
                   confidence=result['confidence'],
                   http_client=ollama_calls[0]['client'] if ollama_calls else 'none')
    
    @pytest.mark.asyncio
    async def test_identify_litellm_http_client(self):
        """Identify which HTTP client LiteLLM actually uses"""
        
        self.call_monitor.api_calls.clear()
        
        # Make a simple call
        result = await self.medical_agent.process_turn(
            symptoms="headache",
            nice_context="Basic assessment"
        )
        
        # Validate the result first
        assert 'outcome' in result, "Result missing outcome field"
        assert 'confidence' in result, "Result missing confidence field"
        assert result['outcome'] in ['emergency', 'routine', 'self', 'inconclusive', 'spam']
        assert 0 <= result['confidence'] <= 100
        
        # Analyze which HTTP clients were used
        ollama_calls = [call for call in self.call_monitor.api_calls if call['is_ollama']]
        clients_used = list(set(call['client'] for call in ollama_calls))
        
        logger.info("🔍 LiteLLM HTTP Client Analysis",
                clients_detected=clients_used,
                total_ollama_calls=len(ollama_calls),
                sample_urls=[call['url'][:50] for call in ollama_calls[:2]],
                result_outcome=result['outcome'],
                result_confidence=result['confidence'])
        
        # Verify we detected the actual HTTP client
        assert len(clients_used) > 0, "No HTTP clients detected"
        assert len(ollama_calls) > 0, "No Ollama calls detected"
        
        # Verify the result indicates real LLM processing occurred
        assert len(result.get('reasoning', '')) > 0, "No reasoning provided - LLM may not have processed"
        
        # Log findings for debugging
        for call in ollama_calls[:3]:  # Show first 3 calls
            logger.info("📡 Detected call",
                    client=call['client'],
                    url=call['url'][:60],
                    method=call['method'])
        
        logger.info("✅ LLM processing verified", 
                outcome=result['outcome'],
                reasoning_length=len(result.get('reasoning', '')))

    
    @pytest.mark.asyncio
    async def test_performance_with_real_calls(self):
        """Test performance characteristics of real LLM calls"""
        
        test_cases = [
            "chest pain",
            "headache", 
            "abdominal pain"
        ]
        
        performance_data = []
        
        for symptoms in test_cases:
            self.call_monitor.api_calls.clear()
            
            start_time = time.time()
            result = await self.medical_agent.process_turn(
                symptoms=symptoms,
                nice_context="Basic assessment"
            )
            processing_time = time.time() - start_time
            
            ollama_calls = [call for call in self.call_monitor.api_calls if call['is_ollama']]
            
            performance_data.append({
                'symptoms': symptoms,
                'processing_time': processing_time,
                'api_calls': len(ollama_calls),
                'outcome': result['outcome'],
                'confidence': result['confidence']
            })
        
        # Analyze performance patterns
        avg_time = sum(p['processing_time'] for p in performance_data) / len(performance_data)
        total_calls = sum(p['api_calls'] for p in performance_data)
        
        logger.info("📈 Performance Analysis",
                   avg_processing_time=f"{avg_time:.2f}s",
                   total_api_calls=total_calls,
                   calls_per_request=total_calls / len(performance_data))
        
        # Performance assertions
        assert avg_time > 0.5, "Average time too fast for real LLM calls"
        assert total_calls > 0, "No API calls detected across all test cases"
        assert all(p['api_calls'] > 0 for p in performance_data), "Some requests made no API calls"
