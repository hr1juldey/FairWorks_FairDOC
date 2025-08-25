# src/tests/utils/llm_warmup.py

"""
LLM Warmup Utility - Production Grade

Warms up LLM models before test execution to ensure fair timing measurements.
Supports both direct Ollama CLI and REST API endpoints.

Single Responsibility: LLM warmup and readiness verification
"""

import asyncio
import subprocess
import time
import httpx
import structlog
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)

class WarmupMethod(str, Enum):
    """LLM warmup methods"""
    OLLAMA_CLI = "ollama_cli"
    REST_API = "rest_api"
    BOTH = "both"

@dataclass
class WarmupConfig:
    """LLM warmup configuration"""
    model_name: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 180
    warmup_method: WarmupMethod = WarmupMethod.REST_API
    verify_response: bool = True
    max_retries: int = 3

@dataclass
class WarmupResult:
    """LLM warmup result"""
    success: bool
    model_name: str
    warmup_time_seconds: float
    method_used: WarmupMethod
    error_message: Optional[str] = None
    model_loaded: bool = False
    response_received: bool = False

class LLMWarmupService:
    """Service for warming up LLM models before tests"""
    
    def __init__(self, config: WarmupConfig):
        self.config = config
        self.warmup_prompt = "Hello! Please respond with 'Ready' to confirm you're loaded."
        self.expected_keywords = ["ready", "hello", "yes", "ok"]
        
    async def warmup_llm(self) -> WarmupResult:
        """
        Warm up LLM using configured method
        
        Returns:
            WarmupResult with success status and timing
        """
        logger.info("🔥 Starting LLM warmup", 
                   model=self.config.model_name, 
                   method=self.config.warmup_method)
        
        start_time = time.time()
        
        try:
            if self.config.warmup_method == WarmupMethod.OLLAMA_CLI:
                result = await self._warmup_via_cli()
            elif self.config.warmup_method == WarmupMethod.REST_API:
                result = await self._warmup_via_api()
            elif self.config.warmup_method == WarmupMethod.BOTH:
                # Try CLI first, fallback to API
                result = await self._warmup_via_cli()
                if not result.success:
                    logger.warning("CLI warmup failed, trying API")
                    result = await self._warmup_via_api()
            else:
                raise ValueError(f"Unknown warmup method: {self.config.warmup_method}")
                
            warmup_time = time.time() - start_time
            result.warmup_time_seconds = warmup_time
            
            if result.success:
                logger.info("✅ LLM warmup completed successfully", 
                           model=self.config.model_name,
                           time_seconds=warmup_time,
                           method=result.method_used)
            else:
                logger.error("❌ LLM warmup failed",
                           model=self.config.model_name,
                           error=result.error_message)
                           
            return result
            
        except Exception as e:
            warmup_time = time.time() - start_time
            error_msg = f"Warmup exception: {str(e)}"
            logger.error("💥 LLM warmup exception", error=error_msg)
            
            return WarmupResult(
                success=False,
                model_name=self.config.model_name,
                warmup_time_seconds=warmup_time,
                method_used=self.config.warmup_method,
                error_message=error_msg
            )
    
    async def _warmup_via_cli(self) -> WarmupResult:
        """Warm up LLM via Ollama CLI commands"""
        logger.info("🖥️ Warming up via Ollama CLI")
        
        try:
            # Check if ollama is available
            check_cmd = ["ollama", "list"]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            
            if check_result.returncode != 0:
                return WarmupResult(
                    success=False,
                    model_name=self.config.model_name,
                    warmup_time_seconds=0,
                    method_used=WarmupMethod.OLLAMA_CLI,
                    error_message="Ollama CLI not available"
                )
            
            # Run interactive warmup session
            cmd = ["ollama", "run", self.config.model_name]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Send warmup prompt
            process.stdin.write(f"{self.warmup_prompt}\n")
            process.stdin.flush()
            
            # Wait for response with timeout
            try:
                stdout, stderr = process.communicate(input="/bye\n", timeout=self.config.timeout_seconds)
                
                # Check if we got a reasonable response
                response_received = any(keyword in stdout.lower() for keyword in self.expected_keywords)
                
                return WarmupResult(
                    success=True,
                    model_name=self.config.model_name,
                    warmup_time_seconds=0,  # Will be set by caller
                    method_used=WarmupMethod.OLLAMA_CLI,
                    model_loaded=True,
                    response_received=response_received
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                return WarmupResult(
                    success=False,
                    model_name=self.config.model_name,
                    warmup_time_seconds=0,
                    method_used=WarmupMethod.OLLAMA_CLI,
                    error_message="CLI warmup timeout"
                )
                
        except Exception as e:
            return WarmupResult(
                success=False,
                model_name=self.config.model_name,
                warmup_time_seconds=0,
                method_used=WarmupMethod.OLLAMA_CLI,
                error_message=f"CLI warmup error: {str(e)}"
            )
    
    async def _warmup_via_api(self) -> WarmupResult:
        """Warm up LLM via REST API"""
        logger.info("🌐 Warming up via REST API")
        
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            try:
                # Check if Ollama API is available
                health_url = f"{self.config.base_url}/api/tags"
                health_response = await client.get(health_url)
                
                if health_response.status_code != 200:
                    return WarmupResult(
                        success=False,
                        model_name=self.config.model_name,
                        warmup_time_seconds=0,
                        method_used=WarmupMethod.REST_API,
                        error_message="Ollama API not available"
                    )
                
                # Send warmup request
                generate_url = f"{self.config.base_url}/api/generate"
                payload = {
                    "model": self.config.model_name,
                    "prompt": self.warmup_prompt,
                    "stream": False
                }
                
                response = await client.post(generate_url, json=payload)
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get("response", "").lower()
                    
                    response_received = any(keyword in response_text for keyword in self.expected_keywords)
                    
                    return WarmupResult(
                        success=True,
                        model_name=self.config.model_name,
                        warmup_time_seconds=0,  # Will be set by caller
                        method_used=WarmupMethod.REST_API,
                        model_loaded=True,
                        response_received=response_received
                    )
                else:
                    return WarmupResult(
                        success=False,
                        model_name=self.config.model_name,
                        warmup_time_seconds=0,
                        method_used=WarmupMethod.REST_API,
                        error_message=f"API request failed: {response.status_code}"
                    )
                    
            except httpx.TimeoutException:
                return WarmupResult(
                    success=False,
                    model_name=self.config.model_name,
                    warmup_time_seconds=0,
                    method_used=WarmupMethod.REST_API,
                    error_message="API warmup timeout"
                )
            except Exception as e:
                return WarmupResult(
                    success=False,
                    model_name=self.config.model_name,
                    warmup_time_seconds=0,
                    method_used=WarmupMethod.REST_API,
                    error_message=f"API warmup error: {str(e)}"
                )

async def warmup_model_for_tests(model_name: str, method: WarmupMethod = WarmupMethod.REST_API) -> WarmupResult:
    """
    Convenience function to warm up a model for tests
    
    Args:
        model_name: Name of the model to warm up
        method: Warmup method to use
        
    Returns:
        WarmupResult indicating success/failure
    """
    config = WarmupConfig(
        model_name=model_name,
        warmup_method=method,
        timeout_seconds=45,
        verify_response=True
    )
    
    warmup_service = LLMWarmupService(config)
    return await warmup_service.warmup_llm()

def warmup_models_sync(model_names: list[str], method: WarmupMethod = WarmupMethod.REST_API) -> Dict[str, WarmupResult]:
    """
    Synchronous wrapper for warming up multiple models
    Used in test configuration where async is not available
    
    Args:
        model_names: List of model names to warm up
        method: Warmup method to use
        
    Returns:
        Dictionary mapping model names to warmup results
    """
    async def _warmup_all():
        results = {}
        for model_name in model_names:
            result = await warmup_model_for_tests(model_name, method)
            results[model_name] = result
        return results
    
    return asyncio.run(_warmup_all())

# Export main interfaces
__all__ = [
    'WarmupConfig',
    'WarmupResult', 
    'WarmupMethod',
    'LLMWarmupService',
    'warmup_model_for_tests',
    'warmup_models_sync'
]
