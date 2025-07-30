"""
Ollama Client - AI Analysis Integration

Provides integration with Ollama for intelligent log analysis,
error detection, and performance insights.

Single responsibility: Ollama API communication and analysis
File: ./ollama_client.py
"""

import aiohttp
import json
import logging
from typing import Dict, Any, Optional
import asyncio

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama AI service"""
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    # Check if our model is available
                    models = [model.get('name', '') for model in data.get('models', [])]
                    if any(self.model in model for model in models):
                        logger.info(f"✅ Ollama connected - {self.model} available")
                        return True
                    else:
                        logger.warning(f"⚠️ Ollama connected but {self.model} not found")
                        return False
                else:
                    logger.error(f"❌ Ollama connection failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Ollama connection error: {e}")
            return False
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text using Ollama"""
        try:
            session = await self._get_session()
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120, connect=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', '')
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    async def analyze(self, content: str, analysis_type: str = "general") -> str:
        """Analyze content with context-specific prompts"""
        
        prompts = {
            "error": f"""
            Analyze this error log and provide:
            1. Root cause analysis
            2. Potential fixes
            3. Prevention strategies
            
            Log content:
            {content}
            """,
            
            "performance": f"""
            Analyze this performance data and provide:
            1. Performance bottlenecks
            2. Optimization recommendations
            3. Resource usage insights
            
            Performance data:
            {content}
            """,
            
            "test": f"""
            Analyze this test output and provide:
            1. Test results summary
            2. Failed test analysis
            3. Improvement suggestions
            
            Test output:
            {content}
            """,
            
            "general": f"""
            Analyze this system log and provide:
            1. Key insights
            2. Potential issues
            3. Recommendations
            
            Log content:
            {content}
            """
        }
        
        prompt = prompts.get(analysis_type, prompts["general"])
        
        try:
            logger.info(f"🤖 Starting analysis: {analysis_type}")
            result = await self.generate(prompt, max_tokens=1500)
            logger.info(f"✅ Analysis completed: {len(result)} characters")
            return result
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            # Try reconnection
            try:
                await self.test_connection()
                logger.info("🔄 Ollama reconnected, retrying...")
                result = await self.generate(prompt, max_tokens=1500)
                return result
            except Exception as retry_error:
                logger.error(f"❌ Retry failed: {retry_error}")
                return f"Analysis failed: {str(e)}. Retry failed: {str(retry_error)}"

    

    async def summarize_logs(self, logs: list) -> str:
        """Summarize multiple log entries"""
        log_text = "\n".join(logs[-50:])  # Last 50 entries
        
        prompt = f"""
        Summarize these system logs focusing on:
        1. Critical events
        2. Error patterns
        3. System health status
        4. Action items
        
        Logs:
        {log_text}
        """
        
        try:
            return await self.generate(prompt, max_tokens=800)
        except Exception as e:
            logger.error(f"❌ Log summarization failed: {e}")
            return f"Summarization failed: {str(e)}"
    
    
    async def detect_anomalies(self, metrics: Dict[str, Any]) -> str:
        """Detect anomalies in system metrics"""
        metrics_text = json.dumps(metrics, indent=2)
        
        prompt = f"""
        Analyze these system metrics for anomalies:
        1. Unusual patterns
        2. Performance degradation
        3. Resource exhaustion risks
        4. Trending issues
        
        Metrics:
        {metrics_text}
        """
        
        try:
            return await self.generate(prompt, max_tokens=1000)
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return f"Anomaly detection failed: {str(e)}"
    
    async def suggest_fixes(self, error_details: str) -> str:
        """Suggest fixes for specific errors"""
        prompt = f"""
        Provide specific, actionable fixes for this error:
        
        Error: {error_details}
        
        Format your response as:
        1. Immediate fix
        2. Root cause fix
        3. Prevention measures
        4. Related documentation
        """
        
        try:
            return await self.generate(prompt, max_tokens=1200)
        except Exception as e:
            logger.error(f"❌ Fix suggestions failed: {e}")
            return f"Fix suggestion failed: {str(e)}"
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔌 Ollama client session closed")
