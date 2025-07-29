"""
Death Note Terminal - Ollama Integration Client

Integrates with DeepSeek-R1 8B for intelligent log analysis and problem identification.
Single responsibility: AI-powered terminal output interpretation.

File: ollama_client.py
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from config import config
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class OllamaResponse:
    """Represents an Ollama model response"""
    
    def __init__(self, response_text: str, metadata: Dict):
        self.text = response_text
        self.metadata = metadata
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class LogAnalyzer:
    """AI-powered log analysis using DeepSeek-R1"""
    
    ANALYSIS_PROMPTS = {
        "error_detection": """
        Analyze the following terminal output for errors, warnings, and issues.
        Focus on identifying:
        1. Critical errors that need immediate attention
        2. Warning signs of potential problems
        3. Performance issues or bottlenecks
        4. Missing dependencies or configuration issues
        
        Provide a concise analysis with specific recommendations.
        
        Terminal output:
        {log_content}
        """,
        
        "test_summary": """
        Analyze this pytest test output and provide a summary:
        1. Overall test results (passed/failed/skipped)
        2. Specific test failures and their causes
        3. Recommendations for fixing failures
        4. Performance insights if available
        
        Pytest output:
        {log_content}
        """,
        
        "server_health": """
        Analyze this server log output for health and performance insights:
        1. Server startup status and any issues
        2. Request patterns and response times
        3. Resource usage concerns
        4. Security or configuration warnings
        
        Server logs:
        {log_content}
        """,
        
        "general_summary": """
        Provide an intelligent summary of this terminal output.
        Focus on the most important information and any actions needed.
        
        Terminal output:
        {log_content}
        """
    }

class OllamaClient:
    """Death Note Terminal Ollama Integration Client"""
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL
        self.timeout = config.OLLAMA_TIMEOUT
        self.max_tokens = config.OLLAMA_MAX_TOKENS
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_connection(self) -> Tuple[bool, str]:
        """Check if Ollama server is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        
                        if self.model in models:
                            return True, f"Connected to Ollama with {self.model}"
                        else:
                            return False, f"Model {self.model} not found. Available: {models}"
                    else:
                        return False, f"Ollama server returned status {response.status}"
        
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[OllamaResponse]:
        """Generate response from Ollama model"""
        
        if not self.session:
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>", "<|endoftext|>"]
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    metadata = {
                        "model": data.get("model", self.model),
                        "total_duration": data.get("total_duration", 0),
                        "load_duration": data.get("load_duration", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration": data.get("eval_duration", 0)
                    }
                    
                    return OllamaResponse(data.get("response", ""), metadata)
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return None
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def analyze_logs(self, log_content: str, analysis_type: str = "general_summary") -> Optional[OllamaResponse]:
        """Analyze log content with specific analysis type"""
        
        if analysis_type not in LogAnalyzer.ANALYSIS_PROMPTS:
            analysis_type = "general_summary"
        
        # Truncate log content if too long (keep recent lines)
        lines = log_content.split('\n')
        if len(lines) > 500:  # Limit to recent 500 lines
            log_content = '\n'.join(lines[-500:])
            log_content = f"[...truncated to last 500 lines...]\n{log_content}"
        
        prompt = LogAnalyzer.ANALYSIS_PROMPTS[analysis_type].format(log_content=log_content)
        
        system_prompt = """You are an expert system administrator and developer helping analyze terminal logs. 
        Provide concise, actionable insights focusing on problems and solutions. 
        Use bullet points for clarity and highlight critical issues."""
        
        return await self.generate_response(prompt, system_prompt)
    
    async def analyze_test_output(self, test_output: str) -> Optional[OllamaResponse]:
        """Specialized analysis for pytest output"""
        return await self.analyze_logs(test_output, "test_summary")
    
    async def analyze_server_logs(self, server_logs: str) -> Optional[OllamaResponse]:
        """Specialized analysis for server logs"""
        return await self.analyze_logs(server_logs, "server_health")
    
    async def detect_errors(self, log_content: str) -> Optional[OllamaResponse]:
        """Focus on error detection and troubleshooting"""
        return await self.analyze_logs(log_content, "error_detection")
    
    def sync_analyze_logs(self, log_content: str, analysis_type: str = "general_summary") -> Optional[Dict]:
        """Synchronous wrapper for log analysis"""
        
        try:
            # Use requests for synchronous operation
            if analysis_type not in LogAnalyzer.ANALYSIS_PROMPTS:
                analysis_type = "general_summary"
            
            # Truncate content
            lines = log_content.split('\n')
            if len(lines) > 500:
                log_content = '\n'.join(lines[-500:])
                log_content = f"[...truncated to last 500 lines...]\n{log_content}"
            
            prompt = LogAnalyzer.ANALYSIS_PROMPTS[analysis_type].format(log_content=log_content)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": """You are an expert system administrator helping analyze logs. 
                Provide concise, actionable insights focusing on problems and solutions.""",
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "analysis": data.get("response", ""),
                    "model": data.get("model", self.model),
                    "duration_ms": data.get("total_duration", 0) // 1000000,
                    "success": True
                }
            else:
                return {
                    "analysis": f"Ollama API error: {response.status_code}",
                    "success": False
                }
        
        except Exception as e:
            logger.error(f"Sync analysis failed: {e}")
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "success": False
            }
    
    async def get_model_info(self) -> Optional[Dict]:
        """Get information about the current model"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/show", json={"name": self.model}) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
        return None

class DeathNoteAnalyzer:
    """Death Note themed wrapper for log analysis"""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.death_note_prompt = """
        Acting as L from Death Note, analyze this log output with sharp deductive reasoning.
        Focus on patterns, anomalies, and logical connections that reveal the truth.
        Provide insights in L's characteristic analytical style.
        """
    
    async def l_analyze(self, log_content: str) -> Optional[str]:
        """Analyze logs in L's detective style"""
        async with self.ollama:
            response = await self.ollama.generate_response(
                f"{log_content}\n\nProvide analysis:",
                self.death_note_prompt
            )
            return response.text if response else None

# Global instances
ollama_client = OllamaClient()
death_note_analyzer = DeathNoteAnalyzer()