"""
Dynamic DSPy LLM Configuration Provider

Auto-discovers available models from .env and ollama list command.
Provides simple interface for LLM management with load balancing.
"""

import dspy
import time
import threading
import subprocess
import structlog
from typing import Dict, Optional, List
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

@dataclass
class LLMMetrics:
    """Simple metrics tracking for load balancing"""
    active_requests: int = 0
    total_requests: int = 0
    avg_response_time: float = 1.0
    last_used: float = 0
    
    @property 
    def load_score(self) -> float:
        """Lower score = less loaded"""
        return self.active_requests + (1.0 / max(self.avg_response_time, 0.1))

class DSPyLLMProvider:
    """Dynamic DSPy LLM provider with auto-discovery"""
    
    _instance: Optional['DSPyLLMProvider'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup()
    
    def _setup(self):
        """Initialize the provider with dynamic model discovery"""
        # Add thread-local storage
        self._thread_local = threading.local()
        
        # Discover available models
        self._available_models = self._discover_models()
        
        # LLM instances cache
        self._llm_cache: Dict[str, dspy.LM] = {}
        self._metrics: Dict[str, LLMMetrics] = defaultdict(LLMMetrics)
        
        # Default model from config
        self._default_model = self._get_default_model()
        
        # Configure DSPy with default model
        self._configure_default()
        
        logger.info(f"✅ DSPy Provider initialized with {len(self._available_models)} models")
        logger.info(f"Default model: {self._default_model}")
    
    def _discover_models(self) -> Dict[str, str]:
        """Discover available models from env and ollama list"""
        models = {}
        
        # Method 1: Try to get from environment variables
        env_models = self._get_models_from_env()
        if env_models:
            models.update(env_models)
            logger.info(f"Found {len(env_models)} models from environment")
        
        # Method 2: Get from ollama list command
        ollama_models = self._get_models_from_ollama()
        if ollama_models:
            # Add ollama models that aren't already configured
            for model_key, model_path in ollama_models.items():
                if model_key not in models:
                    models[model_key] = model_path
            logger.info(f"Found {len(ollama_models)} models from ollama")
        
        if not models:
            # Fallback - basic model that should exist
            models = {'gemma3n': 'ollama/gemma3n:e4b'}
            logger.warning("No models discovered, using fallback")
        
        return models
    
    def _get_models_from_env(self) -> Dict[str, str]:
        """Get models from environment variables"""
        models = {}
        
        # Check for explicit model list in env (if added)
        available_models_env = getattr(settings_v2, 'AVAILABLE_MODELS', None)
        if available_models_env:
            # Parse comma-separated model list
            model_list = [m.strip() for m in available_models_env.split(',')]
            for model in model_list:
                if ':' in model:
                    key = model.split(':')[0]
                    models[key] = f'ollama/{model}'
                else:
                    models[model] = f'ollama/{model}'
        
        # Always include the configured default model
        default_model = settings_v2.FAIRDOC_V2_DSPy_MODEL
        if default_model:
            key = default_model.split(':')[0] if ':' in default_model else default_model
            models[key] = f'ollama/{default_model}'
        
        return models
    
    def _get_models_from_ollama(self) -> Dict[str, str]:
        """Get available models from ollama list command"""
        models = {}
        
        try:
            # Run ollama list command
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header line
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            full_model_name = parts[0]  # e.g., "deepseek-r1:8b"
                            
                            # Keep FULL model name as key (including version)
                            models[full_model_name] = f'ollama/{full_model_name}'
                            
                            # ALSO add short version for backward compatibility
                            short_name = full_model_name.split(':')[0]
                            if short_name not in models:
                                models[short_name] = f'ollama/{full_model_name}'
                
                logger.debug(f"Discovered models from ollama: {list(models.keys())}")
            else:
                logger.warning("ollama list command failed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Could not run ollama list: {e}")
        
        return models

    
    def _get_default_model(self) -> str:
        """Get default model from config, ensuring it exists in available models"""
        config_model = settings_v2.FAIRDOC_V2_DSPy_MODEL.lower()
        
        # First try exact match
        for key in self._available_models:
            if key.lower() == config_model.split(':')[0].lower():
                return key
        
        # Then try partial match
        for key in self._available_models:
            if key.lower() in config_model or config_model.split(':')[0].lower() in key.lower():
                return key
        
        # Fallback to first available
        if self._available_models:
            return next(iter(self._available_models.keys()))
        
        raise RuntimeError("No models available for default configuration")
    
    def _configure_default(self):
        """Configure DSPy with default model"""
        try:
            default_llm = self._get_llm_instance(self._default_model)
            dspy.configure(lm=default_llm)
            logger.info(f"✅ DSPy configured with default: {self._default_model}")
        except Exception as e:
            logger.error(f"❌ Failed to configure default DSPy model: {e}")
    
    def _get_llm_instance(self, model_key: str, **kwargs) -> dspy.LM:
        """Get or create LLM instance with caching"""
        cache_key = f"{model_key}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        if model_key not in self._available_models:
            raise ValueError(f"Model {model_key} not available. Available: {list(self._available_models.keys())}")
        
        model_path = self._available_models[model_key]
        

        # Filter params based on provider
        if 'ollama/' in model_path:
            # Remove OpenAI-specific params for Ollama
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in ['n', 'response_format', 'logprobs']}
            
            # Also ensure we don't pass num_candidates as 'n'
            if 'num_candidates' in filtered_kwargs:
                # Rename num_candidates to a supported parameter or remove it
                filtered_kwargs.pop('num_candidates', None)
        else:
            filtered_kwargs = kwargs

        # Create LLM with config
        llm_params = {
            'model': model_path,
            'api_base': settings_v2.OLLAMA_BASE_URL,
            'temperature': kwargs.get('temperature', 0.0),
            'max_tokens': kwargs.get('max_tokens', 4000),
            **kwargs,
            # **filtered_kwargs
        }
        
        try:
            llm = dspy.LM(**llm_params)
            self._llm_cache[cache_key] = llm
            logger.debug(f"✅ Created LLM instance: {model_key}")
            return llm
        except Exception as e:
            logger.error(f"❌ Failed to create LLM {model_key}: {e}")
            raise
    
    def get_llm(self, model: str = None, **kwargs) -> dspy.LM:
        """Thread-safe LM instance retrieval"""
        if not hasattr(self._thread_local, 'llm_cache'):
            self._thread_local.llm_cache = {}
        
        target_model = model or self._default_model
        cache_key = f"{target_model}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key not in self._thread_local.llm_cache:
            self._thread_local.llm_cache[cache_key] = self._get_llm_instance(target_model, **kwargs)
        
        # Update metrics
        self._metrics[target_model].active_requests += 1
        self._metrics[target_model].total_requests += 1
        self._metrics[target_model].last_used = time.time()
        
        return self._thread_local.llm_cache[cache_key]

    
    def get_best_available_llm(self, exclude: List[str] = None, **kwargs) -> dspy.LM:
        """Get least loaded available LLM"""
        exclude = exclude or []
        available = [k for k in self._available_models.keys() if k not in exclude]
        
        if not available:
            return self.get_llm(**kwargs)
        
        # Select based on load
        best_model = min(available, key=lambda m: self._metrics[m].load_score)
        return self.get_llm(best_model, **kwargs)
    
    def record_response_time(self, model: str, response_time: float):
        """Update response time metrics"""
        metric = self._metrics[model]
        metric.avg_response_time = (metric.avg_response_time + response_time) / 2
        metric.active_requests = max(0, metric.active_requests - 1)
    
    @contextmanager
    def using_model(self, model: str, **kwargs):
        """Context manager for using specific model"""
        start_time = time.time()
        original_llm = dspy.settings.lm
        
        try:
            # Switch to requested model
            target_llm = self.get_llm(model, **kwargs)
            dspy.configure(lm=target_llm)
            logger.debug(f"🔄 Switched to model: {model}")
            
            yield target_llm
            
        finally:
            # Record metrics and restore
            response_time = time.time() - start_time
            self.record_response_time(model, response_time)
            dspy.configure(lm=original_llm)
    
    def refresh_models(self):
        """Refresh available models (useful for runtime updates)"""
        logger.info("🔄 Refreshing available models...")
        self._available_models = self._discover_models()
        # Clear cache to force recreation with new models
        self._llm_cache.clear()
        logger.info(f"✅ Refreshed: {len(self._available_models)} models available")
    
    def list_models(self) -> Dict[str, str]:
        """List available models"""
        return self._available_models.copy()
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get current load metrics"""
        return {
            model: {
                'active_requests': metrics.active_requests,
                'total_requests': metrics.total_requests,
                'avg_response_time': metrics.avg_response_time,
                'load_score': metrics.load_score,
                'last_used': metrics.last_used
            }
            for model, metrics in self._metrics.items()
        }

# Global instance
_provider: Optional[DSPyLLMProvider] = None

def get_llm_provider() -> DSPyLLMProvider:
    """Get global LLM provider instance"""
    global _provider
    if _provider is None:
        _provider = DSPyLLMProvider()
    return _provider

def ensure_dspy_configured(model_name: str = None) -> bool:
    """
    Ensure DSPy is configured with the specified model

    Args:
        model_name: Optional model name to use

    Returns:
        bool: True if configuration was successful
    """
    try:
        provider = get_llm_provider()

        # If model_name specified, try to get that specific model
        if model_name:
            # Try to create LLM instance to verify it works
            llm_instance = provider.get_llm(model_name)
            
            # Check if we're in an async context
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                if loop and loop.is_running():
                    # Use context instead of configure in async environments
                    # Note: This sets up the context but doesn't override global config
                    # The actual usage will use dspy.context() when needed
                    dspy.configure(lm=llm_instance)
                    logger.info(f"✅ DSPy prepared for async context with model: {model_name}")
                    return True
            except RuntimeError:
                pass
            
            # Normal synchronous configuration
            dspy.configure(lm=llm_instance)
            logger.info(f"✅ DSPy configured with model: {model_name}")
        else:
            # Use default configuration (already done in provider init)
            logger.info(f"✅ DSPy already configured with default model: {provider._default_model}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to configure DSPy: {e}")
        return False

# Convenience functions
def llm(model: str = None, **kwargs) -> dspy.LM:
    """Get LLM instance - main interface"""
    return get_llm_provider().get_llm(model, **kwargs)

def fast_llm(**kwargs) -> dspy.LM:
    """Get fastest available LLM (prefer gemma3)"""
    provider = get_llm_provider()
    # Try gemma3 first, then any available
    if 'gemma3' in provider._available_models:
        return provider.get_llm('gemma3', **kwargs)
    return provider.get_best_available_llm(**kwargs)

def reasoning_llm(**kwargs) -> dspy.LM:
    """Get reasoning LLM (prefer deepseek-r1)"""
    provider = get_llm_provider()
    # Try deepseek-r1 first, then any available
    if 'deepseek-r1' in provider._available_models:
        return provider.get_llm('deepseek-r1', **kwargs)
    return provider.get_best_available_llm(**kwargs)

def balanced_llm(**kwargs) -> dspy.LM:
    """Get least loaded LLM"""
    return get_llm_provider().get_best_available_llm(**kwargs)

@contextmanager
def using(model: str, **kwargs):
    """Use specific model temporarily"""
    with get_llm_provider().using_model(model, **kwargs) as model_llm:
        yield model_llm

def list_available_models() -> List[str]:
    """List all available model keys"""
    return list(get_llm_provider().list_models().keys())

def refresh_available_models():
    """Refresh available models from ollama"""
    get_llm_provider().refresh_models()

def get_load_metrics() -> Dict:
    """Get current system load metrics"""
    return get_llm_provider().get_metrics()
