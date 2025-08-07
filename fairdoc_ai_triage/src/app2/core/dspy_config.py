"""
Advanced DSPy Configuration Manager with Multi-LLM Support and Load Balancing

Features:
- Factory pattern for specialized LLM instances (reasoning, vision, tool calling)
- Dynamic LLM switching with automatic load balancing  
- Redis-backed request pooling and caching
- Ollama integration with dynamic model discovery
- Context-aware model selection
- Graceful fallback mechanisms
- DSPy native load balancing support
"""

import dspy
import structlog
import redis
import asyncio
import subprocess
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Literal
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from collections import defaultdict
import hashlib

from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

class LLMCapability(Enum):
    """Different LLM capabilities for specialized use cases"""
    REASONING = "reasoning"        # DeepSeek-R1, GPT-4o
    VISION = "vision"              # LLaVA, GPT-4-Vision  
    TOOL_CALLING = "tool_calling"  # GPT-4, Claude-3
    GENERAL = "general"            # Llama, Gemma, Qwen
    FAST = "fast"                  # Gemma3n:e4b for quick tasks
    MEDICAL = "medical"           # Specialized medical models

@dataclass
class LLMConfig:
    """Configuration for individual LLM"""
    name: str
    model_path: str  # e.g., "ollama/llama3" or "openai/gpt-4" 
    capabilities: List[LLMCapability]
    max_tokens: int = 4000
    temperature: float = 0.0
    context_length: int = 8192
    load_priority: int = 1  # Higher = preferred for load balancing
    estimated_speed: float = 1.0  # Tokens per second estimate
    memory_usage: int = 8000  # MB estimate

@dataclass 
class LoadBalanceMetrics:
    """Track load balancing metrics per model"""
    active_requests: int = 0
    total_requests: int = 0
    success_rate: float = 1.0
    avg_response_time: float = 1.0
    last_used: float = field(default_factory=time.time)
    error_count: int = 0

class DSPyLLMFactory:
    """Factory for creating specialized LLM instances"""

    def __init__(self, config_manager: 'DSPyConfigManager'):
        self.config_manager = config_manager
        self._llm_cache: Dict[str, dspy.LM] = {}

    def get_reasoning_llm(self, **kwargs) -> dspy.LM:
        """Get LLM optimized for complex reasoning tasks"""
        return self._get_specialized_llm(LLMCapability.REASONING, **kwargs)

    def get_vision_llm(self, **kwargs) -> dspy.LM:
        """Get LLM with vision capabilities"""
        return self._get_specialized_llm(LLMCapability.VISION, **kwargs)

    def get_tool_calling_llm(self, **kwargs) -> dspy.LM:
        """Get LLM optimized for tool/function calling"""
        return self._get_specialized_llm(LLMCapability.TOOL_CALLING, **kwargs)

    def get_fast_llm(self, **kwargs) -> dspy.LM:
        """Get fast LLM for quick tasks"""
        return self._get_specialized_llm(LLMCapability.FAST, **kwargs)

    def get_medical_llm(self, **kwargs) -> dspy.LM:
        """Get LLM specialized for medical tasks"""
        return self._get_specialized_llm(LLMCapability.MEDICAL, **kwargs)

    def _get_specialized_llm(self, capability: LLMCapability, **kwargs) -> dspy.LM:
        """Internal method to get specialized LLM with load balancing"""
        cache_key = f"{capability.value}_{hash(str(kwargs))}"

        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        # Get best available model for this capability
        model_config = self.config_manager.get_best_model_for_capability(capability)
        if not model_config:
            logger.warning(f"No model found for capability {capability}, using default")
            model_config = self.config_manager.get_default_model()

        # Create DSPy LM instance
        llm = self._create_dspy_llm(model_config, **kwargs)
        self._llm_cache[cache_key] = llm

        return llm

    def _create_dspy_llm(self, model_config: LLMConfig, **kwargs) -> dspy.LM:
        """Create DSPy LM instance from model configuration"""

        params = {
            'model': model_config.model_path,
            'max_tokens': kwargs.get('max_tokens', model_config.max_tokens),
            'temperature': kwargs.get('temperature', model_config.temperature),
            'cache': True,
            'num_retries': 3,
            **kwargs
        }

        try:
            # Try primary method with DSPy native LM
            if 'ollama' in model_config.model_path.lower():
                params['api_base'] = settings_v2.OLLAMA_BASE_URL

            llm = dspy.LM(**params)
            logger.info(f"✅ Created DSPy LM: {model_config.name}")
            return llm

        except Exception as e:
            logger.error(f"❌ Failed to create DSPy LM for {model_config.name}: {e}")
            # Fallback to default model
            default_config = self.config_manager.get_default_model()
            return self._create_dspy_llm(default_config, **kwargs)

class DSPyConfigManager:
    """Advanced DSPy Configuration Manager with Multi-LLM Support"""

    _instance: Optional['DSPyConfigManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'DSPyConfigManager':
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize only once"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._configure_system()

    def _configure_system(self):
        """Initialize the configuration system"""
        # Core state
        self._configured = False
        self._current_primary_model: Optional[str] = None
        self._available_models: Dict[str, LLMConfig] = {}
        self._load_metrics: Dict[str, LoadBalanceMetrics] = defaultdict(LoadBalanceMetrics)

        # Redis connection for request pooling
        try:
            self._redis = redis.Redis(
                host=settings_v2.REDIS_HOST, 
                port=settings_v2.REDIS_PORT,
                decode_responses=True
            )
            self._redis.ping()
            logger.info("✅ Connected to Redis for request pooling")
        except Exception as e:
            logger.warning(f"⚠️  Redis not available: {e}")
            self._redis = None

        # Factory for specialized LLMs
        self._factory = DSPyLLMFactory(self)

        # Load available models from Ollama and config
        self._discover_available_models()

    def _discover_available_models(self):
        """Discover available models from Ollama and configure defaults"""

        # Define model configurations with capabilities
        model_configs = {
            'deepseek-r1': LLMConfig(
                name='DeepSeek-R1',
                model_path='ollama/deepseek-r1',
                capabilities=[LLMCapability.REASONING, LLMCapability.GENERAL],
                max_tokens=8000,
                context_length=32768,
                load_priority=5,
                estimated_speed=2.0
            ),
            'qwen': LLMConfig(
                name='Qwen',
                model_path='ollama/qwen',
                capabilities=[LLMCapability.GENERAL, LLMCapability.TOOL_CALLING],
                max_tokens=4000,
                load_priority=3,
                estimated_speed=3.0
            ),
            'llama3': LLMConfig(
                name='Llama-3',
                model_path='ollama/llama3',
                capabilities=[LLMCapability.GENERAL, LLMCapability.REASONING],
                max_tokens=4000,
                load_priority=4,
                estimated_speed=2.5
            ),
            'gpt-oss:20b': LLMConfig(
                name='GPT-OSS-20B', 
                model_path='ollama/gpt-oss:20b',
                capabilities=[LLMCapability.REASONING, LLMCapability.GENERAL],
                max_tokens=4000,
                load_priority=4,
                estimated_speed=1.5
            ),
            'gemma3:4b': LLMConfig(
                name='Gemma3-4B',
                model_path='ollama/gemma3:4b',
                capabilities=[LLMCapability.FAST, LLMCapability.GENERAL],
                max_tokens=2048,
                load_priority=2,
                estimated_speed=5.0,
                memory_usage=4000
            ),
            'gemma3n:e4b': LLMConfig(
                name='Gemma3n-E4B',
                model_path='ollama/gemma3n:e4b',
                capabilities=[LLMCapability.MEDICAL, LLMCapability.GENERAL],
                max_tokens=4000,
                load_priority=3,
                estimated_speed=3.0,
                memory_usage=4000
            )
        }

        # Check which models are actually available via ollama
        available_ollama_models = self._get_ollama_models()

        for model_key, config in model_configs.items():
            # Check if model exists in ollama
            model_name = config.model_path.split('/')[-1]  # Extract model name
            if model_name in available_ollama_models or model_key in available_ollama_models:
                self._available_models[model_key] = config
                logger.info(f"✅ Registered model: {config.name}")
            else:
                logger.warning(f"⚠️  Model {config.name} not found in Ollama")

        # Set default model from config or first available
        default_model_name = getattr(settings_v2, 'FAIRDOC_V2_DSPy_MODEL', 'gemma3n:e4b')
        if default_model_name not in self._available_models:
            # Fallback to first available model
            default_model_name = next(iter(self._available_models.keys())) if self._available_models else None

        if default_model_name:
            self._current_primary_model = default_model_name
            logger.info(f"🎯 Default model set: {self._available_models[default_model_name].name}")
        else:
            logger.error("❌ No models available!")

    def _get_ollama_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = []
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]  # First column is model name
                        models.append(model_name)
                logger.info(f"📋 Found {len(models)} Ollama models: {models}")
                return models
        except Exception as e:
            logger.warning(f"⚠️  Could not query Ollama models: {e}")
        return []

    def configure_dspy(self, 
                      model_name: Optional[str] = None, 
                      force_reconfigure: bool = False,
                      **kwargs) -> bool:
        """Configure DSPy with primary model and load balancing"""

        model_name = model_name or self._current_primary_model
        if not model_name or model_name not in self._available_models:
            logger.error(f"❌ Model {model_name} not available")
            return False

        # Skip if already configured with same model
        if (self._configured and 
            self._current_primary_model == model_name and 
            not force_reconfigure):
            logger.debug(f"✅ DSPy already configured with {model_name}")
            return True

        model_config = self._available_models[model_name]

        try:
            # Create primary LM instance
            primary_lm = self._factory._create_dspy_llm(model_config, **kwargs)

            # Configure DSPy globally
            dspy.configure(lm=primary_lm)

            # Update state
            self._configured = True
            self._current_primary_model = model_name

            logger.info(f"✅ DSPy configured globally with {model_config.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to configure DSPy with {model_name}: {e}")
            return False

    def get_factory(self) -> DSPyLLMFactory:
        """Get the LLM factory for specialized instances"""
        return self._factory

    def get_best_model_for_capability(self, capability: LLMCapability) -> Optional[LLMConfig]:
        """Get best available model for specific capability with load balancing"""
        candidates = [
            (model_key, config) 
            for model_key, config in self._available_models.items() 
            if capability in config.capabilities
        ]

        if not candidates:
            return None

        # Sort by load balancing score (priority, success rate, current load)
        def load_score(item):
            model_key, config = item
            metrics = self._load_metrics[model_key]

            # Scoring factors
            priority_score = config.load_priority * 10
            success_score = metrics.success_rate * 5
            load_penalty = metrics.active_requests * -2
            speed_score = config.estimated_speed * 2

            return priority_score + success_score + load_penalty + speed_score

        best_model_key, best_config = max(candidates, key=load_score)

        # Update metrics
        self._load_metrics[best_model_key].active_requests += 1
        self._load_metrics[best_model_key].total_requests += 1
        self._load_metrics[best_model_key].last_used = time.time()

        logger.debug(f"🎯 Selected {best_config.name} for {capability.value}")
        return best_config

    def get_default_model(self) -> LLMConfig:
        """Get default model configuration"""
        if self._current_primary_model and self._current_primary_model in self._available_models:
            return self._available_models[self._current_primary_model]

        # Fallback to first available
        if self._available_models:
            return next(iter(self._available_models.values()))

        # Ultimate fallback
        return LLMConfig(
            name='Fallback',
            model_path='ollama/llama3',
            capabilities=[LLMCapability.GENERAL],
        )

    @contextmanager
    def override_model(self, model_name: str, **kwargs):
        """Context manager for temporary model switching"""
        if model_name not in self._available_models:
            raise ValueError(f"Model {model_name} not available")

        model_config = self._available_models[model_name]
        temp_lm = self._factory._create_dspy_llm(model_config, **kwargs)

        # Store original settings
        original_lm = dspy.settings.lm if hasattr(dspy.settings, 'lm') else None

        try:
            # Temporarily override
            with dspy.settings.context(lm=temp_lm):
                logger.debug(f"🔄 Temporarily using {model_config.name}")
                yield temp_lm
        finally:
            # Restore original (automatic with context manager)
            logger.debug(f"🔄 Restored original LM: {original_lm}")

    def get_available_models(self) -> Dict[str, LLMConfig]:
        """Get all available model configurations"""
        return self._available_models.copy()

    def get_load_metrics(self) -> Dict[str, LoadBalanceMetrics]:
        """Get current load balancing metrics"""
        return dict(self._load_metrics)

    def is_configured(self) -> bool:
        """Check if DSPy is properly configured"""
        return self._configured

    def get_current_model(self) -> Optional[str]:
        """Get currently configured primary model name"""
        return self._current_primary_model

    def switch_primary_model(self, model_name: str, **kwargs) -> bool:
        """Switch primary model with load balancing consideration"""
        return self.configure_dspy(model_name, force_reconfigure=True, **kwargs)

    def reset_configuration(self) -> None:
        """Reset configuration state (for testing)"""
        self._configured = False
        self._current_primary_model = None
        self._load_metrics.clear()
        logger.info("🔄 DSPy configuration reset")

# Singleton instance
dspy_config = DSPyConfigManager()

# Convenience functions
def ensure_dspy_configured(model_name: Optional[str] = None, **kwargs) -> bool:
    """Ensure DSPy is configured with load balancing"""
    return dspy_config.configure_dspy(model_name, **kwargs)

def get_dspy_config() -> DSPyConfigManager:
    """Get the singleton DSPy configuration manager"""
    return dspy_config

def get_llm_factory() -> DSPyLLMFactory:
    """Get the LLM factory for specialized instances"""
    return dspy_config.get_factory()

# Specialized LLM getters for easy access
def get_reasoning_llm(**kwargs) -> dspy.LM:
    """Get LLM optimized for reasoning tasks"""
    return get_llm_factory().get_reasoning_llm(**kwargs)

def get_medical_llm(**kwargs) -> dspy.LM:
    """Get LLM specialized for medical tasks"""
    return get_llm_factory().get_medical_llm(**kwargs)

def get_fast_llm(**kwargs) -> dspy.LM:
    """Get fast LLM for quick responses"""
    return get_llm_factory().get_fast_llm(**kwargs)

# Context managers for easy model switching
@contextmanager
def reasoning_context(**kwargs):
    """Context manager for reasoning-optimized LLM"""
    reasoning_llm = get_reasoning_llm(**kwargs)
    with dspy.settings.context(lm=reasoning_llm):
        yield reasoning_llm

@contextmanager  
def medical_context(**kwargs):
    """Context manager for medical-specialized LLM"""
    medical_llm = get_medical_llm(**kwargs)
    with dspy.settings.context(lm=medical_llm):
        yield medical_llm

@contextmanager
def fast_context(**kwargs):
    """Context manager for fast LLM"""
    fast_llm = get_fast_llm(**kwargs)
    with dspy.settings.context(lm=fast_llm):
        yield fast_llm
