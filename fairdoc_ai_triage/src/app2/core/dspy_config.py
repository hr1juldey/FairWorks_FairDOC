"""
Centralized DSPy Configuration Manager

Single point of configuration for all DSPy LLM instances across app2 and tests.
Implements singleton pattern to ensure only one LLM configuration exists.
"""

import dspy
import structlog
from typing import Optional, Dict, Any
from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

class DSPyConfigManager:
    """Singleton manager for DSPy LLM configuration"""
    
    _instance: Optional['DSPyConfigManager'] = None
    _configured: bool = False
    _current_model: Optional[str] = None
    _current_config: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'DSPyConfigManager':
        """Ensure singleton instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize only once"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
    
    def configure_dspy(self, 
                      model_name: Optional[str] = None,
                      force_reconfigure: bool = False) -> bool:
        """
        Configure DSPy with Ollama LLM - called once for entire application
        
        Args:
            model_name: Model name to use (defaults to settings)
            force_reconfigure: Force reconfiguration even if already configured
            
        Returns:
            bool: True if configuration was successful
        """
        model_name = model_name or settings_v2.FAIRDOC_V2_DSPy_MODEL
        
        # Skip if already configured with same model
        if (self._configured and 
            self._current_model == model_name and 
            not force_reconfigure):
            logger.debug("✅ DSPy already configured", model=model_name)
            return True
            
        try:
            # Primary method: Enhanced Ollama integration
            lm = dspy.LM(
                model=f'ollama/{model_name}',
                api_base=settings_v2.OLLAMA_BASE_URL,
                temperature=0.0,
                max_tokens=8000,
                cache=True
            )
            
            # Configure DSPy globally - this affects ALL DSPy modules
            dspy.configure(lm=lm)
            
            # Store configuration state
            self._configured = True
            self._current_model = model_name
            self._current_config = {
                'model': model_name,
                'api_base': settings_v2.OLLAMA_BASE_URL,
                'provider': 'ollama'
            }
            
            logger.info("✅ DSPy configured globally", 
                       model=model_name,
                       api_base=settings_v2.OLLAMA_BASE_URL,
                       singleton=True)
            return True
            
        except Exception as primary_error:
            logger.warning("⚠️ Primary Ollama config failed, trying fallback", 
                          error=str(primary_error))
            
            try:
                # Fallback: OpenAI-compatible endpoint
                lm = dspy.OpenAI(
                    api_base='http://localhost:11434/v1/',
                    api_key='ollama',
                    model=model_name,
                    model_type='chat',
                    temperature=0.0,
                    max_tokens=4000
                )
                
                dspy.configure(lm=lm)
                
                self._configured = True
                self._current_model = model_name
                self._current_config = {
                    'model': model_name,
                    'api_base': 'http://localhost:11434/v1/',
                    'provider': 'openai_compatible'
                }
                
                logger.info("✅ DSPy configured with fallback", 
                           model=model_name, 
                           singleton=True)
                return True
                
            except Exception as fallback_error:
                logger.error("❌ Both DSPy configuration methods failed",
                           primary_error=str(primary_error),
                           fallback_error=str(fallback_error))
                self._configured = False
                return False
    
    def is_configured(self) -> bool:
        """Check if DSPy is properly configured"""
        return self._configured
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get current DSPy configuration details"""
        return self._current_config.copy() if self._current_config else None
    
    def get_current_model(self) -> Optional[str]:
        """Get currently configured model name"""
        return self._current_model
    
    def reset_configuration(self) -> None:
        """Reset configuration state (for testing)"""
        self._configured = False
        self._current_model = None
        self._current_config = None
        logger.info("🔄 DSPy configuration reset")

# Singleton instance
dspy_config = DSPyConfigManager()

def ensure_dspy_configured(model_name: Optional[str] = None) -> bool:
    """
    Convenience function to ensure DSPy is configured
    
    Args:
        model_name: Optional model name to use
        
    Returns:
        bool: True if configuration successful
    """
    return dspy_config.configure_dspy(model_name)

def get_dspy_config() -> DSPyConfigManager:
    """Get the singleton DSPy configuration manager"""
    return dspy_config
