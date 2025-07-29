"""
Death Note Terminal - Configuration Management

Centralized configuration for ports, paths, and system settings.
Follows single responsibility principle for easy maintenance.

File: config.py
"""

from pathlib import Path
from typing import Dict, List
import os

class Config:
    """Death Note Terminal Configuration Manager"""
    
    # Application Settings
    APP_NAME = "Death Note Terminal"
    VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8999
    DEBUG = True
    
    # Project Structure
    ROOT_DIR = Path(__file__).parent.parent
    SRC_DIR = ROOT_DIR / "src"
    TESTS_DIR = SRC_DIR / "tests"
    
    # Server Configurations
    SERVER_CONFIGS: Dict[str, Dict] = {
        "v1_only": {
            "name": "V1 Only",
            "module": "src.app.main:app",
            "port": 8001,
            "description": "Legacy API (V1) on dedicated port",
            "endpoints": ["/api/v1"]
        },
        "v2_only": {
            "name": "V2 Only", 
            "module": "src.app2.main_v2:app",
            "port": 8002,
            "description": "Modern API (V2) on dedicated port",
            "endpoints": ["/api/v2"]
        },
        "combined": {
            "name": "V1 + V2 Combined",
            "module": "src.app.main:app",  # V2 mounted in V1
            "port": 8000,
            "description": "Both APIs on single port",
            "endpoints": ["/api/v1", "/api/v2"]
        }
    }
    
    # Port Management
    AVAILABLE_PORTS = [8000, 8001, 8002, 8003, 8004, 8005]
    RESERVED_PORTS = [8999, 11434]  # Terminal app, Ollama
    
    # Test Configuration
    TEST_CATEGORIES = ["unit", "integration", "e2e"]
    PYTEST_DEFAULT_ARGS = ["-v", "--tb=short", "--no-header"]
    MAX_TEST_OUTPUT_LINES = 1000
    
    # Terminal Settings
    TERMINAL_COLS = 120
    TERMINAL_ROWS = 30
    MAX_TERMINAL_SESSIONS = 10
    TERMINAL_TIMEOUT = 3600  # 1 hour
    
    # Ollama Integration
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "deepseek-r1:8b"
    OLLAMA_TIMEOUT = 30
    OLLAMA_MAX_TOKENS = 2000
    
    # Death Note Theme
    THEME_CONFIG = {
        "default_theme": "dark",
        "enable_animations": True,
        "paper_texture": True,
        "gothic_fonts": True,
        "shadow_effects": True
    }
    
    # WebSocket Settings
    WS_PING_INTERVAL = 20
    WS_PING_TIMEOUT = 10
    WS_MAX_CONNECTIONS = 50
    
    # Security Settings
    CORS_ORIGINS = ["http://localhost:8999", "http://127.0.0.1:8999"]
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    RATE_LIMIT = "100/minute"
    
    # Logging Configuration
    LOG_LEVEL = "INFO" if not DEBUG else "DEBUG"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def get_server_config(cls, server_type: str) -> Dict:
        """Get configuration for specific server type"""
        return cls.SERVER_CONFIGS.get(server_type, {})
    
    @classmethod
    def get_test_path(cls, test_type: str) -> Path:
        """Get path for specific test category"""
        if test_type not in cls.TEST_CATEGORIES:
            raise ValueError(f"Invalid test type: {test_type}")
        return cls.TESTS_DIR / test_type
    
    @classmethod
    def is_port_available(cls, port: int) -> bool:
        """Check if port is available for use"""
        return port in cls.AVAILABLE_PORTS and port not in cls.RESERVED_PORTS
    
    @classmethod
    def get_custom_port_config(cls, server_type: str, custom_port: int) -> Dict:
        """Create custom port configuration"""
        base_config = cls.get_server_config(server_type).copy()
        if base_config:
            base_config["port"] = custom_port
            base_config["custom"] = True
        return base_config
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check if test directories exist
        for test_type in cls.TEST_CATEGORIES:
            test_path = cls.get_test_path(test_type)
            if not test_path.exists():
                issues.append(f"Test directory missing: {test_path}")
        
        # Check port conflicts
        used_ports = set()
        for config in cls.SERVER_CONFIGS.values():
            port = config["port"]
            if port in used_ports:
                issues.append(f"Port conflict detected: {port}")
            used_ports.add(port)
        
        # Check required directories
        if not cls.SRC_DIR.exists():
            issues.append(f"Source directory missing: {cls.SRC_DIR}")
        
        return issues

# Environment-specific overrides
if os.getenv("DEATH_NOTE_ENV") == "production":
    Config.DEBUG = False
    Config.LOG_LEVEL = "WARNING"
    Config.CORS_ORIGINS = []  # Restrict in production

if os.getenv("DEATH_NOTE_OLLAMA_URL"):
    Config.OLLAMA_BASE_URL = os.getenv("DEATH_NOTE_OLLAMA_URL")

if os.getenv("DEATH_NOTE_OLLAMA_MODEL"):
    Config.OLLAMA_MODEL = os.getenv("DEATH_NOTE_OLLAMA_MODEL")

# Singleton configuration instance
config = Config()
