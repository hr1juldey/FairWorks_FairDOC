"""
Environment-aware configuration for Fairdoc backend.
Supports switching between local dev, test server, and production environments.
"""
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
import chromadb
from chromadb.config import Settings as ChromaSettings
import aioredis
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentSettings(BaseSettings):
    """Initial settings to determine which environment to load."""
    ENVIRONMENT: str = "local"  # local, test, prod
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def load_environment_config() -> str:
    """Load the appropriate .env file based on ENVIRONMENT setting."""
    # Load base .env first
    load_dotenv()
    
    # Get environment setting
    env_settings = EnvironmentSettings()
    environment = env_settings.ENVIRONMENT.lower()
    
    # Load environment-specific config
    env_files = {
        "local": ".env.local",
        "dev": ".env.local",  # Alias for local
        "test": ".env.test",
        "staging": ".env.test",  # Alias for test
        "prod": ".env.prod",
        "production": ".env.prod"  # Alias for prod
    }
    
    if environment in env_files:
        env_file = env_files[environment]
        if Path(env_file).exists():
            load_dotenv(env_file, override=True)
            logger.info(f"✅ Loaded environment config: {env_file}")
        else:
            logger.warning(f"⚠️ Environment file not found: {env_file}, using base config")
    
    return environment

class Settings(BaseSettings):
    """Load all settings from environment variables with environment switching support."""
    
    # Environment Information
    ENVIRONMENT: str = "local"
    
    # Project Structure
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # API Configuration
    APP_NAME: str
    API_HOST: str
    API_PORT: int
    DEBUG: bool
    
    # Security
    SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL_CLINICAL: str
    OLLAMA_MODEL_CHAT: str
    OLLAMA_MODEL_CLASSIFICATION: str
    OLLAMA_TIMEOUT: int
    OLLAMA_MAX_TOKENS: int
    OLLAMA_TEMPERATURE: float
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str
    CHROMA_COLLECTION_MEDICAL: str
    CHROMA_COLLECTION_PATIENTS: str
    CHROMA_COLLECTION_BIAS: str
    
    # Redis Configuration
    REDIS_URL: str
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int
    REDIS_MAX_CONNECTIONS: int
    
    # WebSocket Configuration
    WS_MAX_CONNECTIONS: int
    WS_HEARTBEAT_INTERVAL: int
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int
    ALLOWED_EXTENSIONS: str
    
    # ML Configuration
    ML_BATCH_SIZE: int
    ML_MAX_CONTEXT_LENGTH: int
    ML_CACHE_SIZE: int
    
    # Bias Detection
    BIAS_THRESHOLD_DEMOGRAPHIC: float
    BIAS_THRESHOLD_EQUALIZED_ODDS: float
    BIAS_THRESHOLD_CALIBRATION: float
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    
    class Config:
        case_sensitive = True
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT.lower() in ["local", "dev"]
    
    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.ENVIRONMENT.lower() in ["test", "staging"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() in ["prod", "production"]
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.PROJECT_ROOT / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.PROJECT_ROOT / "logs"
    
    @property
    def uploads_dir(self) -> Path:
        """Get uploads directory path."""
        return self.PROJECT_ROOT / "uploads"
    
    @property
    def allowed_extensions_list(self) -> list:
        """Get allowed file extensions as list."""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    def get_cors_origins(self) -> list:
        """Get CORS origins based on environment."""
        if self.is_production:
            return [
                "https://yourdomain.com",
                "https://api.yourdomain.com"
            ]
        elif self.is_testing:
            return [
                "https://test.yourdomain.com",
                "https://staging.yourdomain.com"
            ]
        else:  # Development
            return [
                "http://localhost:3000",
                "http://localhost:8080",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080"
            ]

class DatabaseManager:
    """Environment-aware database manager."""
    
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chroma_client: Optional[chromadb.Client] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.collections: Dict[str, chromadb.Collection] = {}
        
    async def initialize(self) -> bool:
        """Initialize database connections with environment-specific settings."""
        try:
            # Create directories
            Path(self.settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
            self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            
            await self._init_chromadb()
            await self._init_redis()
            
            logger.info(f"✅ Databases initialized for {self.settings.ENVIRONMENT} environment")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database init failed: {e}")
            return False
    
    async def _init_chromadb(self) -> None:
        """Initialize ChromaDB with environment-specific settings."""
        chroma_settings = ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=self.settings.is_development  # Only allow reset in dev
        )
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIRECTORY,
            settings=chroma_settings
        )
        
        # Create collections with environment-specific prefixes
        env_prefix = f"{self.settings.ENVIRONMENT}_" if not self.settings.is_production else ""
        
        collections = [
            f"{env_prefix}{self.settings.CHROMA_COLLECTION_MEDICAL}",
            f"{env_prefix}{self.settings.CHROMA_COLLECTION_PATIENTS}",
            f"{env_prefix}{self.settings.CHROMA_COLLECTION_BIAS}"
        ]
        
        for name in collections:
            collection = self.chroma_client.get_or_create_collection(name=name)
            self.collections[name] = collection
            logger.info(f"ChromaDB collection '{name}': {collection.count()} documents")
    
    async def _init_redis(self) -> None:
        """Initialize Redis with environment-specific settings."""
        # Add connection retry logic for production
        max_retries = 3 if self.settings.is_production else 1
        
        for attempt in range(max_retries):
            try:
                self.redis_client = aioredis.from_url(
                    self.settings.REDIS_URL,
                    password=self.settings.REDIS_PASSWORD,
                    db=self.settings.REDIS_DB,
                    max_connections=self.settings.REDIS_MAX_CONNECTIONS,
                    decode_responses=True,
                    retry_on_timeout=self.settings.is_production,
                    socket_connect_timeout=10,
                    socket_keepalive=True
                )
                await self.redis_client.ping()
                logger.info(f"✅ Redis connected ({self.settings.ENVIRONMENT})")
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check with environment information."""
        status = {
            "environment": self.settings.ENVIRONMENT,
            "chromadb": "unknown",
            "redis": "unknown"
        }
        
        try:
            if self.chroma_client:
                total = sum(c.count() for c in self.collections.values())
                status["chromadb"] = f"healthy - {total} documents"
        except Exception as e:
            status["chromadb"] = f"error - {str(e)}"
        
        try:
            if self.redis_client:
                await self.redis_client.ping()
                status["redis"] = "healthy"
        except Exception as e:
            status["redis"] = f"error - {str(e)}"
        
        return status
    
    async def close(self) -> None:
        """Close connections."""
        if self.redis_client:
            await self.redis_client.close()


# Load environment configuration first
current_environment = load_environment_config()

# Global instances
settings = Settings()
db_manager = DatabaseManager(settings)

# Log current environment
logger.info(f"🚀 Fairdoc backend starting in {current_environment.upper()} environment")
