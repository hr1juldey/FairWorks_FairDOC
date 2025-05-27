# 🚀 **10-Day Fairdoc AI Implementation Roadmap**
## **Solo Developer | 12 Hours/Day | 120 Total Hours | Always-Live Localhost**

---

## **📊 Current State Assessment**
```
✅ Project structure: Fairdoc/backend/ with datamodels
✅ Core datamodels: 8/9 files completed  
✅ Requirements.txt: Python 3.13 compatible
✅ Environment files: .env templates ready
❌ No working localhost
❌ No database connections
❌ No API endpoints
❌ No frontend
```

---

## **🎯 Final Goal After 10 Days**
```
✅ Working full-stack Fairdoc AI application on localhost
✅ NHS-compatible medical triage system
✅ Real-time chat with AI assistant
✅ Multi-modal file upload (images, audio)
✅ Bias monitoring dashboard
✅ NHS EHR integration (mock)
✅ Complete authentication system
✅ Production-ready deployment setup
```

---

# **DAY 1: Foundation & Live Server Setup**
## **🕐 12 Hours | Goal: Working localhost with health check**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 1.1: Fix Existing Files (1 hour)**
```bash
# Current working directory: Fairdoc/backend/

# Fix datamodel filenames
mv datamodels/chatmodels.py datamodels/chat_models.py
mv datamodels/medical_model.py datamodels/medical_models.py

# Verify all datamodel files exist
ls -la datamodels/
# Expected output:
# auth_models.py
# base_models.py  
# bias_models.py
# chat_models.py
# file_models.py
# medical_models.py
# ml_models.py
# nhs_ehr_models.py
```

#### **Step 1.2: Create Basic Database Connection (1 hour)**
Create: `data/database/connection_manager.py`
```python
"""
Basic database connection manager for Fairdoc AI
Keeps connections alive and handles reconnections
"""
import asyncio
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import settings

logger = logging.getLogger(__name__)

# Create base for ORM models
Base = declarative_base()

class DatabaseManager:
    """Manages database connections with auto-reconnect"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self._connected = False
    
    def initialize(self):
        """Initialize database connections"""
        try:
            # Sync engine for basic operations
            self.engine = create_engine(
                settings.POSTGRES_URL,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 min
                echo=settings.DEBUG  # Log SQL in debug mode
            )
            
            # Async engine for FastAPI
            self.async_engine = create_async_engine(
                settings.POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://"),
                pool_pre_ping=True,
                pool_recycle=300,
                echo=settings.DEBUG
            )
            
            # Session makers
            self.SessionLocal = sessionmaker(bind=self.engine)
            self.AsyncSessionLocal = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._connected = True
            logger.info("✅ Database connected successfully")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected
    
    def get_session(self):
        """Get sync database session"""
        if not self._connected:
            self.initialize()
        return self.SessionLocal()
    
    def get_async_session(self):
        """Get async database session"""
        if not self._connected:
            self.initialize()
        return self.AsyncSessionLocal()

# Global database manager instance
db_manager = DatabaseManager()

def get_db():
    """Dependency to get database session"""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    """Dependency to get async database session"""
    async with db_manager.get_async_session() as db:
        yield db
```

#### **Step 1.3: Create Core Configuration (1 hour)**
Update: `core/config.py`
```python
"""
Configuration management for Fairdoc AI
Handles environment variables and settings
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # App settings
    APP_NAME: str = "Fairdoc AI Medical Triage"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Server settings  
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    POSTGRES_URL: str = "postgresql://fairdoc:password@localhost:5432/fairdoc_dev"
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # AI Model settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    ENABLE_GPU: bool = True
    
    # NHS settings (mock for development)
    NHS_API_BASE_URL: str = "https://sandbox.api.nhs.uk"
    NHS_CLIENT_ID: str = "mock_client_id"
    NHS_CLIENT_SECRET: str = "mock_client_secret"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings
```

#### **Step 1.4: Create Basic Health Check API (1 hour)**
Create: `api/health/routes.py`
```python
"""
Health check endpoints for Fairdoc AI
Monitors system status and dependencies
"""
import time
import psutil
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from data.database.connection_manager import db_manager
from core.config import get_settings

router = APIRouter()

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    services: Dict[str, str]
    system: Dict[str, Any]

class SystemInfo(BaseModel):
    """System information model"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    python_version: str

start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(settings = Depends(get_settings)):
    """
    Main health check endpoint
    Returns system status and service health
    """
    current_time = datetime.utcnow()
    uptime = time.time() - start_time
    
    # Check database connection
    db_status = "connected" if db_manager.is_connected() else "disconnected"
    
    # Get system info
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "python_version": "3.13.3"
    }
    
    # Overall status
    services = {
        "api": "running",
        "database": db_status,
        "redis": "not_implemented",  # Will implement later
        "ollama": "not_implemented"   # Will implement later
    }
    
    overall_status = "healthy" if all(
        status in ["running", "connected", "not_implemented"] 
        for status in services.values()
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=current_time,
        uptime_seconds=uptime,
        version=settings.VERSION,
        services=services,
        system=system_info
    )

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with more system information"""
    return {
        "database": {
            "connected": db_manager.is_connected(),
            "pool_size": "not_implemented"
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "used_percent": psutil.virtual_memory().percent
        },
        "cpu": {
            "count": psutil.cpu_count(),
            "usage_percent": psutil.cpu_percent(interval=1)
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "used_percent": psutil.disk_usage('/').percent
        }
    }

@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity test"""
    return {"message": "pong", "timestamp": datetime.utcnow()}
```

### **📅 Afternoon Session (4 hours): 12:00 PM - 4:00 PM**

#### **Step 1.5: Create Main FastAPI Application (1 hour)**
Update: `app.py`
```python
"""
Main FastAPI application for Fairdoc AI Medical Triage System
Entry point for the backend API
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# Import routers
from api.health.routes import router as health_router

# Import core components
from core.config import settings
from data.database.connection_manager import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Fairdoc AI Medical Triage System...")
    
    # Initialize database
    db_manager.initialize()
    
    logger.info("✅ Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Fairdoc AI...")
    logger.info("✅ Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered medical triage system with bias monitoring",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "path": str(request.url)
        }
    )

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
```

#### **Step 1.6: Create Environment Configuration (1 hour)**
Update: `.env`
```env
# Fairdoc AI Configuration
# Development Environment

# Application Settings
APP_NAME=Fairdoc AI Medical Triage
DEBUG=true
VERSION=1.0.0

# Server Settings
HOST=0.0.0.0
PORT=8000

# Database Settings
POSTGRES_URL=postgresql://fairdoc:password@localhost:5432/fairdoc_dev

# Redis Settings  
REDIS_URL=redis://localhost:6379/0

# Security Settings
SECRET_KEY=dev-secret-key-change-in-production-12345
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Model Settings
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_GPU=true

# NHS Mock Settings
NHS_API_BASE_URL=https://sandbox.api.nhs.uk
NHS_CLIENT_ID=mock_client_id
NHS_CLIENT_SECRET=mock_client_secret
```

#### **Step 1.7: Setup Docker Development Environment (1 hour)**
Create: `docker-compose.dev.yml`
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    container_name: fairdoc_postgres
    environment:
      POSTGRES_DB: fairdoc_dev
      POSTGRES_USER: fairdoc
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fairdoc -d fairdoc_dev"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: fairdoc_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Adminer for database management
  adminer:
    image: adminer
    container_name: fairdoc_adminer
    ports:
      - "8080:8080"
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: fairdoc_network
```

#### **Step 1.8: Create Database Initialization Script (1 hour)**
Create: `init.sql`
```sql
-- Fairdoc AI Database Initialization
-- Creates basic tables for Day 1 setup

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create basic health check table
CREATE TABLE IF NOT EXISTS health_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert initial health check record
INSERT INTO health_checks (status, details) VALUES (
    'healthy',
    '{"message": "Database initialized successfully", "version": "1.0.0"}'
);

-- Create basic users table (for future auth)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create basic sessions table (for future auth)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create basic medical assessments table (for future triage)
CREATE TABLE IF NOT EXISTS medical_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID,
    chief_complaint TEXT,
    assessment_data JSONB,
    risk_level VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_assessments_patient_id ON medical_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON medical_assessments(created_at);

-- Log successful initialization
INSERT INTO health_checks (status, details) VALUES (
    'initialized',
    '{"message": "All tables created successfully", "tables": ["health_checks", "users", "user_sessions", "medical_assessments"]}'
);
```

### **📅 Evening Session (4 hours): 6:00 PM - 10:00 PM**

#### **Step 1.9: Create Development Startup Script (30 minutes)**
Create: `scripts/dev-start.sh`
```bash
#!/bin/bash
# Fairdoc AI Development Startup Script
# Starts all services with live reload

set -e  # Exit on any error

echo "🚀 Starting Fairdoc AI Development Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start database services
echo "📊 Starting database services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
sleep 10

# Check database connection
echo "🔍 Checking database connection..."
while ! docker exec fairdoc_postgres pg_isready -U fairdoc -d fairdoc_dev > /dev/null 2>&1; do
    echo "⏳ Waiting for PostgreSQL..."
    sleep 2
done
echo "✅ PostgreSQL is ready"

while ! docker exec fairdoc_redis redis-cli ping > /dev/null 2>&1; do
    echo "⏳ Waiting for Redis..."
    sleep 2
done
echo "✅ Redis is ready"

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python -m venv venv
fi

echo "📦 Installing/updating Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Start the FastAPI server
echo "🚀 Starting FastAPI server..."
echo "📡 API will be available at: http://localhost:8000"
echo "📚 API docs will be available at: http://localhost:8000/docs"
echo "🗄️  Database admin at: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all services"

# Start with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Cleanup function
cleanup() {
    echo ""
    echo "🔄 Shutting down services..."
    docker-compose -f docker-compose.dev.yml down
    echo "✅ All services stopped"
}

# Set trap to cleanup on exit
trap cleanup EXIT
```

#### **Step 1.10: Make Script Executable and Test Setup (30 minutes)**
```bash
# Make the script executable
chmod +x scripts/dev-start.sh

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Create basic project structure
mkdir -p api/health
mkdir -p data/database
mkdir -p core

# Verify all files are in place
find . -name "*.py" -o -name "*.yml" -o -name "*.sql" -o -name "*.sh" | sort
```

#### **Step 1.11: First Live Test (1 hour)**
```bash
# Start the development environment
./scripts/dev-start.sh

# In another terminal, test the endpoints:
curl http://localhost:8000/
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ping

# Expected responses:
# / should return app info
# /api/v1/health should return system status
# /api/v1/ping should return "pong"
```

#### **Step 1.12: Create Day 1 Verification Checklist (30 minutes)**
Create: `verification/day1-checklist.md`
```markdown
# Day 1 Verification Checklist

## ✅ Infrastructure Setup
- [ ] PostgreSQL running on localhost:5432
- [ ] Redis running on localhost:6379  
- [ ] FastAPI server running on localhost:8000
- [ ] Adminer database admin on localhost:8080

## ✅ API Endpoints Working
- [ ] GET / returns app information
- [ ] GET /api/v1/health returns system status
- [ ] GET /api/v1/ping returns "pong"
- [ ] GET /docs shows API documentation

## ✅ Database Connection
- [ ] Database connection successful
- [ ] Health checks table created
- [ ] Basic tables created (users, sessions, assessments)

## ✅ Live Reload
- [ ] Code changes trigger automatic reload
- [ ] No errors in console output
- [ ] Services restart gracefully

## 🎯 Success Criteria for Day 1
All checkboxes above must be ✅ before proceeding to Day 2
```

#### **Step 1.13: End of Day 1 Testing (1.5 hours)**
```bash
# Complete system test
echo "Testing complete system..."

# Test database connection
curl -s http://localhost:8000/api/v1/health | jq .

# Test auto-reload by modifying health endpoint
# Change version number in core/config.py
# Verify API docs update automatically

# Test error handling
curl -s http://localhost:8000/nonexistent

# Test CORS headers
curl -i -H "Origin: http://localhost:3000" http://localhost:8000/api/v1/health

# Performance test
for i in {1..10}; do
    curl -s http://localhost:8000/api/v1/ping > /dev/null
    echo "Request $i completed"
done
```

### **📈 Day 1 Success Metrics**
- ✅ `curl http://localhost:8000/health` returns 200 status
- ✅ Database queries execute successfully  
- ✅ Auto-reload works when files change
- ✅ All services start without errors
- ✅ API documentation accessible at `/docs`

---

# **DAY 2: Authentication & User Management**
## **🕐 12 Hours | Goal: Complete auth system with JWT tokens**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 2.1: Create Authentication Models (1 hour)**
Create: `datamodels/auth_models_extended.py`
```python
"""
Extended authentication models for Fairdoc AI
Builds on base auth models with JWT and session management
"""
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr
from datamodels.auth_models import User, UserCreate, UserResponse  # Import existing

class TokenData(BaseModel):
    """JWT token data"""
    username: Optional[str] = None
    user_id: Optional[UUID] = None
    scopes: List[str] = []

class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: UUID
    user_email: str

class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str
    remember_me: bool = False

class PasswordReset(BaseModel):
    """Password reset request"""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8)

class UserProfile(BaseModel):
    """Extended user profile"""
    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    role: str = "patient"  # patient, doctor, admin
    is_verified: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None
    profile_picture: Optional[str] = None
    phone: Optional[str] = None
    
class UserUpdate(BaseModel):
    """User profile update"""
    full_name: Optional[str] = None
    phone: Optional[str] = None
    profile_picture: Optional[str] = None
```

#### **Step 2.2: Create Authentication Service (1 hour)**
Create: `services/auth_service.py`
```python
"""
Authentication service for Fairdoc AI
Handles user registration, login, JWT tokens, and password management
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import settings
from data.repositories.auth_repository import AuthRepository
from datamodels.auth_models_extended import Token, LoginRequest, UserCreate

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT security
security = HTTPBearer()

class AuthService:
    """Authentication service with JWT token management"""
    
    def __init__(self):
        self.auth_repo = AuthRepository()
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def register_user(self, user_data: UserCreate, db: Session) -> dict:
        """Register a new user"""
        try:
            # Check if user already exists
            existing_user = await self.auth_repo.get_user_by_email(user_data.email, db)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password
            hashed_password = self.get_password_hash(user_data.password)
            
            # Create user
            user = await self.auth_repo.create_user(
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                db=db
            )
            
            logger.info(f"User registered successfully: {user.email}")
            
            return {
                "user_id": user.id,
                "email": user.email,
                "message": "User registered successfully"
            }
            
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def authenticate_user(self, login_data: LoginRequest, db: Session) -> Token:
        """Authenticate user and return token"""
        try:
            # Get user by email
            user = await self.auth_repo.get_user_by_email(login_data.email, db)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password"
                )
            
            # Verify password
            if not self.verify_password(login_data.password, user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password"
                )
            
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is disabled"
                )
            
            # Create access token
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            if login_data.remember_me:
                access_token_expires = timedelta(days=30)  # 30 days for remember me
            
            access_token = self.create_access_token(
                data={"sub": str(user.id), "email": user.email},
                expires_delta=access_token_expires
            )
            
            # Update last login
            await self.auth_repo.update_last_login(user.id, db)
            
            logger.info(f"User authenticated successfully: {user.email}")
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=int(access_token_expires.total_seconds()),
                user_id=user.id,
                user_email=user.email
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    async def get_current_user(self, token: str, db: Session):
        """Get current user from JWT token"""
        try:
            payload = self.verify_token(token)
            user_id = payload.get("sub")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
            
            user = await self.auth_repo.get_user_by_id(UUID(user_id), db)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            return user
            
        except Exception as e:
            logger.error(f"Get current user failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

# Global auth service instance
auth_service = AuthService()
```

#### **Step 2.3: Create Authentication Repository (1 hour)**
Create: `data/repositories/auth_repository.py`
```python
"""
Authentication repository for Fairdoc AI
Handles user data operations with the database
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Boolean, DateTime, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base

from data.database.connection_manager import Base

logger = logging.getLogger(__name__)

# SQLAlchemy User model
class UserModel(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(50), default="patient")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class AuthRepository:
    """Repository for user authentication operations"""
    
    async def create_user(
        self, 
        email: str, 
        hashed_password: str, 
        full_name: Optional[str] = None,
        db: Session = None
    ) -> UserModel:
        """Create a new user"""
        try:
            user = UserModel(
                email=email,
                hashed_password=hashed_password,
                full_name=full_name
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"User created successfully: {email}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create user {email}: {e}")
            raise
    
    async def get_user_by_email(self, email: str, db: Session) -> Optional[UserModel]:
        """Get user by email address"""
        try:
            user = db.query(UserModel).filter(UserModel.email == email).first()
            return user
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def get_user_by_id(self, user_id: UUID, db: Session) -> Optional[UserModel]:
        """Get user by ID"""
        try:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    async def update_last_login(self, user_id: UUID, db: Session) -> bool:
        """Update user's last login timestamp"""
        try:
            db.query(UserModel).filter(UserModel.id == user_id).update(
                {"last_login": datetime.utcnow()}
            )
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update last login for user {user_id}: {e}")
            return False
    
    async def update_user_profile(
        self, 
        user_id: UUID, 
        update_data: dict, 
        db: Session
    ) -> Optional[UserModel]:
        """Update user profile information"""
        try:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                return None
            
            for field, value in update_data.items():
                if hasattr(user, field) and value is not None:
                    setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            
            logger.info(f"User profile updated: {user.email}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update user profile {user_id}: {e}")
            return None
    
    async def deactivate_user(self, user_id: UUID, db: Session) -> bool:
        """Deactivate a user account"""
        try:
            db.query(UserModel).filter(UserModel.id == user_id).update(
                {"is_active": False, "updated_at": datetime.utcnow()}
            )
            db.commit()
            logger.info(f"User deactivated: {user_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to deactivate user {user_id}: {e}")
            return False
```

#### **Step 2.4: Create Authentication Dependencies (1 hour)**
Create: `core/dependencies.py`
```python
"""
FastAPI dependencies for Fairdoc AI
Handles authentication, database sessions, and common dependencies
"""
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from data.database.connection_manager import get_db
from services.auth_service import auth_service
from data.repositories.auth_repository import UserModel

logger = logging.getLogger(__name__)

# HTTP Bearer token security
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> UserModel:
    """
    Dependency to get current authenticated user
    Used to protect endpoints that require authentication
    """
    token = credentials.credentials
    user = await auth_service.get_current_user(token, db)
    return user

async def get_current_active_user(
    current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    """
    Dependency to get current active user
    Ensures user account is not disabled
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    return current_user

async def get_current_superuser(
    current_user: UserModel = Depends(get_current_active_user)
) -> UserModel:
    """
    Dependency to get current superuser
    Used for admin-only endpoints
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[UserModel]:
    """
    Dependency to get current user if authenticated, None otherwise
    Used for endpoints that work for both authenticated and anonymous users
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user = await auth_service.get_current_user(token, db)
        return user
    except HTTPException:
        return None

# Database dependency (re-exported for convenience)
def get_database() -> Session:
    """Get database session dependency"""
    return get_db()
```

### **📅 Afternoon Session (4 hours): 12:00 PM - 4:00 PM**

#### **Step 2.5: Create Authentication API Routes (1.5 hours)**
Create: `api/auth/routes.py`
```python
"""
Authentication API routes for Fairdoc AI
Handles user registration, login, logout, and profile management
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from core.dependencies import get_db, get_current_active_user, get_optional_user
from services.auth_service import auth_service
from datamodels.auth_models_extended import (
    Token, LoginRequest, UserCreate, UserProfile, UserUpdate, PasswordReset
)
from data.repositories.auth_repository import UserModel

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/register", response_model=dict)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    
    **Request body:**
    ```
    {
        "email": "patient@example.com",
        "password": "securepassword123",
        "full_name": "John Doe"
    }
    ```
    """
    try:
        result = await auth_service.register_user(user_data, db)
        logger.info(f"User registration successful: {user_data.email}")
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": result["user_id"],
            "email": result["email"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login_user(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token
    
    **Request body:**
    ```
    {
        "email": "patient@example.com", 
        "password": "securepassword123",
        "remember_me": false
    }
    ```
    """
    try:
        token = await auth_service.authenticate_user(login_data, db)
        logger.info(f"User login successful: {login_data.email}")
        return token
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/logout")
async def logout_user(
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Logout current user
    Note: JWT tokens are stateless, so this is mainly for client-side cleanup
    """
    logger.info(f"User logout: {current_user.email}")
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: UserModel = Depends(get_current_active_user)
):
    """Get current user's profile information"""
    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@router.put("/profile", response_model=UserProfile)
async def update_user_profile(
    profile_data: UserUpdate,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user's profile information"""
    try:
        from data.repositories.auth_repository import AuthRepository
        auth_repo = AuthRepository()
        
        # Prepare update data (only non-None fields)
        update_data = {k: v for k, v in profile_data.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for update"
            )
        
        updated_user = await auth_repo.update_user_profile(
            current_user.id, update_data, db
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Profile updated for user: {current_user.email}")
        
        return UserProfile(
            id=updated_user.id,
            email=updated_user.email,
            full_name=updated_user.full_name,
            role=updated_user.role,
            is_verified=updated_user.is_verified,
            created_at=updated_user.created_at,
            last_login=updated_user.last_login
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.get("/verify-token")
async def verify_token(
    current_user: UserModel = Depends(get_current_active_user)
):
    """Verify if the current token is valid"""
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }

@router.post("/password-reset")
async def request_password_reset(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
):
    """
    Request password reset (mock implementation for development)
    In production, this would send an email with reset link
    """
    try:
        from data.repositories.auth_repository import AuthRepository
        auth_repo = AuthRepository()
        
        user = await auth_repo.get_user_by_email(reset_data.email, db)
        
        if user:
            # In production, send reset email here
            logger.info(f"Password reset requested for: {reset_data.email}")
            
        # Always return success (don't reveal if email exists)
        return {
            "success": True,
            "message": "If the email exists, a password reset link has been sent"
        }
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return {
            "success": True,
            "message": "If the email exists, a password reset link has been sent"
        }

@router.get("/me/stats")
async def get_user_stats(
    current_user: UserModel = Depends(get_current_active_user)
):
    """Get current user's statistics and activity"""
    # Mock statistics for now
    return {
        "user_id": current_user.id,
        "total_assessments": 0,  # Will implement when we add medical features
        "last_assessment": None,
        "account_age_days": (datetime.utcnow() - current_user.created_at).days,
        "role": current_user.role,
        "is_verified": current_user.is_verified
    }
```

#### **Step 2.6: Update Main App with Auth Routes (30 minutes)**
Update: `app.py` (add auth router)
```python
# Add this import near the top
from api.auth.routes import router as auth_router

# Add this line after the existing router includes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
```

#### **Step 2.7: Create Database Migration for Users Table (30 minutes)**
Create: `data/database/migrations/create_users_table.py`
```python
"""
Database migration to create users table with proper indexes
"""
from sqlalchemy import text
from data.database.connection_manager import db_manager

def upgrade():
    """Create users table and related tables"""
    
    engine = db_manager.engine
    
    with engine.connect() as conn:
        # Drop existing tables if they exist (for clean setup)
        conn.execute(text("""
            DROP TABLE IF EXISTS user_sessions CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
        """))
        
        # Create users table
        conn.execute(text("""
            CREATE TABLE users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                role VARCHAR(50) DEFAULT 'patient',
                is_active BOOLEAN DEFAULT true,
                is_verified BOOLEAN DEFAULT false,
                is_superuser BOOLEAN DEFAULT false,
                last_login TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))
        
        # Create indexes
        conn.execute(text("""
            CREATE INDEX idx_users_email ON users(email);
            CREATE INDEX idx_users_role ON users(role);
            CREATE INDEX idx_users_active ON users(is_active);
            CREATE INDEX idx_users_created_at ON users(created_at);
        """))
        
        # Create user sessions table
        conn.execute(text("""
            CREATE TABLE user_sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                session_token VARCHAR(255) UNIQUE NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                ip_address INET,
                user_agent TEXT
            );
        """))
        
        # Create session indexes
        conn.execute(text("""
            CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
            CREATE INDEX idx_sessions_token ON user_sessions(session_token);
            CREATE INDEX idx_sessions_expires ON user_sessions(expires_at);
        """))
        
        # Create default admin user
        conn.execute(text("""
            INSERT INTO users (email, hashed_password, full_name, role, is_superuser, is_verified) VALUES 
            ('admin@fairdoc.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeNa5oFb/aZhp.XdO', 'System Admin', 'admin', true, true);
        """))
        
        conn.commit()
        print("✅ Users table created successfully")

if __name__ == "__main__":
    upgrade()
```

#### **Step 2.8: Test Authentication System (1.5 hours)**
Create: `tests/test_auth.py`
```python
"""
Test authentication system
"""
import requests
import json

# Base URL for API
BASE_URL = "http://localhost:8000/api/v1"

def test_user_registration():
    """Test user registration"""
    print("🧪 Testing user registration...")
    
    url = f"{BASE_URL}/auth/register"
    data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    print("✅ Registration test passed")
    
    return result

def test_user_login():
    """Test user login"""
    print("\n🧪 Testing user login...")
    
    url = f"{BASE_URL}/auth/login"
    data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "remember_me": False
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert "access_token" in result
    print("✅ Login test passed")
    
    return result["access_token"]

def test_protected_endpoint(token):
    """Test accessing protected endpoint"""
    print("\n🧪 Testing protected endpoint...")
    
    url = f"{BASE_URL}/auth/profile"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert "email" in result
    print("✅ Protected endpoint test passed")

def test_token_verification(token):
    """Test token verification"""
    print("\n🧪 Testing token verification...")
    
    url = f"{BASE_URL}/auth/verify-token"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert result["valid"] == True
    print("✅ Token verification test passed")

def test_unauthorized_access():
    """Test unauthorized access"""
    print("\n🧪 Testing unauthorized access...")
    
    url = f"{BASE_URL}/auth/profile"
    # No Authorization header
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    
    assert response.status_code == 401
    print("✅ Unauthorized access test passed")

if __name__ == "__main__":
    print("🚀 Starting authentication tests...")
    
    # Run tests
    test_user_registration()
    token = test_user_login()
    test_protected_endpoint(token)
    test_token_verification(token)
    test_unauthorized_access()
    
    print("\n🎉 All authentication tests passed!")
```

### **📅 Evening Session (4 hours): 6:00 PM - 10:00 PM**

#### **Step 2.9: Create Frontend Authentication Setup (2 hours)**
Create: `frontend/package.json`
```json
{
  "name": "fairdoc-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "axios": "^1.3.0",
    "@headlessui/react": "^1.7.0",
    "@heroicons/react": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "@vitejs/plugin-react": "^3.1.0",
    "autoprefixer": "^10.4.13",
    "postcss": "^8.4.21",
    "tailwindcss": "^3.2.6",
    "vite": "^4.1.0"
  }
}
```

Create: `frontend/src/App.jsx`
```jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { useAuth } from './hooks/useAuth';
import Login from './components/Login';
import Register from './components/Register';
import Dashboard from './components/Dashboard';
import Navbar from './components/Navbar';

// Protected Route component
function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      
        
          
          Loading...
        
      
    );
  }

  return isAuthenticated ? children : ;
}

// Public Route component (redirect to dashboard if authenticated)
function PublicRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      
        
          
          Loading...
        
      
    );
  }

  return isAuthenticated ?  : children;
}

function App() {
  return (
    
      
        
          
          
            {/* Public routes */}
            
                  
                
              } 
            />
            
                  
                
              } 
            />

            {/* Protected routes */}
            
                  
                
              } 
            />

            {/* Default redirect */}
            } />
            
            {/* 404 fallback */}
            } />
          
        
      
    
  );
}

export default App;
```

#### **Step 2.10: Create Auth Context and Hook (30 minutes)**
Create: `frontend/src/contexts/AuthContext.jsx`
```jsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../utils/api';

const AuthContext = createContext({});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  const isAuthenticated = !!token && !!user;

  // Load user data on app start
  useEffect(() => {
    const loadUser = async () => {
      if (token) {
        try {
          const userData = await authAPI.getProfile(token);
          setUser(userData);
        } catch (error) {
          console.error('Failed to load user:', error);
          logout();
        }
      }
      setLoading(false);
    };

    loadUser();
  }, [token]);

  const login = async (email, password, rememberMe = false) => {
    try {
      const response = await authAPI.login(email, password, rememberMe);
      const { access_token, user_email, user_id } = response;
      
      localStorage.setItem('token', access_token);
      setToken(access_token);
      
      // Get full user profile
      const userData = await authAPI.getProfile(access_token);
      setUser(userData);
      
      return { success: true };
    } catch (error) {
      console.error('Login failed:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed' 
      };
    }
  };

  const register = async (email, password, fullName) => {
    try {
      const response = await authAPI.register(email, password, fullName);
      return { success: true, data: response };
    } catch (error) {
      console.error('Registration failed:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Registration failed' 
      };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  const updateProfile = async (profileData) => {
    try {
      const updatedUser = await authAPI.updateProfile(profileData, token);
      setUser(updatedUser);
      return { success: true };
    } catch (error) {
      console.error('Profile update failed:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Profile update failed' 
      };
    }
  };

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    login,
    register,
    logout,
    updateProfile
  };

  return (
    
      {children}
    
  );
}

export { AuthContext };
```

Create: `frontend/src/hooks/useAuth.js`
```javascript
import { useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';

export function useAuth() {
  const context = useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
}
```

#### **Step 2.11: Create API Client (30 minutes)**
Create: `frontend/src/utils/api.js`
```javascript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API endpoints
export const authAPI = {
  login: async (email, password, rememberMe = false) => {
    const response = await api.post('/auth/login', {
      email,
      password,
      remember_me: rememberMe,
    });
    return response.data;
  },

  register: async (email, password, fullName) => {
    const response = await api.post('/auth/register', {
      email,
      password,
      full_name: fullName,
    });
    return response.data;
  },

  getProfile: async (token) => {
    const response = await api.get('/auth/profile', {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  },

  updateProfile: async (profileData, token) => {
    const response = await api.put('/auth/profile', profileData, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  },

  verifyToken: async (token) => {
    const response = await api.get('/auth/verify-token', {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  },

  logout: async () => {
    const response = await api.post('/auth/logout');
    return response.data;
  },
};

// Health API endpoints
export const healthAPI = {
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  ping: async () => {
    const response = await api.get('/ping');
    return response.data;
  },
};

export default api;
```

#### **Step 2.12: Create Login Component (1 hour)**
Create: `frontend/src/components/Login.jsx`
```jsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const result = await login(email, password, rememberMe);
    
    if (!result.success) {
      setError(result.error);
    }
    
    setLoading(false);
  };

  return (
    
      
        
          
            
              
            
          
          
            Sign in to Fairdoc AI
          
          
            Or{' '}
            
              create a new account
            
          
        
        
        
          {error && (
            
              {error}
            
          )}
          
          
            
              
                Email address
              
               setEmail(e.target.value)}
              />
            
            
              
                Password
              
               setPassword(e.target.value)}
              />
            
          

          
            
               setRememberMe(e.target.checked)}
              />
              
                Remember me
              
            

            
              
                Forgot your password?
              
            
          

          
            
              {loading ? (
                
              ) : (
                'Sign in'
              )}
            
          
        

        
          
            Demo credentials: admin@fairdoc.ai / password
          
        
      
    
  );
}

export default Login;
```

### **📈 Day 2 Success Metrics**
- ✅ User registration works via API
- ✅ User login returns JWT token
- ✅ Protected endpoints require valid token
- ✅ Frontend login/register forms functional
- ✅ Token persistence in localStorage
- ✅ Auto-redirect based on auth status

---

# **DAY 3: Basic Medical Triage System**
## **🕐 12 Hours | Goal: Simple symptom input and AI response**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 3.1: Create Medical Data Models (1 hour)**
Create: `datamodels/medical_models_extended.py`
```python
"""
Extended medical models for basic triage functionality
Builds on existing medical models with practical triage features
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from datamodels.base_models import (
    BaseEntity, ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity
)

class SymptomSeverity(str, Enum):
    """Simple symptom severity scale"""
    MILD = "mild"           # 1-3
    MODERATE = "moderate"   # 4-6  
    SEVERE = "severe"       # 7-10

class BodyPart(str, Enum):
    """Common body parts for symptom location"""
    HEAD = "head"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    BACK = "back"
    ARMS = "arms"
    LEGS = "legs"
    GENERAL = "general"

class TriageDecision(str, Enum):
    """Basic triage decisions"""
    SELF_CARE = "self_care"
    GP_APPOINTMENT = "gp_appointment"
    URGENT_CARE = "urgent_care"
    EMERGENCY = "emergency"

class SimpleSymptom(BaseModel):
    """Simple symptom for basic triage"""
    name: str = Field(..., max_length=100)
    severity: SymptomSeverity
    duration_hours: int = Field(..., ge=0, le=8760)  # Max 1 year
    body_part: BodyPart
    description: Optional[str] = Field(None, max_length=500)

class PatientInput(BaseModel):
    """Basic patient input for triage"""
    # Patient demographics
    age: int = Field(..., ge=0, le=150)
    gender: Gender
    ethnicity: Optional[Ethnicity] = None
    
    # Chief complaint
    chief_complaint: str = Field(..., max_length=500)
    symptoms: List[SimpleSymptom] = Field(..., min_items=1, max_items=10)
    
    # Basic medical history
    has_allergies: bool = Field(default=False)
    takes_medications: bool = Field(default=False)
    has_medical_conditions: bool = Field(default=False)
    
    # Contact info (optional for demo)
    phone: Optional[str] = None
    emergency_contact: Optional[str] = None

class TriageResponse(BaseModel):
    """Basic triage response"""
    assessment_id: UUID = Field(default_factory=uuid4)
    
    # Risk assessment
    risk_level: RiskLevel
    urgency_level: UrgencyLevel
    
    # Triage decision
    recommended_action: TriageDecision
    explanation: str = Field(..., max_length=1000)
    
    # AI confidence
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Next steps
    next_steps: List[str] = Field(default_factory=list)
    when_to_seek_help: List[str] = Field(default_factory=list)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
```

#### **Step 3.2: Create Basic Medical Repository (1 hour)**
Create: `data/repositories/medical_repository.py`
```python
"""
Medical repository for Fairdoc AI
Handles medical assessment data operations
"""
import logging
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from data.database.connection_manager import Base

logger = logging.getLogger(__name__)

class MedicalAssessmentModel(Base):
    """Medical assessment database model"""
    __tablename__ = "medical_assessments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Patient info
    patient_age = Column(Integer, nullable=False)
    patient_gender = Column(String(20), nullable=False)
    patient_ethnicity = Column(String(50), nullable=True)
    
    # Assessment data
    chief_complaint = Column(Text, nullable=False)
    symptoms_data = Column(JSON, nullable=False)  # Store symptoms as JSON
    
    # Results
    risk_level = Column(String(20), nullable=False)
    urgency_level = Column(String(20), nullable=False)
    recommended_action = Column(String(50), nullable=False)
    explanation = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Metadata
    processing_time_ms = Column(Float, nullable=True)
    ai_model_used = Column(String(100), default="basic_rule_engine")
    
    # Audit
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class MedicalRepository:
    """Repository for medical assessment operations"""
    
    async def create_assessment(
        self,
        patient_input: dict,
        triage_response: dict,
        db: Session
    ) -> MedicalAssessmentModel:
        """Create a new medical assessment record"""
        try:
            assessment = MedicalAssessmentModel(
                patient_age=patient_input['age'],
                patient_gender=patient_input['gender'],
                patient_ethnicity=patient_input.get('ethnicity'),
                chief_complaint=patient_input['chief_complaint'],
                symptoms_data=patient_input['symptoms'],
                risk_level=triage_response['risk_level'],
                urgency_level=triage_response['urgency_level'],
                recommended_action=triage_response['recommended_action'],
                explanation=triage_response['explanation'],
                confidence_score=triage_response['confidence_score'],
                processing_time_ms=triage_response.get('processing_time_ms')
            )
            
            db.add(assessment)
            db.commit()
            db.refresh(assessment)
            
            logger.info(f"Medical assessment created: {assessment.id}")
            return assessment
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create medical assessment: {e}")
            raise
    
    async def get_assessment_by_id(self, assessment_id: UUID, db: Session) -> Optional[MedicalAssessmentModel]:
        """Get assessment by ID"""
        try:
            assessment = db.query(MedicalAssessmentModel).filter(
                MedicalAssessmentModel.id == assessment_id
            ).first()
            return assessment
        except Exception as e:
            logger.error(f"Failed to get assessment {assessment_id}: {e}")
            return None
    
    async def get_recent_assessments(self, limit: int = 10, db: Session = None) -> List[MedicalAssessmentModel]:
        """Get recent assessments for monitoring"""
        try:
            assessments = db.query(MedicalAssessmentModel).order_by(
                MedicalAssessmentModel.created_at.desc()
            ).limit(limit).all()
            return assessments
        except Exception as e:
            logger.error(f"Failed to get recent assessments: {e}")
            return []
    
    async def get_stats(self, db: Session) -> dict:
        """Get basic statistics"""
        try:
            total_assessments = db.query(MedicalAssessmentModel).count()
            
            # Count by risk level
            risk_counts = {}
            for risk in ['low', 'moderate', 'high', 'critical']:
                count = db.query(MedicalAssessmentModel).filter(
                    MedicalAssessmentModel.risk_level == risk
                ).count()
                risk_counts[risk] = count
            
            # Count by urgency
            urgency_counts = {}
            for urgency in ['routine', 'urgent', 'emergent', 'immediate']:
                count = db.query(MedicalAssessmentModel).filter(
                    MedicalAssessmentModel.urgency_level == urgency
                ).count()
                urgency_counts[urgency] = count
            
            return {
                "total_assessments": total_assessments,
                "risk_distribution": risk_counts,
                "urgency_distribution": urgency_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
```

#### **Step 3.3: Create Basic AI Triage Service (1 hour)**
Create: `services/medical_ai_service.py`
```python
"""
Basic medical AI service for Fairdoc AI
Implements rule-based triage logic as foundation
"""
import logging
import time
from typing import Dict, List, Any
from datetime import datetime

from datamodels.medical_models_extended import (
    PatientInput, TriageResponse, TriageDecision, SymptomSeverity
)
from datamodels.base_models import RiskLevel, UrgencyLevel

logger = logging.getLogger(__name__)

class BasicTriageEngine:
    """Rule-based triage engine for initial implementation"""
    
    def __init__(self):
        # High-risk symptoms that require immediate attention
        self.red_flag_symptoms = {
            'chest pain': ['heart attack', 'cardiac'],
            'severe headache': ['stroke', 'neurological'],
            'difficulty breathing': ['respiratory emergency'],
            'severe abdominal pain': ['appendicitis', 'obstruction'],
            'loss of consciousness': ['emergency'],
            'severe bleeding': ['trauma', 'emergency']
        }
        
        # Symptom severity mappings
        self.severity_weights = {
            SymptomSeverity.MILD: 1,
            SymptomSeverity.MODERATE: 3,
            SymptomSeverity.SEVERE: 7
        }
        
        # Age risk factors
        self.age_risk_factors = {
            (0, 2): 1.5,      # Very young
            (65, 150): 1.3,   # Elderly
            (2, 65): 1.0      # Adults
        }
    
    def calculate_risk_score(self, patient_input: PatientInput) -> float:
        """Calculate basic risk score from 0.0 to 1.0"""
        base_score = 0.0
        
        # Symptom severity contribution
        for symptom in patient_input.symptoms:
            severity_weight = self.severity_weights.get(symptom.severity, 1)
            symptom_score = severity_weight / 10.0  # Normalize to 0-1
            
            # Check for red flag symptoms
            symptom_lower = symptom.name.lower()
            for red_flag in self.red_flag_symptoms:
                if red_flag in symptom_lower:
                    symptom_score *= 2.0  # Double the score for red flags
                    break
            
            # Duration factor (longer duration = slightly higher risk)
            if symptom.duration_hours > 24:
                symptom_score *= 1.2
            elif symptom.duration_hours > 168:  # 1 week
                symptom_score *= 1.5
            
            base_score += symptom_score
        
        # Age factor
        age = patient_input.age
        age_multiplier = 1.0
        for age_range, multiplier in self.age_risk_factors.items():
            if age_range[0]  tuple:
        """Determine urgency level and recommended action"""
        
        # Check for immediate red flags
        has_red_flags = False
        for symptom in patient_input.symptoms:
            symptom_lower = symptom.name.lower()
            for red_flag in self.red_flag_symptoms:
                if red_flag in symptom_lower and symptom.severity == SymptomSeverity.SEVERE:
                    has_red_flags = True
                    break
        
        if has_red_flags or risk_score > 0.8:
            return UrgencyLevel.IMMEDIATE, TriageDecision.EMERGENCY
        elif risk_score > 0.6:
            return UrgencyLevel.URGENT, TriageDecision.URGENT_CARE
        elif risk_score > 0.3:
            return UrgencyLevel.ROUTINE, TriageDecision.GP_APPOINTMENT
        else:
            return UrgencyLevel.ROUTINE, TriageDecision.SELF_CARE
    
    def generate_explanation(self, patient_input: PatientInput, risk_score: float, urgency: UrgencyLevel) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Age considerations
        if patient_input.age  65:
            explanations.append("Older adults may have increased health risks.")
        
        # Symptom analysis
        severe_symptoms = [s for s in patient_input.symptoms if s.severity == SymptomSeverity.SEVERE]
        if severe_symptoms:
            explanations.append(f"You have {len(severe_symptoms)} severe symptom(s) that need attention.")
        
        # Duration concerns
        long_duration_symptoms = [s for s in patient_input.symptoms if s.duration_hours > 72]
        if long_duration_symptoms:
            explanations.append("Some symptoms have persisted for several days.")
        
        # Medical history
        if patient_input.has_medical_conditions:
            explanations.append("Your medical history may affect symptom significance.")
        
        # Risk-based recommendations
        if risk_score > 0.7:
            explanations.append("Your symptoms suggest a potentially serious condition.")
        elif risk_score > 0.4:
            explanations.append("Your symptoms warrant medical evaluation.")
        else:
            explanations.append("Your symptoms appear manageable with basic care.")
        
        return " ".join(explanations)

class MedicalAIService:
    """Main medical AI service coordinating triage"""
    
    def __init__(self):
        self.triage_engine = BasicTriageEngine()
    
    async def assess_patient(self, patient_input: PatientInput) -> TriageResponse:
        """Main patient assessment endpoint"""
        start_time = time.time()
        
        try:
            # Calculate risk score
            risk_score = self.triage_engine.calculate_risk_score(patient_input)
            
            # Determine risk level
            if risk_score > 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score > 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score > 0.3:
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.LOW
            
            # Determine urgency and action
            urgency_level, recommended_action = self.triage_engine.determine_urgency_and_action(
                risk_score, patient_input
            )
            
            # Generate explanation
            explanation = self.triage_engine.generate_explanation(
                patient_input, risk_score, urgency_level
            )
            
            # Generate next steps
            next_steps = self._generate_next_steps(recommended_action, patient_input)
            when_to_seek_help = self._generate_warning_signs(patient_input)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            response = TriageResponse(
                risk_level=risk_level,
                urgency_level=urgency_level,
                recommended_action=recommended_action,
                explanation=explanation,
                confidence_score=min(0.95, max(0.6, 1.0 - (risk_score * 0.3))),  # Mock confidence
                next_steps=next_steps,
                when_to_seek_help=when_to_seek_help,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"Patient assessment completed: {recommended_action}, risk: {risk_level}")
            return response
            
        except Exception as e:
            logger.error(f"Patient assessment failed: {e}")
            # Return safe fallback response
            return TriageResponse(
                risk_level=RiskLevel.MODERATE,
                urgency_level=UrgencyLevel.URGENT,
                recommended_action=TriageDecision.GP_APPOINTMENT,
                explanation="Unable to complete automated assessment. Please consult a healthcare provider.",
                confidence_score=0.5,
                next_steps=["Contact your GP or NHS 111"],
                when_to_seek_help=["If symptoms worsen", "If you feel unwell"],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _generate_next_steps(self, action: TriageDecision, patient_input: PatientInput) -> List[str]:
        """Generate appropriate next steps"""
        if action == TriageDecision.EMERGENCY:
            return [
                "Call 999 immediately",
                "Go to A&E now",
                "Do not drive yourself",
                "Have someone stay with you"
            ]
        elif action == TriageDecision.URGENT_CARE:
            return [
                "Contact NHS 111 for urgent advice",
                "Visit an urgent care center",
                "Monitor symptoms closely",
                "Seek immediate help if symptoms worsen"
            ]
        elif action == TriageDecision.GP_APPOINTMENT:
            return [
                "Book an appointment with your GP",
                "Contact your practice within 24-48 hours",
                "Keep a symptom diary",
                "Rest and monitor symptoms"
            ]
        else:  # SELF_CARE
            return [
                "Rest and stay hydrated",
                "Take over-the-counter pain relief if needed",
                "Monitor symptoms for changes",
                "Contact GP if symptoms persist or worsen"
            ]
    
    def _generate_warning_signs(self, patient_input: PatientInput) -> List[str]:
        """Generate warning signs to watch for"""
        general_warnings = [
            "Symptoms become much worse",
            "You develop a high fever",
            "You have difficulty breathing",
            "You feel very unwell"
        ]
        
        # Add specific warnings based on symptoms
        symptom_warnings = []
        for symptom in patient_input.symptoms:
            if 'chest' in symptom.name.lower():
                symptom_warnings.append("Chest pain spreads to arms, neck, or jaw")
            elif 'head' in symptom.name.lower():
                symptom_warnings.append("Sudden severe headache or vision changes")
            elif 'abdom' in symptom.name.lower():
                symptom_warnings.append("Severe abdominal pain or vomiting")
        
        return general_warnings + symptom_warnings

# Global service instance
medical_ai_service = MedicalAIService()
```

#### **Step 3.4: Create Medical API Routes (1 hour)**
Create: `api/medical/routes.py`
```python
"""
Medical API routes for Fairdoc AI
Handles patient triage and medical assessments
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.dependencies import get_db, get_optional_user
from services.medical_ai_service import medical_ai_service
from data.repositories.medical_repository import MedicalRepository
from datamodels.medical_models_extended import PatientInput, TriageResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/triage", response_model=TriageResponse)
async def triage_patient(
    patient_input: PatientInput,
    db: Session = Depends(get_db),
    current_user = Depends(get_optional_user)  # Optional auth for demo
):
    """
    Perform AI-powered medical triage assessment
    
    **Request Example:**
    ```
    {
        "age": 35,
        "gender": "female", 
        "chief_complaint": "I have chest pain and feel dizzy",
        "symptoms": [
            {
                "name": "chest pain",
                "severity": "moderate",
                "duration_hours": 2,
                "body_part": "chest",
                "description": "Sharp pain in center of chest"
            },
            {
                "name": "dizziness", 
                "severity": "mild",
                "duration_hours": 1,
                "body_part": "head"
            }
        ],
        "has_allergies": false,
        "takes_medications": true,
        "has_medical_conditions": false
    }
    ```
    """
    try:
        logger.info(f"Processing triage for patient: age {patient_input.age}, chief complaint: {patient_input.chief_complaint}")
        
        # Perform AI assessment
        triage_response = await medical_ai_service.assess_patient(patient_input)
        
        # Store assessment in database
        medical_repo = MedicalRepository()
        assessment_record = await medical_repo.create_assessment(
            patient_input=patient_input.dict(),
            triage_response=triage_response.dict(),
            db=db
        )
        
        # Update response with assessment ID
        triage_response.assessment_id = assessment_record.id
        
        logger.info(f"Triage completed: {triage_response.recommended_action}, confidence: {triage_response.confidence_score}")
        
        return triage_response
        
    except Exception as e:
        logger.error(f"Triage assessment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Triage assessment failed. Please try again or contact support."
        )

@router.get("/assessment/{assessment_id}", response_model=dict)
async def get_assessment(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_optional_user)
):
    """Get a specific medical assessment by ID"""
    try:
        from uuid import UUID
        assessment_uuid = UUID(assessment_id)
        
        medical_repo = MedicalRepository()
        assessment = await medical_repo.get_assessment_by_id(assessment_uuid, db)
        
        if not assessment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        return {
            "assessment_id": assessment.id,
            "patient_age": assessment.patient_age,
            "patient_gender": assessment.patient_gender,
            "chief_complaint": assessment.chief_complaint,
            "symptoms": assessment.symptoms_data,
            "risk_level": assessment.risk_level,
            "urgency_level": assessment.urgency_level,
            "recommended_action": assessment.recommended_action,
            "explanation": assessment.explanation,
            "confidence_score": assessment.confidence_score,
            "created_at": assessment.created_at,
            "processing_time_ms": assessment.processing_time_ms
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid assessment ID format"
        )
    except Exception as e:
        logger.error(f"Failed to get assessment {assessment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assessment"
        )

@router.get("/stats")
async def get_medical_stats(
    db: Session = Depends(get_db),
    current_user = Depends(get_optional_user)
):
    """Get basic medical assessment statistics"""
    try:
        medical_repo = MedicalRepository()
        stats = await medical_repo.get_stats(db)
        
        return {
            "statistics": stats,
            "generated_at": datetime.utcnow(),
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Failed to get medical stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )

@router.get("/recent-assessments")
async def get_recent_assessments(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user = Depends(get_optional_user)
):
    """Get recent medical assessments (for monitoring dashboard)"""
    try:
        medical_repo = MedicalRepository()
        assessments = await medical_repo.get_recent_assessments(limit, db)
        
        # Convert to response format (anonymized)
        recent_data = []
        for assessment in assessments:
            recent_data.append({
                "assessment_id": assessment.id,
                "age_group": f"{(assessment.patient_age // 10) * 10}s",  # Anonymize age
                "gender": assessment.patient_gender,
                "risk_level": assessment.risk_level,
                "urgency_level": assessment.urgency_level,
                "recommended_action": assessment.recommended_action,
                "confidence_score": assessment.confidence_score,
                "created_at": assessment.created_at,
                "processing_time_ms": assessment.processing_time_ms
            })
        
        return {
            "recent_assessments": recent_data,
            "total_count": len(recent_data),
            "retrieved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent assessments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent assessments"
        )

@router.post("/quick-check")
async def quick_symptom_check(
    symptom_name: str,
    severity: str,
    age: int,
    db: Session = Depends(get_db)
):
    """Quick symptom checker for single symptoms"""
    try:
        from datamodels.medical_models_extended import SimpleSymptom, SymptomSeverity, BodyPart
        from datamodels.base_models import Gender
        
        # Create minimal patient input
        quick_input = PatientInput(
            age=age,
            gender=Gender.UNKNOWN,  # Default for quick check
            chief_complaint=f"Quick check for {symptom_name}",
            symptoms=[
                SimpleSymptom(
                    name=symptom_name,
                    severity=SymptomSeverity(severity),
                    duration_hours=1,  # Assume recent
                    body_part=BodyPart.GENERAL
                )
            ]
        )
        
        # Get quick assessment
        response = await medical_ai_service.assess_patient(quick_input)
        
        return {
            "symptom": symptom_name,
            "risk_level": response.risk_level,
            "recommended_action": response.recommended_action,
            "quick_advice": response.explanation,
            "confidence": response.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Quick check failed for {symptom_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quick symptom check failed"
        )
```

### **📅 Afternoon Session (4 hours): 12:00 PM - 4:00 PM**

#### **Step 3.5: Update Main App with Medical Routes (30 minutes)**
Update: `app.py`
```python
# Add this import
from api.medical.routes import router as medical_router

# Add this line after other router includes
app.include_router(medical_router, prefix="/api/v1/medical", tags=["Medical Triage"])
```

#### **Step 3.6: Create Database Migration for Medical Tables (30 minutes)**
Create: `data/database/migrations/create_medical_tables.py`
```python
"""
Database migration for medical assessment tables
"""
from sqlalchemy import text
from data.database.connection_manager import db_manager

def upgrade():
    """Create medical assessment tables"""
    
    engine = db_manager.engine
    
    with engine.connect() as conn:
        # Drop existing table if exists
        conn.execute(text("DROP TABLE IF EXISTS medical_assessments CASCADE;"))
        
        # Create medical assessments table
        conn.execute(text("""
            CREATE TABLE medical_assessments (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                patient_age INTEGER NOT NULL,
                patient_gender VARCHAR(20) NOT NULL,
                patient_ethnicity VARCHAR(50),
                chief_complaint TEXT NOT NULL,
                symptoms_data JSONB NOT NULL,
                risk_level VARCHAR(20) NOT NULL,
                urgency_level VARCHAR(20) NOT NULL,
                recommended_action VARCHAR(50) NOT NULL,
                explanation TEXT NOT NULL,
                confidence_score FLOAT NOT NULL,
                processing_time_ms FLOAT,
                ai_model_used VARCHAR(100) DEFAULT 'basic_rule_engine',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))
        
        # Create indexes for performance
        conn.execute(text("""
            CREATE INDEX idx_medical_assessments_created_at ON medical_assessments(created_at);
            CREATE INDEX idx_medical_assessments_risk_level ON medical_assessments(risk_level);
            CREATE INDEX idx_medical_assessments_urgency_level ON medical_assessments(urgency_level);
            CREATE INDEX idx_medical_assessments_age ON medical_assessments(patient_age);
            CREATE INDEX idx_medical_assessments_gender ON medical_assessments(patient_gender);
        """))
        
        conn.commit()
        print("✅ Medical assessment tables created successfully")

if __name__ == "__main__":
    upgrade()
```

#### **Step 3.7: Create Frontend Triage Components (2 hours)**
Create: `frontend/src/components/SymptomForm.jsx`
```jsx
import React, { useState } from 'react';

const SYMPTOM_SEVERITIES = [
  { value: 'mild', label: 'Mild (1-3)', description: 'Slight discomfort' },
  { value: 'moderate', label: 'Moderate (4-6)', description: 'Noticeable discomfort' },
  { value: 'severe', label: 'Severe (7-10)', description: 'Significant pain/discomfort' }
];

const BODY_PARTS = [
  { value: 'head', label: 'Head' },
  { value: 'chest', label: 'Chest' },
  { value: 'abdomen', label: 'Abdomen' },
  { value: 'back', label: 'Back' },
  { value: 'arms', label: 'Arms' },
  { value: 'legs', label: 'Legs' },
  { value: 'general', label: 'General/Whole body' }
];

function SymptomForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    chief_complaint: '',
    symptoms: [{ name: '', severity: '', duration_hours: '', body_part: '', description: '' }],
    has_allergies: false,
    takes_medications: false,
    has_medical_conditions: false,
    phone: '',
    emergency_contact: ''
  });

  const [errors, setErrors] = useState({});

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const handleSymptomChange = (index, field, value) => {
    const newSymptoms = [...formData.symptoms];
    newSymptoms[index] = {
      ...newSymptoms[index],
      [field]: value
    };
    setFormData(prev => ({
      ...prev,
      symptoms: newSymptoms
    }));
  };

  const addSymptom = () => {
    if (formData.symptoms.length  ({
        ...prev,
        symptoms: [...prev.symptoms, { name: '', severity: '', duration_hours: '', body_part: '', description: '' }]
      }));
    }
  };

  const removeSymptom = (index) => {
    if (formData.symptoms.length > 1) {
      const newSymptoms = formData.symptoms.filter((_, i) => i !== index);
      setFormData(prev => ({
        ...prev,
        symptoms: newSymptoms
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    // Required fields
    if (!formData.age || formData.age  150) {
      newErrors.age = 'Please enter a valid age between 0-150';
    }
    
    if (!formData.gender) {
      newErrors.gender = 'Please select your gender';
    }
    
    if (!formData.chief_complaint.trim()) {
      newErrors.chief_complaint = 'Please describe your main concern';
    }

    // Validate symptoms
    formData.symptoms.forEach((symptom, index) => {
      if (!symptom.name.trim()) {
        newErrors[`symptom_${index}_name`] = 'Symptom name is required';
      }
      if (!symptom.severity) {
        newErrors[`symptom_${index}_severity`] = 'Please select severity';
      }
      if (!symptom.duration_hours || symptom.duration_hours  {
    e.preventDefault();
    
    if (validateForm()) {
      // Convert duration to numbers
      const processedData = {
        ...formData,
        age: parseInt(formData.age),
        symptoms: formData.symptoms.map(symptom => ({
          ...symptom,
          duration_hours: parseInt(symptom.duration_hours)
        }))
      };
      
      onSubmit(processedData);
    }
  };

  return (
    
      Medical Triage Assessment
      
      
        {/* Basic Information */}
        
          Basic Information
          
          
            
              
                Age *
              
               handleInputChange('age', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.age ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder="Enter your age"
              />
              {errors.age && {errors.age}}
            
            
            
              
                Gender *
              
               handleInputChange('gender', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.gender ? 'border-red-500' : 'border-gray-300'
                }`}
              >
                Select gender
                Male
                Female
                Other
                Prefer not to say
              
              {errors.gender && {errors.gender}}
            
          
        

        {/* Chief Complaint */}
        
          
            What is your main concern today? *
          
           handleInputChange('chief_complaint', e.target.value)}
            rows={3}
            maxLength={500}
            className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              errors.chief_complaint ? 'border-red-500' : 'border-gray-300'
            }`}
            placeholder="Please describe your main symptoms or concern..."
          />
          
            {formData.chief_complaint.length}/500 characters
          
          {errors.chief_complaint && {errors.chief_complaint}}
        

        {/* Symptoms */}
        
          Symptoms
          
          {formData.symptoms.map((symptom, index) => (
            
              
                Symptom {index + 1}
                {formData.symptoms.length > 1 && (
                   removeSymptom(index)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Remove
                  
                )}
              
              
              
                
                  
                    Symptom Name *
                  
                   handleSymptomChange(index, 'name', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                      errors[`symptom_${index}_name`] ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="e.g., headache, chest pain"
                  />
                  {errors[`symptom_${index}_name`] && (
                    {errors[`symptom_${index}_name`]}
                  )}
                
                
                
                  
                    Severity *
                  
                   handleSymptomChange(index, 'severity', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                      errors[`symptom_${index}_severity`] ? 'border-red-500' : 'border-gray-300'
                    }`}
                  >
                    Select severity
                    {SYMPTOM_SEVERITIES.map(severity => (
                      
                        {severity.label} - {severity.description}
                      
                    ))}
                  
                  {errors[`symptom_${index}_severity`] && (
                    {errors[`symptom_${index}_severity`]}
                  )}
                
                
                
                  
                    Duration (hours) *
                  
                   handleSymptomChange(index, 'duration_hours', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                      errors[`symptom_${index}_duration`] ? 'border-red-500' : 'border-gray-300'
                    }`}
                    placeholder="How many hours"
                  />
                  {errors[`symptom_${index}_duration`] && (
                    {errors[`symptom_${index}_duration`]}
                  )}
                
                
                
                  
                    Body Part *
                  
                   handleSymptomChange(index, 'body_part', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                      errors[`symptom_${index}_body_part`] ? 'border-red-500' : 'border-gray-300'
                    }`}
                  >
                    Select body part
                    {BODY_PARTS.map(part => (
                      
                        {part.label}
                      
                    ))}
                  
                  {errors[`symptom_${index}_body_part`] && (
                    {errors[`symptom_${index}_body_part`]}
                  )}
                
              
              
              
                
                  Additional Description
                
                 handleSymptomChange(index, 'description', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Any additional details about this symptom"
                />
              
            
          ))}
          
          {formData.symptoms.length 
              + Add Another Symptom
            
          )}
        

        {/* Medical History */}
        
          Medical History
          
          
            
               handleInputChange('has_allergies', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              I have known allergies
            
            
            
               handleInputChange('takes_medications', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              I currently take medications
            
            
            
               handleInputChange('has_medical_conditions', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              I have existing medical conditions
            
          
        

        {/* Contact Information */}
        
          Contact Information (Optional)
          
          
            
              
                Phone Number
              
               handleInputChange('phone', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Your phone number"
              />
            
            
            
              
                Emergency Contact
              
               handleInputChange('emergency_contact', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Emergency contact person"
              />
            
          
        

        {/* Submit Button */}
        
          
            {loading ? (
              
                
                Analyzing Symptoms...
              
            ) : (
              'Get Medical Assessment'
            )}
          
        
      
    
  );
}

export default SymptomForm;
```

Create: `frontend/src/components/TriageResult.jsx`
```jsx
import React from 'react';
import { CheckCircleIcon, ExclamationTriangleIcon, XCircleIcon } from '@heroicons/react/24/outline';

const RISK_LEVEL_CONFIG = {
  low: {
    color: 'green',
    bgColor: 'bg-green-50',
    textColor: 'text-green-800',
    borderColor: 'border-green-200',
    icon: CheckCircleIcon,
    label: 'Low Risk'
  },
  moderate: {
    color: 'yellow',
    bgColor: 'bg-yellow-50',
    textColor: 'text-yellow-800',
    borderColor: 'border-yellow-200',
    icon: ExclamationTriangleIcon,
    label: 'Moderate Risk'
  },
  high: {
    color: 'orange',
    bgColor: 'bg-orange-50',
    textColor: 'text-orange-800',
    borderColor: 'border-orange-200',
    icon: ExclamationTriangleIcon,
    label: 'High Risk'
  },
  critical: {
    color: 'red',
    bgColor: 'bg-red-50',
    textColor: 'text-red-800',
    borderColor: 'border-red-200',
    icon: XCircleIcon,
    label: 'Critical Risk'
  }
};

const ACTION_CONFIG = {
  self_care: {
    color: 'green',
    label: 'Self Care',
    description: 'You can manage this at home with self-care measures'
  },
  gp_appointment: {
    color: 'blue',
    label: 'GP Appointment',
    description: 'You should book an appointment with your GP'
  },
  urgent_care: {
    color: 'orange',
    label: 'Urgent Care',
    description: 'You need urgent medical attention'
  },
  emergency: {
    color: 'red',
    label: 'Emergency',
    description: 'This is a medical emergency - seek immediate help'
  }
};

function TriageResult({ result, onNewAssessment }) {
  const riskConfig = RISK_LEVEL_CONFIG[result.risk_level] || RISK_LEVEL_CONFIG.moderate;
  const actionConfig = ACTION_CONFIG[result.recommended_action] || ACTION_CONFIG.gp_appointment;
  const RiskIcon = riskConfig.icon;

  const confidencePercentage = Math.round(result.confidence_score * 100);

  return (
    
      {/* Header */}
      
        Assessment Results
        Based on the symptoms you provided
      

      {/* Risk Level Alert */}
      
        
          
          
            
              {riskConfig.label}
            
            
              {actionConfig.description}
            
          
          
            
              Confidence: {confidencePercentage}%
            
          
        
      

      {/* Recommended Action */}
      
        Recommended Action
        
        
          {actionConfig.label}
        
        
        
          {result.explanation}
        
      

      {/* Next Steps */}
      {result.next_steps && result.next_steps.length > 0 && (
        
          What to do next:
          
            {result.next_steps.map((step, index) => (
              
                
                  {index + 1}
                
                {step}
              
            ))}
          
        
      )}

      {/* Warning Signs */}
      {result.when_to_seek_help && result.when_to_seek_help.length > 0 && (
        
          
            
            Seek immediate help if:
          
          
            {result.when_to_seek_help.map((warning, index) => (
              
                •
                {warning}
              
            ))}
          
        
      )}

      {/* Emergency Contact */}
      {result.recommended_action === 'emergency' && (
        
          
            
            Emergency Action Required
            This requires immediate medical attention
            
              
                📞 Call 999 Now
              
              
                Or go to your nearest A&E department immediately
              
            
          
        
      )}

      {/* Assessment Details */}
      
        Assessment Details
        
          
            Assessment ID:
            {result.assessment_id}
          
          
            Processed at:
            {new Date(result.created_at).toLocaleString()}
          
          
            Processing time:
            {Math.round(result.processing_time_ms)}ms
          
        
      

      {/* Actions */}
      
        
          Start New Assessment
        
        
         window.print()}
          className="px-6 py-3 bg-gray-600 text-white font-semibold rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          Print Results
        
      

      {/* Disclaimer */}
      
        
          Disclaimer: This assessment is for informational purposes only and does not replace professional medical advice. 
          Always consult with a qualified healthcare provider for medical concerns.
        
      
    
  );
}

export default TriageResult;
```

#### **Step 3.8: Create Main Triage Page (1 hour)**
Create: `frontend/src/pages/Triage.jsx`
```jsx
import React, { useState } from 'react';
import SymptomForm from '../components/SymptomForm';
import TriageResult from '../components/TriageResult';
import api from '../utils/api';

function Triage() {
  const [currentStep, setCurrentStep] = useState('form'); // 'form', 'loading', 'result'
  const [triageResult, setTriageResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleTriageSubmit = async (formData) => {
    setLoading(true);
    setError('');
    setCurrentStep('loading');

    try {
      console.log('Submitting triage data:', formData);
      
      const response = await api.post('/medical/triage', formData);
      console.log('Triage response:', response.data);
      
      setTriageResult(response.data);
      setCurrentStep('result');
      
    } catch (err) {
      console.error('Triage submission failed:', err);
      setError(err.response?.data?.detail || 'Failed to process assessment. Please try again.');
      setCurrentStep('form');
    } finally {
      setLoading(false);
    }
  };

  const handleNewAssessment = () => {
    setCurrentStep('form');
    setTriageResult(null);
    setError('');
  };

  const renderLoadingScreen = () => (
    
      
        
        Analyzing Your Symptoms
        Our AI is processing your information...
        
        
          ✓ Reviewing symptoms and severity
          ✓ Checking medical history
          ✓ Calculating risk assessment
          ✓ Determining recommendations
        
        
        
          
            Please wait... This assessment typically takes 5-10 seconds.
          
        
      
    
  );

  return (
    
      
        
        {/* Header */}
        
          
            AI Medical Triage
          
          
            Get instant medical guidance based on your symptoms
          
          
          {/* Progress indicator */}
          
            
              
                1
              
              Symptoms
            
            
            
            
            
              
                2
              
              Analysis
            
            
            
            
            
              
                3
              
              Results
            
          
        

        {/* Error Message */}
        {error && (
          
            
              
                
                  
                    
                  
                
                
                  Assessment Error
                  {error}
                
              
            
          
        )}

        {/* Main Content */}
        {currentStep === 'form' && (
          
        )}
        
        {currentStep === 'loading' && renderLoadingScreen()}
        
        {currentStep === 'result' && triageResult && (
          
        )}

        {/* Info Section */}
        
          
            
              
                
                  
                
              
              AI-Powered Analysis
              Our AI analyzes your symptoms using medical knowledge and guidelines to provide accurate assessments.
            
            
            
              
                
                  
                
              
              Secure & Private
              Your health information is encrypted and secure. We never store personal data without consent.
            
            
            
              
                
                  
                
              
              Evidence-Based
              Recommendations based on NHS guidelines and clinical best practices for safe, reliable guidance.
            
          
        
      
    
  );
}

export default Triage;
```

### **📅 Evening Session (4 hours): 6:00 PM - 10:00 PM**

#### **Step 3.9: Add Triage Route to App (30 minutes)**
Update: `frontend/src/App.jsx`
```jsx
// Add import
import Triage from './pages/Triage';

// Add route in the Routes section
} 
/>
```

#### **Step 3.10: Update Navigation (30 minutes)**
Create: `frontend/src/components/Navbar.jsx`
```jsx
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

function Navbar() {
  const { isAuthenticated, user, logout } = useAuth();
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    
      
        
          {/* Logo */}
          
            
              🏥 Fairdoc AI
            
          

          {/* Navigation Links */}
          
            
              
                Medical Triage
              
              
              {isAuthenticated && (
                <>
                  
                    Dashboard
                  
                  
                  
                    AI Chat
                  
                
              )}
            
          

          {/* User Menu */}
          
            {isAuthenticated ? (
              
                
                  Welcome, {user?.full_name || user?.email}
                
                
                  Logout
                
              
            ) : (
              
                
                  Login
                
                
                  Sign Up
                
              
            )}
          
        
      
    
  );
}

export default Navbar;
```

#### **Step 3.11: Update API Utils (30 minutes)**
Update: `frontend/src/utils/api.js`
```javascript
// Add medical API endpoints
export const medicalAPI = {
  submitTriage: async (triageData) => {
    const response = await api.post('/medical/triage', triageData);
    return response.data;
  },

  getAssessment: async (assessmentId) => {
    const response = await api.get(`/medical/assessment/${assessmentId}`);
    return response.data;
  },

  getStats: async () => {
    const response = await api.get('/medical/stats');
    return response.data;
  },

  getRecentAssessments: async (limit = 10) => {
    const response = await api.get(`/medical/recent-assessments?limit=${limit}`);
    return response.data;
  },

  quickCheck: async (symptomName, severity, age) => {
    const response = await api.post('/medical/quick-check', {
      symptom_name: symptomName,
      severity: severity,
      age: age
    });
    return response.data;
  }
};
```

#### **Step 3.12: Create Simple Dashboard (1 hour)**
Create: `frontend/src/components/Dashboard.jsx`
```jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { medicalAPI, healthAPI } from '../utils/api';

function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [recentAssessments, setRecentAssessments] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      const [statsData, recentData, healthData] = await Promise.all([
        medicalAPI.getStats(),
        medicalAPI.getRecentAssessments(5),
        healthAPI.getHealth()
      ]);
      
      setStats(statsData.statistics);
      setRecentAssessments(recentData.recent_assessments);
      setSystemHealth(healthData);
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      
        
          
          Loading dashboard...
        
      
    );
  }

  const getRiskLevelColor = (riskLevel) => {
    const colors = {
      low: 'text-green-600 bg-green-100',
      moderate: 'text-yellow-600 bg-yellow-100',
      high: 'text-orange-600 bg-orange-100',
      critical: 'text-red-600 bg-red-100'
    };
    return colors[riskLevel] || 'text-gray-600 bg-gray-100';
  };

  return (
    
      
        
        {/* Header */}
        
          
            Welcome back{user?.full_name ? `, ${user.full_name}` : ''}!
          
          
            Here's what's happening with Fairdoc AI today.
          
        

        {/* Quick Actions */}
        
          
            
              
                
                  
                
              
              
                Start Medical Triage
                Get AI-powered health assessment
              
            
          

          
            
              
                
                  
                
              
              
                AI Health Chat
                Chat with medical AI assistant
              
            
          

          
            
              
                
                  
                
              
              
                System Analytics
                View system performance
              
            
          
        

        {/* Statistics Cards */}
        {stats && (
          
            
              
                
                  
                    
                      
                    
                  
                
                
                  Total Assessments
                  {stats.total_assessments}
                
              
            

            
              
                
                  
                    
                      
                    
                  
                
                
                  Low Risk Cases
                  {stats.risk_distribution?.low || 0}
                
              
            

            
              
                
                  
                    
                      
                    
                  
                
                
                  High Risk Cases
                  {stats.risk_distribution?.high || 0}
                
              
            

            
              
                
                  
                    
                      
                    
                  
                
                
                  Critical Cases
                  {stats.risk_distribution?.critical || 0}
                
              
            
          
        )}

        {/* Recent Assessments */}
        
          
            Recent Assessments
          
          
          {recentAssessments.length > 0 ? (
            
              
                
                  
                    
                      Patient
                    
                    
                      Risk Level
                    
                    
                      Action
                    
                    
                      Confidence
                    
                    
                      Time
                    
                  
                
                
                  {recentAssessments.map((assessment) => (
                    
                      
                        {assessment.age_group}, {assessment.gender}
                      
                      
                        
                          {assessment.risk_level}
                        
                      
                      
                        {assessment.recommended_action.replace('_', ' ')}
                      
                      
                        {Math.round(assessment.confidence_score * 100)}%
                      
                      
                        {new Date(assessment.created_at).toLocaleString()}
                      
                    
                  ))}
                
              
            
          ) : (
            
              No assessments yet. Start your first assessment
            
          )}
        

        {/* System Status */}
        {systemHealth && (
          
            System Status
            
              
                
                API: {systemHealth.status}
              
              
                
                Database: {systemHealth.services?.database || 'unknown'}
              
              
                Uptime: {Math.round(systemHealth.uptime_seconds / 60)} minutes
              
            
          
        )}
      
    
  );
}

export default Dashboard;
```

#### **Step 3.13: Run Database Migration (30 minutes)**
```bash
# In backend directory
cd Fairdoc/backend

# Run the migration
python data/database/migrations/create_medical_tables.py

# Verify tables were created
docker exec -it fairdoc_postgres psql -U fairdoc -d fairdoc_dev -c "\dt"

# Should show medical_assessments table
```

#### **Step 3.14: Test Complete Triage System (1.5 hours)**
```bash
# Start all services
./scripts/dev-start.sh

# In another terminal, install frontend dependencies
cd Fairdoc/frontend
npm install

# Start frontend
npm run dev

# Test the complete flow:
# 1. Navigate to http://localhost:3000/triage
# 2. Fill out symptom form
# 3. Submit and verify AI response
# 4. Check database for stored assessment
```

```python
"""
End-to-end test for triage flow
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_complete_triage_flow():
    """Test complete triage flow from symptom input to assessment result"""
    print("🧪 Testing complete triage flow...")
    
    # Test data - chest pain case
    triage_data = {
        "age": 45,
        "gender": "female",
        "chief_complaint": "I have severe chest pain that started 2 hours ago",
        "symptoms": [
            {
                "name": "chest pain",
                "severity": "severe",
                "duration_hours": 2,
                "body_part": "chest",
                "description": "Sharp pain in center of chest, radiating to left arm"
            },
            {
                "name": "nausea",
                "severity": "moderate", 
                "duration_hours": 1,
                "body_part": "general",
                "description": "Feeling sick to stomach"
            }
        ],
        "has_allergies": false,
        "takes_medications": true,
        "has_medical_conditions": true
    }
    
    # Submit triage assessment
    response = requests.post(f"{BASE_URL}/medical/triage", json=triage_data)
    print(f"Triage Status: {response.status_code}")
    
    assert response.status_code == 200
    result = response.json()
    
    # Verify response structure
    assert "assessment_id" in result
    assert "risk_level" in result
    assert "urgency_level" in result
    assert "recommended_action" in result
    assert "confidence_score" in result
    assert "explanation" in result
    
    print(f"✅ Triage assessment completed")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Recommended Action: {result['recommended_action']}")
    print(f"   Confidence: {result['confidence_score']:.2f}")
    
    # Test retrieval of assessment
    assessment_id = result["assessment_id"]
    get_response = requests.get(f"{BASE_URL}/medical/assessment/{assessment_id}")
    assert get_response.status_code == 200
    
    print("✅ Assessment retrieval test passed")
    
    return result

def test_quick_symptom_check():
    """Test quick symptom checker"""
    print("\n🧪 Testing quick symptom check...")
    
    response = requests.post(f"{BASE_URL}/medical/quick-check", params={
        "symptom_name": "headache",
        "severity": "mild", 
        "age": 30
    })
    
    assert response.status_code == 200
    result = response.json()
    
    assert "risk_level" in result
    assert "recommended_action" in result
    print("✅ Quick symptom check test passed")

def test_medical_stats():
    """Test medical statistics endpoint"""
    print("\n🧪 Testing medical statistics...")
    
    response = requests.get(f"{BASE_URL}/medical/stats")
    assert response.status_code == 200
    
    result = response.json()
    assert "statistics" in result
    print("✅ Medical statistics test passed")

def test_recent_assessments():
    """Test recent assessments endpoint"""
    print("\n🧪 Testing recent assessments...")
    
    response = requests.get(f"{BASE_URL}/medical/recent-assessments?limit=5")
    assert response.status_code == 200
    
    result = response.json()
    assert "recent_assessments" in result
    print("✅ Recent assessments test passed")

if __name__ == "__main__":
    print("🚀 Starting triage flow tests...")
    
    # Run complete test suite
    test_complete_triage_flow()
    test_quick_symptom_check()
    test_medical_stats()
    test_recent_assessments()
    
    print("\n🎉 All triage flow tests passed!")
```

### **📈 Day 3 Success Metrics**
- ✅ `POST /api/v1/medical/triage` accepts symptom data and returns assessment
- ✅ Database stores medical assessments correctly
- ✅ Frontend triage form submits and displays results
- ✅ All test cases pass
- ✅ Live reload continues working

---

# **DAY 4: Real-Time Chat Interface** 💬
## **🕐 12 Hours | Goal: WebSocket chat with context preservation**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 4.1: Create Chat Models Extended (1 hour)**
Create: `datamodels/chat_models_extended.py`
```python
"""
Extended chat models for real-time medical conversations
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from datamodels.base_models import BaseEntity, ValidationMixin, MetadataMixin
from datamodels.chat_models import MessageType, ChatMessage  # Import existing

class ChatSessionStatus(str, Enum):
    """Chat session status"""
    ACTIVE = "active"
    WAITING = "waiting"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class AgentType(str, Enum):
    """Types of chat agents"""
    TRIAGE_BOT = "triage_bot"
    MEDICAL_AI = "medical_ai"
    HUMAN_DOCTOR = "human_doctor"
    ESCALATION_BOT = "escalation_bot"

class MessageIntent(str, Enum):
    """Intent classification for messages"""
    SYMPTOM_REPORT = "symptom_report"
    PAIN_SCALE = "pain_scale"
    MEDICAL_HISTORY = "medical_history"
    QUESTION = "question"
    EMERGENCY = "emergency"
    FOLLOWUP = "followup"
    GENERAL = "general"

class ChatSession(BaseEntity, ValidationMixin, MetadataMixin):
    """Real-time chat session with context management"""
    
    session_id: UUID = Field(default_factory=uuid4)
    patient_id: Optional[UUID] = None
    
    # Session details
    status: ChatSessionStatus = Field(default=ChatSessionStatus.ACTIVE)
    current_agent: AgentType = Field(default=AgentType.TRIAGE_BOT)
    
    # Patient context
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    patient_ethnicity: Optional[str] = None
    
    # Session tracking
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Conversation state
    message_count: int = Field(default=0, ge=0)
    assessment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    collected_symptoms: List[str] = Field(default_factory=list)
    
    # AI state
    current_model: Optional[str] = None
    context_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    
    # Escalation tracking
    escalation_requested: bool = Field(default=False)
    escalation_reason: Optional[str] = None
    human_takeover: bool = Field(default=False)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        self.update_timestamp()
    
    def calculate_duration(self) -> timedelta:
        """Calculate session duration"""
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at

class RealTimeMessage(BaseEntity, ValidationMixin):
    """Real-time chat message with AI processing"""
    
    message_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    
    # Message content
    message_type: MessageType
    content: str = Field(..., max_length=2000)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Sender information
    sender_type: Literal["patient", "ai", "doctor"] = "patient"
    sender_id: Optional[UUID] = None
    
    # AI processing
    intent: Optional[MessageIntent] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    entities_extracted: Dict[str, Any] = Field(default_factory=dict)
    sentiment: Optional[str] = None
    urgency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Context management
    context_used: bool = Field(default=False)
    response_time_ms: Optional[float] = None
    processing_cost: Optional[float] = Field(None, ge=0.0)
    
    # Message timing
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None

class ChatResponse(BaseModel):
    """AI-generated chat response"""
    
    response_id: UUID = Field(default_factory=uuid4)
    message_id: UUID
    session_id: UUID
    
    # Response content
    text: str = Field(..., max_length=1000)
    quick_replies: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    
    # AI metadata
    model_used: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    
    # Medical context
    next_questions: List[str] = Field(default_factory=list)
    assessment_complete: bool = Field(default=False)
    escalation_needed: bool = Field(default=False)
    
    # Bias monitoring
    bias_score: float = Field(default=0.0, ge=0.0, le=1.0)
    bias_warning: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    
    type: Literal["message", "typing", "status", "error"] = "message"
    session_id: UUID
    content: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Client information
    client_id: Optional[str] = None
    user_agent: Optional[str] = None
```

#### **Step 4.2: Create WebSocket Manager (1 hour)**
Create: `core/websocket_manager.py`
```python
"""
WebSocket connection manager for real-time chat
"""
import json
import logging
from typing import Dict, List, Optional, Set
from uuid import UUID
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time chat"""
    
    def __init__(self):
        # Active connections by session ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Session to user mapping
        self.session_users: Dict[str, UUID] = {}
        
        # User to sessions mapping (for multiple devices)
        self.user_sessions: Dict[UUID, Set[str]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[UUID] = None):
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Store connection
            self.active_connections[session_id] = websocket
            
            # Track user if provided
            if user_id:
                self.session_users[session_id] = user_id
                
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = set()
                self.user_sessions[user_id].add(session_id)
            
            # Store metadata
            self.connection_metadata[session_id] = {
                "connected_at": datetime.utcnow(),
                "user_id": user_id,
                "message_count": 0,
                "last_activity": datetime.utcnow()
            }
            
            logger.info(f"WebSocket connected: session={session_id}, user={user_id}")
            
            # Send connection confirmation
            await self.send_personal_message({
                "type": "status",
                "message": "Connected to Fairdoc AI",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }, session_id)
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
    
    def disconnect(self, session_id: str):
        """Disconnect a WebSocket connection"""
        try:
            # Get user ID before removing
            user_id = self.session_users.get(session_id)
            
            # Remove connection
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            
            # Clean up user tracking
            if user_id and user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            if session_id in self.session_users:
                del self.session_users[session_id]
            
            # Clean up metadata
            if session_id in self.connection_metadata:
                metadata = self.connection_metadata[session_id]
                duration = datetime.utcnow() - metadata["connected_at"]
                logger.info(f"WebSocket disconnected: session={session_id}, duration={duration}, messages={metadata['message_count']}")
                del self.connection_metadata[session_id]
            
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                await websocket.send_text(json.dumps(message, default=str))
                
                # Update activity
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]["last_activity"] = datetime.utcnow()
                    self.connection_metadata[session_id]["message_count"] += 1
                
            except Exception as e:
                logger.error(f"Failed to send message to session {session_id}: {e}")
                # Remove broken connection
                self.disconnect(session_id)
    
    async def send_to_user(self, message: dict, user_id: UUID):
        """Send message to all sessions of a user"""
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id].copy():
                await self.send_personal_message(message, session_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected sessions"""
        disconnected_sessions = []
        
        for session_id in self.active_connections.copy():
            try:
                await self.send_personal_message(message, session_id)
            except Exception as e:
                logger.error(f"Failed to broadcast to session {session_id}: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)
    
    async def send_typing_indicator(self, session_id: str, is_typing: bool = True):
        """Send typing indicator"""
        await self.send_personal_message({
            "type": "typing",
            "is_typing": is_typing,
            "timestamp": datetime.utcnow().isoformat()
        }, session_id)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_user_connection_count(self, user_id: UUID) -> int:
        """Get number of connections for a specific user"""
        return len(self.user_sessions.get(user_id, set()))
    
    def is_user_connected(self, user_id: UUID) -> bool:
        """Check if user has any active connections"""
        return user_id in self.user_sessions and len(self.user_sessions[user_id]) > 0
    
    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        """Get metadata for a session"""
        return self.connection_metadata.get(session_id)

# Global connection manager instance
manager = ConnectionManager()
```

#### **Step 4.3: Create Chat Repository (1 hour)**
Create: `data/repositories/chat_repository.py`
```python
"""
Chat repository for storing conversation data
"""
import logging
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from data.database.connection_manager import Base

logger = logging.getLogger(__name__)

class ChatSessionModel(Base):
    """Chat session database model"""
    __tablename__ = "chat_sessions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid4)
    patient_id = Column(PG_UUID(as_uuid=True), nullable=True)
    
    # Session details
    status = Column(String(20), default="active")
    current_agent = Column(String(50), default="triage_bot")
    
    # Patient context
    patient_age = Column(Integer, nullable=True)
    patient_gender = Column(String(20), nullable=True)
    patient_ethnicity = Column(String(50), nullable=True)
    
    # Session tracking
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_activity = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Conversation state
    message_count = Column(Integer, default=0)
    assessment_progress = Column(Float, default=0.0)
    collected_symptoms = Column(JSON, default=list)
    
    # AI state
    current_model = Column(String(100), nullable=True)
    context_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    
    # Escalation tracking
    escalation_requested = Column(Boolean, default=False)
    escalation_reason = Column(Text, nullable=True)
    human_takeover = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatMessageModel(Base):
    """Chat message database model"""
    __tablename__ = "chat_messages"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), nullable=False)
    
    # Message content
    message_type = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    attachments = Column(JSON, default=list)
    
    # Sender information
    sender_type = Column(String(20), default="patient")
    sender_id = Column(PG_UUID(as_uuid=True), nullable=True)
    
    # AI processing
    intent = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    entities_extracted = Column(JSON, default=dict)
    sentiment = Column(String(20), nullable=True)
    urgency_score = Column(Float, nullable=True)
    
    # Context management
    context_used = Column(Boolean, default=False)
    response_time_ms = Column(Float, nullable=True)
    processing_cost = Column(Float, nullable=True)
    
    # Message timing
    sent_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class ChatRepository:
    """Repository for chat operations"""
    
    async def create_session(
        self,
        patient_id: Optional[UUID] = None,
        patient_age: Optional[int] = None,
        patient_gender: Optional[str] = None,
        patient_ethnicity: Optional[str] = None,
        db: Session = None
    ) -> ChatSessionModel:
        """Create a new chat session"""
        try:
            session = ChatSessionModel(
                patient_id=patient_id,
                patient_age=patient_age,
                patient_gender=patient_gender,
                patient_ethnicity=patient_ethnicity
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            logger.info(f"Chat session created: {session.session_id}")
            return session
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create chat session: {e}")
            raise
    
    async def get_session_by_id(self, session_id: UUID, db: Session) -> Optional[ChatSessionModel]:
        """Get chat session by ID"""
        try:
            session = db.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).first()
            return session
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None
    
    async def update_session_activity(self, session_id: UUID, db: Session) -> bool:
        """Update session last activity"""
        try:
            db.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).update({
                "last_activity": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update session activity {session_id}: {e}")
            return False
    
    async def save_message(
        self,
        session_id: UUID,
        content: str,
        message_type: str = "text",
        sender_type: str = "patient",
        sender_id: Optional[UUID] = None,
        attachments: List = None,
        db: Session = None
    ) -> ChatMessageModel:
        """Save a chat message"""
        try:
            message = ChatMessageModel(
                session_id=session_id,
                message_type=message_type,
                content=content,
                sender_type=sender_type,
                sender_id=sender_id,
                attachments=attachments or []
            )
            
            db.add(message)
            
            # Update session message count
            db.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).update({
                "message_count": ChatSessionModel.message_count + 1,
                "last_activity": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            db.commit()
            db.refresh(message)
            
            return message
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save chat message: {e}")
            raise
    
    async def get_session_messages(
        self, 
        session_id: UUID, 
        limit: int = 50, 
        offset: int = 0,
        db: Session = None
    ) -> List[ChatMessageModel]:
        """Get messages for a session"""
        try:
            messages = db.query(ChatMessageModel).filter(
                ChatMessageModel.session_id == session_id
            ).order_by(ChatMessageModel.sent_at.asc()).offset(offset).limit(limit).all()
            
            return messages
        except Exception as e:
            logger.error(f"Failed to get session messages {session_id}: {e}")
            return []
    
    async def complete_session(self, session_id: UUID, db: Session) -> bool:
        """Mark session as completed"""
        try:
            db.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session_id
            ).update({
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to complete session {session_id}: {e}")
            return False
    
    async def get_active_sessions(self, limit: int = 20, db: Session = None) -> List[ChatSessionModel]:
        """Get active chat sessions"""
        try:
            sessions = db.query(ChatSessionModel).filter(
                ChatSessionModel.status.in_(["active", "waiting"])
            ).order_by(ChatSessionModel.last_activity.desc()).limit(limit).all()
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
```

#### **Step 4.4: Create Chat Orchestrator Service (1 hour)**
Create: `services/chat_orchestrator.py`
```python
"""
Chat orchestrator service for managing AI conversations
"""
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from data.repositories.chat_repository import ChatRepository
from datamodels.chat_models_extended import (
    ChatSession, RealTimeMessage, ChatResponse, MessageIntent, AgentType
)

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Orchestrates AI-powered medical chat conversations"""
    
    def __init__(self):
        self.chat_repo = ChatRepository()
        
        # Simple rule-based responses for Day 4
        self.response_templates = {
            "greeting": [
                "Hello! I'm your AI medical assistant. How can I help you today?",
                "Hi there! I'm here to help with your health concerns. What's bothering you?",
                "Welcome to Fairdoc AI. Please describe your symptoms or health concern."
            ],
            "symptom_followup": [
                "Thank you for sharing that. Can you tell me more about when this started?",
                "I understand. On a scale of 1-10, how would you rate the severity?",
                "That's helpful information. Are there any other symptoms you're experiencing?"
            ],
            "pain_scale": [
                "On a scale of 1-10, with 10 being the worst pain imaginable, how would you rate your pain?",
                "Can you describe the type of pain? Is it sharp, dull, throbbing, or burning?",
                "Does the pain stay in one place or does it spread to other areas?"
            ],
            "emergency": [
                "This sounds like it may require immediate medical attention.",
                "Based on what you've told me, I recommend calling emergency services.",
                "Please consider going to A&E or calling 999 immediately."
            ],
            "reassurance": [
                "Thank you for providing that information. Let me ask a few more questions to help assess your situation.",
                "You're doing great providing these details. This helps me understand your condition better.",
                "I appreciate your patience. These questions help ensure you get the right care."
            ]
        }
        
        # Emergency keywords that trigger urgent responses
        self.emergency_keywords = [
            "chest pain", "can't breathe", "choking", "unconscious", "severe bleeding", 
            "suicide", "overdose", "stroke", "heart attack", "severe allergic reaction"
        ]
        
        # Symptom keywords for intent classification
        self.symptom_keywords = [
            "pain", "ache", "hurt", "fever", "nausea", "dizzy", "tired", "headache",
            "cough", "cold", "rash", "swelling", "bleeding", "vomiting"
        ]
    
    async def process_patient_message(
        self,
        session_id: UUID,
        message_content: str,
        message_type: str = "text",
        attachments: List = None,
        db = None
    ) -> ChatResponse:
        """Process incoming patient message and generate AI response"""
        
        start_time = time.time()
        
        try:
            # Get or create session
            session = await self.chat_repo.get_session_by_id(session_id, db)
            if not session:
                session = await self.chat_repo.create_session(db=db)
                session_id = session.session_id
            
            # Save patient message
            message = await self.chat_repo.save_message(
                session_id=session_id,
                content=message_content,
                message_type=message_type,
                sender_type="patient",
                attachments=attachments or [],
                db=db
            )
            
            # Classify message intent
            intent = self.classify_intent(message_content)
            
            # Check for emergency keywords
            is_emergency = self.detect_emergency(message_content)
            
            # Generate appropriate response
            if is_emergency:
                response_text = self.get_emergency_response(message_content)
                quick_replies = ["Call 999", "Go to A&E", "Contact GP urgently"]
                escalation_needed = True
            elif intent == MessageIntent.SYMPTOM_REPORT:
                response_text = self.get_symptom_followup_response(message_content, session)
                quick_replies = self.get_symptom_quick_replies(message_content)
                escalation_needed = False
            elif intent == MessageIntent.PAIN_SCALE:
                response_text = self.get_pain_scale_response(message_content)
                quick_replies = ["1-3 (Mild)", "4-6 (Moderate)", "7-8 (Severe)", "9-10 (Unbearable)"]
                escalation_needed = False
            else:
                response_text = self.get_general_response(message_content, session)
                quick_replies = ["Tell me more", "Yes", "No", "I'm not sure"]
                escalation_needed = False
            
            # Calculate processing metrics
            processing_time = (time.time() - start_time) * 1000
            
            # Create response
            response = ChatResponse(
                message_id=message.message_id,
                session_id=session_id,
                text=response_text,
                quick_replies=quick_replies,
                suggested_actions=self.get_suggested_actions(intent, is_emergency),
                model_used="rule_based_v1",
                confidence=0.8 if not is_emergency else 0.95,
                processing_time_ms=processing_time,
                tokens_used=len(message_content.split()) + len(response_text.split()),
                cost_usd=0.001,  # Mock cost
                next_questions=self.get_next_questions(intent),
                escalation_needed=escalation_needed,
                bias_score=0.02  # Mock bias score
            )
            
            # Save AI response
            await self.chat_repo.save_message(
                session_id=session_id,
                content=response_text,
                message_type="text",
                sender_type="ai",
                db=db
            )
            
            # Update session activity
            await self.chat_repo.update_session_activity(session_id, db)
            
            logger.info(f"Chat message processed: session={session_id}, intent={intent}, emergency={is_emergency}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process chat message: {e}")
            
            # Return safe fallback response
            return ChatResponse(
                message_id=UUID("00000000-0000-0000-0000-000000000000"),
                session_id=session_id,
                text="I'm sorry, I'm having trouble processing your message right now. Please try again or contact your healthcare provider if this is urgent.",
                quick_replies=["Try again", "Contact support"],
                suggested_actions=["Restart conversation", "Call NHS 111"],
                model_used="fallback",
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cost_usd=0.0,
                escalation_needed=True
            )
    
    def classify_intent(self, message: str) -> MessageIntent:
        """Classify message intent using simple keyword matching"""
        message_lower = message.lower()
        
        # Check for emergency
        if any(keyword in message_lower for keyword in self.emergency_keywords):
            return MessageIntent.EMERGENCY
        
        # Check for pain scale indicators
        if any(word in message_lower for word in ["pain", "hurt", "ache"]) and any(num in message for num in "12345678910"):
            return MessageIntent.PAIN_SCALE
        
        # Check for symptoms
        if any(keyword in message_lower for keyword in self.symptom_keywords):
            return MessageIntent.SYMPTOM_REPORT
        
        # Check for questions
        if message.strip().endswith("?") or message_lower.startswith(("what", "how", "when", "why", "where", "can", "should", "will")):
            return MessageIntent.QUESTION
        
        return MessageIntent.GENERAL
    
    def detect_emergency(self, message: str) -> bool:
        """Detect emergency situations"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.emergency_keywords)
    
    def get_emergency_response(self, message: str) -> str:
        """Generate emergency response"""
        import random
        base_responses = self.response_templates["emergency"]
        return random.choice(base_responses)
    
    def get_symptom_followup_response(self, message: str, session) -> str:
        """Generate follow-up response for symptoms"""
        import random
        if session.message_count  str:
        """Generate pain scale response"""
        import random
        return random.choice(self.response_templates["pain_scale"])
    
    def get_general_response(self, message: str, session) -> str:
        """Generate general response"""
        import random
        if session.message_count == 0:
            return random.choice(self.response_templates["greeting"])
        else:
            return random.choice(self.response_templates["reassurance"])
    
    def get_symptom_quick_replies(self, message: str) -> List[str]:
        """Get quick reply options for symptoms"""
        return [
            "It just started",
            "A few hours ago", 
            "Since yesterday",
            "For several days",
            "More than a week"
        ]
    
    def get_suggested_actions(self, intent: MessageIntent, is_emergency: bool) -> List[str]:
        """Get suggested actions based on intent"""
        if is_emergency:
            return ["Call 999", "Go to A&E", "Contact emergency services"]
        elif intent == MessageIntent.SYMPTOM_REPORT:
            return ["Continue assessment", "Get medical advice", "Monitor symptoms"]
        else:
            return ["Ask more questions", "Provide more details", "Request help"]
    
    def get_next_questions(self, intent: MessageIntent) -> List[str]:
        """Get next questions to ask"""
        if intent == MessageIntent.SYMPTOM_REPORT:
            return [
                "When did this symptom start?",
                "How severe is it on a scale of 1-10?",
                "Does anything make it better or worse?",
                "Do you have any other symptoms?"
            ]
        elif intent == MessageIntent.PAIN_SCALE:
            return [
                "Where exactly is the pain located?",
                "What type of pain is it?",
                "Does the pain spread to other areas?"
            ]
        else:
            return [
                "Can you tell me more about your concern?",
                "When did you first notice this?",
                "Have you experienced this before?"
            ]

# Global chat orchestrator instance
chat_orchestrator = ChatOrchestrator()
```

### **📅 Afternoon Session (4 hours): 12:00 PM - 4:00 PM**

#### **Step 4.5: Create WebSocket Chat Routes (1.5 hours)**
Create: `api/chat/websocket_routes.py`
```python
"""
WebSocket routes for real-time chat functionality
"""
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
from uuid import UUID, uuid4
from typing import Optional

from core.websocket_manager import manager
from core.dependencies import get_db
from services.chat_orchestrator import chat_orchestrator
from data.repositories.chat_repository import ChatRepository

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/chat/{session_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time medical chat
    
    Usage:
    ```
    const ws = new WebSocket('ws://localhost:8000/ws/chat/session-123?token=jwt-token');
    ```
    """
    
    chat_repo = ChatRepository()
    user_id = None
    
    try:
        # Validate session ID format
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            await websocket.close(code=1003, reason="Invalid session ID format")
            return
        
        # TODO: Validate JWT token if provided
        # For now, accept all connections for demo
        
        # Connect to WebSocket manager
        await manager.connect(websocket, session_id, user_id)
        
        # Get or create chat session
        session = await chat_repo.get_session_by_id(session_uuid, db)
        if not session:
            session = await chat_repo.create_session(db=db)
            session_id = str(session.session_id)
            logger.info(f"Created new chat session: {session_id}")
        
        # Send welcome message
        welcome_response = await chat_orchestrator.process_patient_message(
            session_id=session_uuid,
            message_content="Hello",
            message_type="system",
            db=db
        )
        
        await manager.send_personal_message({
            "type": "response",
            "content": {
                "text": welcome_response.text,
                "quick_replies": welcome_response.quick_replies,
                "suggested_actions": welcome_response.suggested_actions
            },
            "ai_state": {
                "confidence": welcome_response.confidence,
                "next_questions": welcome_response.next_questions,
                "assessment_progress": 0.0
            },
            "bias_monitoring": {
                "bias_score": welcome_response.bias_score,
                "bias_warning": welcome_response.bias_warning
            },
            "session_id": session_id
        }, session_id)
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                logger.info(f"Received WebSocket message: {message_data}")
                
                # Validate message structure
                if "type" not in message_data:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "Message must include 'type' field"
                    }, session_id)
                    continue
                
                message_type = message_data["type"]
                
                if message_type == "message":
                    # Process patient message
                    content = message_data.get("content", {})
                    text = content.get("text", "")
                    attachments = content.get("attachments", [])
                    
                    if not text.strip():
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Message text cannot be empty"
                        }, session_id)
                        continue
                    
                    # Send typing indicator
                    await manager.send_typing_indicator(session_id, True)
                    
                    # Process with chat orchestrator
                    response = await chat_orchestrator.process_patient_message(
                        session_id=session_uuid,
                        message_content=text,
                        message_type="text",
                        attachments=attachments,
                        db=db
                    )
                    
                    # Stop typing indicator
                    await manager.send_typing_indicator(session_id, False)
                    
                    # Send AI response
                    response_message = {
                        "type": "response",
                        "content": {
                            "text": response.text,
                            "quick_replies": response.quick_replies,
                            "suggested_actions": response.suggested_actions
                        },
                        "ai_state": {
                            "confidence": response.confidence,
                            "next_questions": response.next_questions,
                            "assessment_complete": response.assessment_complete,
                            "escalation_needed": response.escalation_needed
                        },
                        "bias_monitoring": {
                            "bias_score": response.bias_score,
                            "bias_warning": response.bias_warning,
                            "fairness_verified": response.bias_score  `session-${Date.now()}`);
  
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    scrollToBottom();
  }, [messages]);

  const connectWebSocket = () => {
    const wsUrl = `ws://localhost:8000/api/v1/ws/chat/${sessionId}`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (!isConnected) {
          connectWebSocket();
        }
      }, 3000);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
  };

  const handleWebSocketMessage = (data) => {
    console.log('Received message:', data);

    if (data.type === 'response') {
      // AI response
      const aiMessage = {
        id: Date.now(),
        type: 'ai',
        text: data.content.text,
        quickReplies: data.content.quick_replies || [],
        suggestedActions: data.content.suggested_actions || [],
        aiState: data.ai_state || {},
        biasMonitoring: data.bias_monitoring || {},
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
      
    } else if (data.type === 'typing') {
      setIsTyping(data.is_typing);
      
    } else if (data.type === 'status') {
      // System status message
      const statusMessage = {
        id: Date.now(),
        type: 'system',
        text: data.message,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, statusMessage]);
      
    } else if (data.type === 'error') {
      // Error message
      const errorMessage = {
        id: Date.now(),
        type: 'error',
        text: `Error: ${data.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      
    } else if (data.type === 'escalation') {
      // Escalation message
      const escalationMessage = {
        id: Date.now(),
        type: 'escalation',
        text: data.message,
        actions: data.actions || [],
        timestamp: new Date()
      };
      setMessages(prev => [...prev, escalationMessage]);
    }
  };

  const sendMessage = () => {
    if (!inputText.trim() || !isConnected) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputText,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);

    // Send message through WebSocket
    const messageData = {
      type: 'message',
      content: {
        text: inputText,
        attachments: []
      },
      metadata: {
        timestamp: new Date().toISOString()
      }
    };

    wsRef.current.send(JSON.stringify(messageData));
    
    // Clear input and show typing
    setInputText('');
    setIsTyping(true);
  };

  const handleQuickReply = (reply) => {
    setInputText(reply);
    setTimeout(() => sendMessage(), 100);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getMessageBubbleClass = (type) => {
    switch (type) {
      case 'user':
        return 'bg-blue-600 text-white ml-auto';
      case 'ai':
        return 'bg-gray-100 text-gray-900 mr-auto';
      case 'system':
        return 'bg-yellow-100 text-yellow-800 mx-auto text-center';
      case 'error':
        return 'bg-red-100 text-red-800 mx-auto text-center';
      case 'escalation':
        return 'bg-orange-100 text-orange-800 mx-auto text-center';
      default:
        return 'bg-gray-100 text-gray-900 mr-auto';
    }
  };

  return (
    
      {/* Header */}
      
        
          Fairdoc AI Medical Assistant
          
            
            {isConnected ? 'Connected' : 'Disconnected'}
          
        
        
          I'm here to help assess your health concerns. Please describe your symptoms.
        
      

      {/* Messages */}
      
        {messages.map((message) => (
          
            
              {message.text}
              
              {/* Quick replies for AI messages */}
              {message.type === 'ai' && message.quickReplies && message.quickReplies.length > 0 && (
                
                  {message.quickReplies.map((reply, index) => (
                     handleQuickReply(reply)}
                      className="block w-full text-left px-2 py-1 text-xs bg-white text-gray-700 rounded border hover:bg-gray-50"
                    >
                      {reply}
                    
                  ))}
                
              )}
              
              {/* Escalation actions */}
              {message.type === 'escalation' && message.actions && message.actions.length > 0 && (
                
                  {message.actions.map((action, index) => (
                    
                      {action}
                    
                  ))}
                
              )}
            
            
            
              {formatTimestamp(message.timestamp)}
            
            
            {/* AI metadata */}
            {message.type === 'ai' && message.aiState && (
              
                Confidence: {Math.round((message.aiState.confidence || 0) * 100)}%
                {message.biasMonitoring?.bias_score !== undefined && (
                  
                    Bias Score: {(message.biasMonitoring.bias_score * 100).toFixed(1)}%
                  
                )}
              
            )}
          
        ))}
        
        {/* Typing indicator */}
        {isTyping && (
          
            
              
                
                
                
              
            
          
        )}
        
        
      

      {/* Input */}
      
        
           setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Describe your symptoms or health concern..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows="2"
            disabled={!isConnected}
          />
          
            
          
        
        
        {!isConnected && (
          
            Connection lost. Attempting to reconnect...
          
        )}
      
    
  );
}

export default ChatWindow;
```

### **📅 Evening Session (4 hours): 6:00 PM - 10:00 PM**

#### **Step 4.9: Create Chat Page (1 hour)**
Create: `frontend/src/pages/Chat.jsx`
```jsx
import React from 'react';
import ChatWindow from '../components/ChatWindow';

function Chat() {
  return (
    
      
        
        {/* Header */}
        
          
            AI Medical Chat
          
          
            Real-time conversation with our medical AI assistant
          
          
          
            How it works:
            
              • Describe your symptoms in natural language
              • Answer follow-up questions to help with assessment
              • Receive instant medical guidance and recommendations
              • Get connected to human doctors when needed
            
          
        

        {/* Chat Interface */}
        

        {/* Safety Notice */}
        
          
            
              
                
              
            
            
              Important Safety Information
              
                
                  This AI assistant provides general health information and should not replace professional medical advice. 
                  If you're experiencing a medical emergency, please call 999 immediately or go to your nearest A&E department.
                
              
            
          
        

        {/* Features */}
        
          
            
              
                
              
            
            Real-time Chat
            Instant responses powered by advanced AI that understands medical terminology and symptoms.
          
          
          
            
              
                
              
            
            Bias Monitoring
            Every interaction is monitored for fairness to ensure equal treatment regardless of demographics.
          
          
          
            
              
                
              
            
            Human Escalation
            Seamless handoff to qualified doctors when human expertise is needed for complex cases.
          
        
      
    
  );
}

export default Chat;
```

#### **Step 4.10: Add Chat Route and Navigation (30 minutes)**
Update: `frontend/src/App.jsx` (add chat route)
```jsx
// Add import
import Chat from './pages/Chat';

// Add route in the Routes section
} 
/>
```

#### **Step 4.11: Run Database Migration and Test (1 hour)**
```bash
# Run chat tables migration
cd Fairdoc/backend
python data/database/migrations/create_chat_tables.py

# Verify tables created
docker exec -it fairdoc_postgres psql -U fairdoc -d fairdoc_dev -c "\dt"

# Should show chat_sessions and chat_messages tables
```

#### **Step 4.12: Test WebSocket Chat System (1.5 hours)**
Create: `tests/test_chat_websocket.py`
```python
"""
Test WebSocket chat functionality
"""
import asyncio
import json
import websockets
import requests

BASE_URL = "http://localhost:8000/api/v1"
WS_URL = "ws://localhost:8000/api/v1/ws/chat"

async def test_websocket_connection():
    """Test WebSocket connection and basic messaging"""
    session_id = f"test-session-{int(asyncio.get_event_loop().time())}"
    uri = f"{WS_URL}/{session_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ WebSocket connected to {uri}")
            
            # Wait for welcome message
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            print(f"📨 Welcome message: {welcome_data}")
            
            # Send a test message
            test_message = {
                "type": "message",
                "content": {
                    "text": "I have a headache"
                }
            }
            
            await websocket.send(json.dumps(test_message))
            print("📤 Sent test message")
            
            # Wait for AI response
            response_msg = await websocket.recv()
            response_data = json.loads(response_msg)
            print(f"📨 AI Response: {response_data}")
            
            # Verify response structure
            assert response_data["type"] == "response"
            assert "content" in response_data
            assert "text" in response_data["content"]
            
            # Send ping
            ping_message = {
                "type": "ping",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await websocket.send(json.dumps(ping_message))
            
            # Wait for pong
            pong_msg = await websocket.recv()
            pong_data = json.loads(pong_msg)
            assert pong_data["type"] == "pong"
            
            print("✅ WebSocket chat test passed")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        raise

def test_chat_api_endpoints():
    """Test chat-related API endpoints"""
    print("🧪 Testing chat API endpoints...")
    
    # Test active sessions endpoint
    response = requests.get(f"{BASE_URL}/chat/sessions/active")
    assert response.status_code == 200
    
    result = response.json()
    assert "active_sessions" in result
    assert "connection_count" in result
    
    print("✅ Chat API endpoints test passed")

async def test_emergency_detection():
    """Test emergency keyword detection"""
    session_id = f"emergency-test-{int(asyncio.get_event_loop().time())}"
    uri = f"{WS_URL}/{session_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for welcome
            await websocket.recv()
            
            # Send emergency message
            emergency_message = {
                "type": "message",
                "content": {
                    "text": "I'm having severe chest pain and can't breathe"
                }
            }
            
            await websocket.send(json.dumps(emergency_message))
            
            # Wait for response
            response_msg = await websocket.recv()
            response_data = json.loads(response_msg)
            
            # Should trigger escalation
            assert response_data["ai_state"]["escalation_needed"] == True
            
            print("✅ Emergency detection test passed")
            
    except Exception as e:
        print(f"❌ Emergency detection test failed: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Starting WebSocket chat tests...")
    
    # Run tests
    asyncio.run(test_websocket_connection())
    test_chat_api_endpoints()
    asyncio.run(test_emergency_detection())
    
    print("🎉 All WebSocket chat tests passed!")
```

#### **Step 4.13: Manual Frontend Test (1 hour)**
```bash
# Start backend
cd Fairdoc/backend
./scripts/dev-start.sh

# Start frontend in another terminal
cd Fairdoc/frontend
npm run dev

# Test complete chat flow:
# 1. Navigate to http://localhost:3000/chat
# 2. Send messages and verify AI responses
# 3. Test quick replies
# 4. Test emergency detection
# 5. Verify WebSocket connection status
```

### **📈 Day 4 Success Metrics**
- ✅ WebSocket connections establish successfully
- ✅ Real-time message exchange works
- ✅ Chat messages stored in database
- ✅ AI responses generated based on intent
- ✅ Emergency detection triggers escalation
- ✅ Frontend chat interface functional
- ✅ Context preservation across messages

---

**Continue with the same detailed format for Days 5-10...**

# **DAY 5: Bias Monitoring Dashboard** ⚖️
## **🕐 12 Hours | Goal: Real-time bias detection and reporting**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 5.1: Create Enhanced Bias Detection Service (1 hour)**
Create: `services/bias_detection_service_extended.py`
```python
"""
Enhanced bias detection service with real-time monitoring
"""
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID
import numpy as np

from datamodels.bias_models import BiasIncident, BiasMetrics, DemographicGroup
from data.repositories.bias_repository import BiasRepository

logger = logging.getLogger(__name__)

class RealTimeBiasMonitor:
    """Real-time bias monitoring and detection system"""
    
    def __init__(self):
        self.bias_repo = BiasRepository()
        
        # Bias thresholds
        self.DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
        self.EQUALIZED_ODDS_THRESHOLD = 0.1
        self.CALIBRATION_THRESHOLD = 0.05
        
        # Alert levels
        self.ALERT_THRESHOLDS = {
            "low": 0.05,
            "medium": 0.1,
            "high": 0.2,
            "critical": 0.3
        }
        
        # Recent decisions cache for real-time analysis
        self.recent_decisions = []
        self.cache_window = timedelta(hours=1)
    
    async def monitor_decision(
        self,
        decision_data: Dict,
        patient_demographics: Dict,
        assessment_id: UUID,
        db = None
    ) -> Dict:
        """Monitor a single decision for bias"""
        
        try:
            # Extract key information
            risk_score = decision_data.get('risk_level', 'unknown')
            recommended_action = decision_data.get('recommended_action', 'unknown')
            confidence = decision_data.get('confidence_score', 0.0)
            
            # Cache decision for real-time analysis
            decision_record = {
                'timestamp': datetime.utcnow(),
                'assessment_id': assessment_id,
                'demographics': patient_demographics,
                'risk_score': risk_score,
                'recommended_action': recommended_action,
                'confidence': confidence
            }
            
            self.recent_decisions.append(decision_record)
            
            # Clean old decisions from cache
            cutoff_time = datetime.utcnow() - self.cache_window
            self.recent_decisions = [
                d for d in self.recent_decisions 
                if d['timestamp'] > cutoff_time
            ]
            
            # Calculate bias metrics
            bias_metrics = await self.calculate_bias_metrics(decision_record)
            
            # Check for bias violations
            bias_alerts = self.check_bias_thresholds(bias_metrics)
            
            # Log any bias incidents
            if bias_alerts:
                await self.log_bias_incident(
                    assessment_id=assessment_id,
                    bias_type=bias_alerts[0]['type'],
                    severity=bias_alerts[0]['severity'],
                    details=bias_alerts[0]['details'],
                    demographics=patient_demographics,
                    decision_data=decision_data,
                    db=db
                )
            
            return {
                'bias_score': bias_metrics['overall_bias_score'],
                'bias_alerts': bias_alerts,
                'demographic_parity': bias_metrics['demographic_parity'],
                'equalized_odds': bias_metrics.get('equalized_odds', 0.0),
                'calibration': bias_metrics.get('calibration', 0.0),
                'monitoring_status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Bias monitoring failed: {e}")
            return {
                'bias_score': 0.0,
                'bias_alerts': [],
                'error': str(e),
                'monitoring_status': 'error'
            }
    
    async def calculate_bias_metrics(self, current_decision: Dict) -> Dict:
        """Calculate bias metrics using recent decisions"""
        
        if len(self.recent_decisions)  Dict:
        """Group decisions by demographic characteristics"""
        
        groups = {}
        
        for decision in decisions:
            demographics = decision['demographics']
            
            # Create group keys for different demographic combinations
            gender = demographics.get('gender', 'unknown')
            ethnicity = demographics.get('ethnicity', 'unknown')
            age_group = self.get_age_group(demographics.get('age', 0))
            
            # Group by gender
            gender_key = f"gender_{gender}"
            if gender_key not in groups:
                groups[gender_key] = []
            groups[gender_key].append(decision)
            
            # Group by ethnicity
            ethnicity_key = f"ethnicity_{ethnicity}"
            if ethnicity_key not in groups:
                groups[ethnicity_key] = []
            groups[ethnicity_key].append(decision)
            
            # Group by age
            age_key = f"age_{age_group}"
            if age_key not in groups:
                groups[age_key] = []
            groups[age_key].append(decision)
        
        return groups
    
    def get_age_group(self, age: int) -> str:
        """Convert age to age group"""
        if age  float:
        """Calculate demographic parity score"""
        
        try:
            # Get risk score rates for each group
            group_rates = {}
            
            for group_name, decisions in groups.items():
                if len(decisions)  float:
        """Calculate equalized odds score (simplified version)"""
        
        try:
            # For now, use confidence score variance as proxy
            group_confidences = {}
            
            for group_name, decisions in groups.items():
                if len(decisions)  1 else 0.0
            
            return min(1.0, confidence_variance)
            
        except Exception as e:
            logger.error(f"Equalized odds calculation failed: {e}")
            return 0.0
    
    def calculate_calibration(self, groups: Dict) -> float:
        """Calculate calibration score"""
        
        try:
            # Simplified calibration based on confidence distribution
            all_confidences = []
            
            for group_name, decisions in groups.items():
                confidences = [d['confidence'] for d in decisions]
                all_confidences.extend(confidences)
            
            if len(all_confidences) < 5:
                return 0.0
            
            # Check if confidence scores are well-distributed
            confidence_std = statistics.stdev(all_confidences)
            confidence_mean = statistics.mean(all_confidences)
            
            # Good calibration should have reasonable variance
            calibration_score = abs(confidence_std -

```
