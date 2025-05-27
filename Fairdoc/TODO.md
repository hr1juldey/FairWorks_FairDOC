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
            
            if len(all_confidences)  List[Dict]:
        """Check if bias metrics exceed thresholds"""
        
        alerts = []
        
        # Check demographic parity
        parity_score = metrics.get('demographic_parity', 0.0)
        if parity_score > self.DEMOGRAPHIC_PARITY_THRESHOLD:
            severity = self.get_alert_severity(parity_score)
            alerts.append({
                'type': 'demographic_parity',
                'severity': severity,
                'score': parity_score,
                'threshold': self.DEMOGRAPHIC_PARITY_THRESHOLD,
                'details': f"Demographic parity violation: {parity_score:.3f} > {self.DEMOGRAPHIC_PARITY_THRESHOLD}"
            })
        
        # Check equalized odds
        odds_score = metrics.get('equalized_odds', 0.0)
        if odds_score > self.EQUALIZED_ODDS_THRESHOLD:
            severity = self.get_alert_severity(odds_score)
            alerts.append({
                'type': 'equalized_odds',
                'severity': severity,
                'score': odds_score,
                'threshold': self.EQUALIZED_ODDS_THRESHOLD,
                'details': f"Equalized odds violation: {odds_score:.3f} > {self.EQUALIZED_ODDS_THRESHOLD}"
            })
        
        return alerts
    
    def get_alert_severity(self, score: float) -> str:
        """Determine alert severity based on score"""
        if score >= self.ALERT_THRESHOLDS["critical"]:
            return "critical"
        elif score >= self.ALERT_THRESHOLDS["high"]:
            return "high"
        elif score >= self.ALERT_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"
    
    async def log_bias_incident(
        self,
        assessment_id: UUID,
        bias_type: str,
        severity: str,
        details: str,
        demographics: Dict,
        decision_data: Dict,
        db = None
    ):
        """Log bias incident to database"""
        
        try:
            incident_data = {
                'assessment_id': assessment_id,
                'bias_type': bias_type,
                'severity': severity,
                'details': details,
                'demographics': demographics,
                'decision_data': decision_data,
                'timestamp': datetime.utcnow()
            }
            
            await self.bias_repo.create_bias_incident(incident_data, db)
            
            logger.warning(f"Bias incident logged: {bias_type} - {severity} - {details}")
            
        except Exception as e:
            logger.error(f"Failed to log bias incident: {e}")

# Global bias monitor instance
bias_monitor = RealTimeBiasMonitor()
```

#### **Step 5.2: Create Bias Repository (1 hour)**
Create: `data/repositories/bias_repository.py`
```python
"""
Bias repository for storing bias monitoring data
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from data.database.connection_manager import Base

logger = logging.getLogger(__name__)

class BiasIncidentModel(Base):
    """Bias incident database model"""
    __tablename__ = "bias_incidents"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid4)
    assessment_id = Column(PG_UUID(as_uuid=True), nullable=False)
    
    # Incident details
    bias_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    details = Column(Text, nullable=False)
    
    # Context data
    demographics = Column(JSON, nullable=False)
    decision_data = Column(JSON, nullable=False)
    
    # Status tracking
    status = Column(String(20), default="open")  # open, investigating, resolved
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(PG_UUID(as_uuid=True), nullable=True)
    
    # Audit
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class BiasMetricsModel(Base):
    """Bias metrics database model"""
    __tablename__ = "bias_metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    metrics_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid4)
    
    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Metrics
    demographic_parity = Column(Float, nullable=False)
    equalized_odds = Column(Float, nullable=False)
    calibration = Column(Float, nullable=False)
    overall_bias_score = Column(Float, nullable=False)
    
    # Sample size
    total_assessments = Column(Integer, nullable=False)
    groups_analyzed = Column(Integer, nullable=False)
    
    # Breakdown by demographics
    metrics_by_gender = Column(JSON, default=dict)
    metrics_by_ethnicity = Column(JSON, default=dict)
    metrics_by_age = Column(JSON, default=dict)
    
    # Audit
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class BiasRepository:
    """Repository for bias monitoring operations"""
    
    async def create_bias_incident(
        self,
        incident_data: Dict,
        db: Session
    ) -> BiasIncidentModel:
        """Create a new bias incident record"""
        try:
            incident = BiasIncidentModel(
                assessment_id=incident_data['assessment_id'],
                bias_type=incident_data['bias_type'],
                severity=incident_data['severity'],
                details=incident_data['details'],
                demographics=incident_data['demographics'],
                decision_data=incident_data['decision_data']
            )
            
            db.add(incident)
            db.commit()
            db.refresh(incident)
            
            logger.info(f"Bias incident created: {incident.incident_id}")
            return incident
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create bias incident: {e}")
            raise
    
    async def get_recent_incidents(
        self, 
        limit: int = 50, 
        severity: Optional[str] = None,
        db: Session = None
    ) -> List[BiasIncidentModel]:
        """Get recent bias incidents"""
        try:
            query = db.query(BiasIncidentModel)
            
            if severity:
                query = query.filter(BiasIncidentModel.severity == severity)
            
            incidents = query.order_by(
                BiasIncidentModel.created_at.desc()
            ).limit(limit).all()
            
            return incidents
        except Exception as e:
            logger.error(f"Failed to get recent incidents: {e}")
            return []
    
    async def get_bias_metrics_summary(
        self,
        time_range: str = "24h",
        db: Session = None
    ) -> Dict:
        """Get bias metrics summary for time range"""
        try:
            # Calculate time range
            if time_range == "1h":
                start_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                start_time = datetime.utcnow() - timedelta(hours=24)
            elif time_range == "7d":
                start_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                start_time = datetime.utcnow() - timedelta(days=30)
            else:
                start_time = datetime.utcnow() - timedelta(hours=24)
            
            # Get incidents in time range
            incidents = db.query(BiasIncidentModel).filter(
                BiasIncidentModel.created_at >= start_time
            ).all()
            
            # Count by severity
            severity_counts = {}
            bias_type_counts = {}
            
            for incident in incidents:
                severity = incident.severity
                bias_type = incident.bias_type
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
            
            # Calculate alert status
            critical_count = severity_counts.get('critical', 0)
            high_count = severity_counts.get('high', 0)
            
            if critical_count > 0:
                alert_status = "critical"
            elif high_count > 5:
                alert_status = "high"
            elif high_count > 0:
                alert_status = "medium"
            else:
                alert_status = "green"
            
            return {
                "time_range": time_range,
                "total_incidents": len(incidents),
                "severity_breakdown": severity_counts,
                "bias_type_breakdown": bias_type_counts,
                "alert_status": alert_status,
                "period_start": start_time,
                "period_end": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get bias metrics summary: {e}")
            return {"error": str(e)}
    
    async def store_metrics_snapshot(
        self,
        metrics_data: Dict,
        db: Session
    ) -> BiasMetricsModel:
        """Store a snapshot of bias metrics"""
        try:
            metrics = BiasMetricsModel(
                period_start=metrics_data['period_start'],
                period_end=metrics_data['period_end'],
                demographic_parity=metrics_data['demographic_parity'],
                equalized_odds=metrics_data['equalized_odds'],
                calibration=metrics_data['calibration'],
                overall_bias_score=metrics_data['overall_bias_score'],
                total_assessments=metrics_data['total_assessments'],
                groups_analyzed=metrics_data['groups_analyzed'],
                metrics_by_gender=metrics_data.get('metrics_by_gender', {}),
                metrics_by_ethnicity=metrics_data.get('metrics_by_ethnicity', {}),
                metrics_by_age=metrics_data.get('metrics_by_age', {})
            )
            
            db.add(metrics)
            db.commit()
            db.refresh(metrics)
            
            logger.info(f"Bias metrics snapshot stored: {metrics.metrics_id}")
            return metrics
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store bias metrics: {e}")
            raise
    
    async def resolve_incident(
        self,
        incident_id: UUID,
        resolution_notes: str,
        resolved_by: UUID,
        db: Session
    ) -> bool:
        """Mark bias incident as resolved"""
        try:
            db.query(BiasIncidentModel).filter(
                BiasIncidentModel.incident_id == incident_id
            ).update({
                "status": "resolved",
                "resolution_notes": resolution_notes,
                "resolved_by": resolved_by,
                "resolved_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            db.commit()
            logger.info(f"Bias incident resolved: {incident_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to resolve incident {incident_id}: {e}")
            return False
```

#### **Step 5.3: Create Bias Monitoring API Routes (1 hour)**
Create: `api/admin/bias_routes.py`
```python
"""
Bias monitoring API routes for Fairdoc AI
Handles bias detection, reporting, and management
"""
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, List

from core.dependencies import get_db, get_current_superuser, get_current_active_user
from services.bias_detection_service_extended import bias_monitor
from data.repositories.bias_repository import BiasRepository
from data.repositories.auth_repository import UserModel

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/bias/dashboard")
async def get_bias_dashboard(
    time_range: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    demographic_filter: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    Get bias monitoring dashboard data
    
    **Query Parameters:**
    - time_range: 1h, 24h, 7d, or 30d
    - demographic_filter: Optional filter by demographic group
    
    **Response:**
    ```
    {
        "summary": {
            "total_assessments": 1247,
            "bias_incidents": 3,
            "overall_fairness_score": 0.94,
            "alert_status": "green"
        },
        "demographic_metrics": {...},
        "recent_incidents": [...],
        "time_range": "24h"
    }
    ```
    """
    try:
        bias_repo = BiasRepository()
        
        # Get summary metrics
        summary = await bias_repo.get_bias_metrics_summary(time_range, db)
        
        # Get recent incidents
        recent_incidents = await bias_repo.get_recent_incidents(limit=10, db=db)
        
        # Format incidents for response
        incidents_data = []
        for incident in recent_incidents:
            incidents_data.append({
                "incident_id": str(incident.incident_id),
                "assessment_id": str(incident.assessment_id),
                "bias_type": incident.bias_type,
                "severity": incident.severity,
                "details": incident.details,
                "status": incident.status,
                "created_at": incident.created_at,
                "demographics": incident.demographics
            })
        
        # Calculate overall fairness score
        total_incidents = summary.get('total_incidents', 0)
        critical_incidents = summary.get('severity_breakdown', {}).get('critical', 0)
        
        # Simple fairness score calculation
        if total_incidents == 0:
            fairness_score = 1.0
        else:
            fairness_score = max(0.0, 1.0 - (total_incidents * 0.1) - (critical_incidents * 0.3))
        
        # Mock demographic metrics (in real implementation, calculate from recent decisions)
        demographic_metrics = {
            "gender": {
                "male": {"assessments": 623, "avg_risk_score": 0.45, "high_risk_rate": 0.12},
                "female": {"assessments": 624, "avg_risk_score": 0.43, "high_risk_rate": 0.11}
            },
            "ethnicity": {
                "white": {"assessments": 934, "avg_risk_score": 0.44, "high_risk_rate": 0.10},
                "asian": {"assessments": 156, "avg_risk_score": 0.46, "high_risk_rate": 0.14},
                "black": {"assessments": 89, "avg_risk_score": 0.42, "high_risk_rate": 0.09},
                "mixed": {"assessments": 68, "avg_risk_score": 0.45, "high_risk_rate": 0.12}
            },
            "age_groups": {
                "18_29": {"assessments": 234, "avg_risk_score": 0.38, "high_risk_rate": 0.08},
                "30_49": {"assessments": 567, "avg_risk_score": 0.44, "high_risk_rate": 0.11},
                "50_64": {"assessments": 298, "avg_risk_score": 0.48, "high_risk_rate": 0.15},
                "65_plus": {"assessments": 148, "avg_risk_score": 0.52, "high_risk_rate": 0.18}
            }
        }
        
        return {
            "summary": {
                "total_assessments": 1247,  # Mock total
                "bias_incidents": summary.get('total_incidents', 0),
                "overall_fairness_score": round(fairness_score, 3),
                "alert_status": summary.get('alert_status', 'green'),
                "time_range": time_range
            },
            "demographic_metrics": demographic_metrics,
            "fairness_metrics": {
                "demographic_parity": 0.08,  # Mock values
                "equalized_odds": 0.06,
                "calibration": 0.04
            },
            "recent_incidents": incidents_data,
            "bias_trends": {
                "last_hour": 0,
                "last_24h": summary.get('total_incidents', 0),
                "last_week": summary.get('total_incidents', 0) * 7,
                "trend": "stable"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get bias dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve bias dashboard data"
        )

@router.get("/bias/incidents")
async def get_bias_incidents(
    limit: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Get list of bias incidents with filtering"""
    try:
        bias_repo = BiasRepository()
        
        incidents = await bias_repo.get_recent_incidents(
            limit=limit,
            severity=severity,
            db=db
        )
        
        incidents_data = []
        for incident in incidents:
            incidents_data.append({
                "incident_id": str(incident.incident_id),
                "assessment_id": str(incident.assessment_id),
                "bias_type": incident.bias_type,
                "severity": incident.severity,
                "details": incident.details,
                "status": incident.status,
                "created_at": incident.created_at,
                "updated_at": incident.updated_at,
                "demographics": incident.demographics,
                "decision_data": incident.decision_data,
                "resolution_notes": incident.resolution_notes,
                "resolved_at": incident.resolved_at
            })
        
        return {
            "incidents": incidents_data,
            "total_count": len(incidents_data),
            "filters_applied": {
                "severity": severity,
                "status": status_filter,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get bias incidents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve bias incidents"
        )

@router.post("/bias/incidents/{incident_id}/resolve")
async def resolve_bias_incident(
    incident_id: str,
    resolution_notes: str,
    current_user: UserModel = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Resolve a bias incident"""
    try:
        from uuid import UUID
        incident_uuid = UUID(incident_id)
        
        bias_repo = BiasRepository()
        
        success = await bias_repo.resolve_incident(
            incident_id=incident_uuid,
            resolution_notes=resolution_notes,
            resolved_by=current_user.id,
            db=db
        )
        
        if success:
            return {
                "success": True,
                "message": "Incident resolved successfully",
                "incident_id": incident_id,
                "resolved_by": str(current_user.id),
                "resolved_at": datetime.utcnow()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found or already resolved"
            )
            
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid incident ID format"
        )
    except Exception as e:
        logger.error(f"Failed to resolve incident {incident_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve incident"
        )

@router.get("/bias/metrics/live")
async def get_live_bias_metrics(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get real-time bias monitoring metrics"""
    try:
        # Get current bias metrics from the monitor
        current_time = datetime.utcnow()
        
        # Mock live metrics (in real implementation, get from bias_monitor)
        live_metrics = {
            "current_sample_size": len(bias_monitor.recent_decisions),
            "monitoring_window_hours": 1,
            "last_updated": current_time,
            "real_time_scores": {
                "demographic_parity": 0.08,
                "equalized_odds": 0.06,
                "calibration": 0.04,
                "overall_bias": 0.06
            },
            "alert_status": "green",
            "active_alerts": [],
            "trends": {
                "improving": True,
                "trend_direction": "stable",
                "change_rate": 0.001
            }
        }
        
        return live_metrics
        
    except Exception as e:
        logger.error(f"Failed to get live bias metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve live bias metrics"
        )

@router.post("/bias/test-alert")
async def test_bias_alert(
    alert_type: str = "demographic_parity",
    severity: str = "medium",
    current_user: UserModel = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """Test bias alert system (for development/testing)"""
    try:
        from uuid import uuid4
        
        # Create test bias incident
        test_incident_data = {
            'assessment_id': uuid4(),
            'bias_type': alert_type,
            'severity': severity,
            'details': f"Test {alert_type} alert with {severity} severity",
            'demographics': {
                "age": 35,
                "gender": "female",
                "ethnicity": "asian"
            },
            'decision_data': {
                "risk_level": "moderate",
                "recommended_action": "gp_appointment",
                "confidence_score": 0.75
            },
            'timestamp': datetime.utcnow()
        }
        
        bias_repo = BiasRepository()
        incident = await bias_repo.create_bias_incident(test_incident_data, db)
        
        return {
            "success": True,
            "message": f"Test {alert_type} alert created",
            "incident_id": str(incident.incident_id),
            "severity": severity,
            "created_at": incident.created_at
        }
        
    except Exception as e:
        logger.error(f"Failed to create test alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test alert"
        )
```

#### **Step 5.4: Create Database Migration for Bias Tables (1 hour)**
Create: `data/database/migrations/create_bias_tables.py`
```python
"""
Database migration for bias monitoring tables
"""
from sqlalchemy import text
from data.database.connection_manager import db_manager

def upgrade():
    """Create bias monitoring tables"""
    
    engine = db_manager.engine
    
    with engine.connect() as conn:
        # Drop existing tables if they exist
        conn.execute(text("DROP TABLE IF EXISTS bias_metrics CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS bias_incidents CASCADE;"))
        
        # Create bias incidents table
        conn.execute(text("""
            CREATE TABLE bias_incidents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                incident_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                assessment_id UUID NOT NULL,
                bias_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                details TEXT NOT NULL,
                demographics JSONB NOT NULL,
                decision_data JSONB NOT NULL,
                status VARCHAR(20) DEFAULT 'open',
                resolution_notes TEXT,
                resolved_at TIMESTAMP WITH TIME ZONE,
                resolved_by UUID,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))
        
        # Create bias metrics table
        conn.execute(text("""
            CREATE TABLE bias_metrics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                metrics_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                demographic_parity FLOAT NOT NULL,
                equalized_odds FLOAT NOT NULL,
                calibration FLOAT NOT NULL,
                overall_bias_score FLOAT NOT NULL,
                total_assessments INTEGER NOT NULL,
                groups_analyzed INTEGER NOT NULL,
                metrics_by_gender JSONB DEFAULT '{}',
                metrics_by_ethnicity JSONB DEFAULT '{}',
                metrics_by_age JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))
        
        # Create indexes
        conn.execute(text("""
            CREATE INDEX idx_bias_incidents_assessment_id ON bias_incidents(assessment_id);
            CREATE INDEX idx_bias_incidents_bias_type ON bias_incidents(bias_type);
            CREATE INDEX idx_bias_incidents_severity ON bias_incidents(severity);
            CREATE INDEX idx_bias_incidents_status ON bias_incidents(status);
            CREATE INDEX idx_bias_incidents_created_at ON bias_incidents(created_at);
            
            CREATE INDEX idx_bias_metrics_period_start ON bias_metrics(period_start);
            CREATE INDEX idx_bias_metrics_period_end ON bias_metrics(period_end);
            CREATE INDEX idx_bias_metrics_overall_score ON bias_metrics(overall_bias_score);
        """))
        
        # Insert sample bias incidents for demo
        conn.execute(text("""
            INSERT INTO bias_incidents 
            (assessment_id, bias_type, severity, details, demographics, decision_data, status) 
            VALUES 
            (uuid_generate_v4(), 'demographic_parity', 'medium', 'Gender disparity in cardiac referrals detected', 
             '{"age": 45, "gender": "female", "ethnicity": "white"}', 
             '{"risk_level": "high", "recommended_action": "urgent_care", "confidence_score": 0.85}', 
             'resolved'),
            (uuid_generate_v4(), 'equalized_odds', 'low', 'Minor confidence variance across age groups', 
             '{"age": 72, "gender": "male", "ethnicity": "asian"}', 
             '{"risk_level": "moderate", "recommended_action": "gp_appointment", "confidence_score": 0.72}', 
             'open'),
            (uuid_generate_v4(), 'demographic_parity', 'high', 'Significant ethnicity bias in emergency classifications', 
             '{"age": 28, "gender": "female", "ethnicity": "black"}', 
             '{"risk_level": "critical", "recommended_action": "emergency", "confidence_score": 0.92}', 
             'investigating');
        """))
        
        conn.commit()
        print("✅ Bias monitoring tables created successfully")

if __name__ == "__main__":
    upgrade()
```

### **📅 Afternoon Session (4 hours): 12:00 PM - 4:00 PM**

#### **Step 5.5: Update Medical Service with Bias Monitoring (1 hour)**
Update: `services/medical_ai_service.py` (add bias monitoring)
```python
# Add this import at the top
from services.bias_detection_service_extended import bias_monitor

# Update the assess_patient method to include bias monitoring
async def assess_patient(self, patient_input: PatientInput) -> TriageResponse:
    """Main patient assessment endpoint with bias monitoring"""
    start_time = time.time()
    
    try:
        # Calculate risk score (existing logic)
        risk_score = self.triage_engine.calculate_risk_score(patient_input)
        
        # Determine risk level and actions (existing logic)
        if risk_score > 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score > 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score > 0.3:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        urgency_level, recommended_action = self.triage_engine.determine_urgency_and_action(
            risk_score, patient_input
        )
        
        explanation = self.triage_engine.generate_explanation(
            patient_input, risk_score, urgency_level
        )
        
        next_steps = self._generate_next_steps(recommended_action, patient_input)
        when_to_seek_help = self._generate_warning_signs(patient_input)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create initial response
        response = TriageResponse(
            risk_level=risk_level,
            urgency_level=urgency_level,
            recommended_action=recommended_action,
            explanation=explanation,
            confidence_score=min(0.95, max(0.6, 1.0 - (risk_score * 0.3))),
            next_steps=next_steps,
            when_to_seek_help=when_to_seek_help,
            processing_time_ms=processing_time_ms
        )
        
        # NEW: Bias monitoring
        try:
            # Extract demographics for bias monitoring
            demographics = {
                'age': patient_input.age,
                'gender': patient_input.gender.value if hasattr(patient_input.gender, 'value') else str(patient_input.gender),
                'ethnicity': patient_input.ethnicity.value if patient_input.ethnicity and hasattr(patient_input.ethnicity, 'value') else str(patient_input.ethnicity) if patient_input.ethnicity else 'unknown'
            }
            
            # Monitor decision for bias
            bias_result = await bias_monitor.monitor_decision(
                decision_data=response.dict(),
                patient_demographics=demographics,
                assessment_id=response.assessment_id,
                db=None  # Will pass db session in API layer
            )
            
            # Add bias information to response (extend the response model if needed)
            # For now, log the bias monitoring result
            logger.info(f"Bias monitoring result: {bias_result}")
            
        except Exception as bias_error:
            logger.error(f"Bias monitoring failed: {bias_error}")
            # Continue with response even if bias monitoring fails
        
        logger.info(f"Patient assessment completed: {recommended_action}, risk: {risk_level}")
        return response
        
    except Exception as e:
        # ... existing error handling
```

#### **Step 5.6: Create Frontend Bias Dashboard Components (2 hours)**
Create: `frontend/src/components/BiasMetrics.jsx`
```jsx
import React, { useState, useEffect } from 'react';
import { ExclamationTriangleIcon, CheckCircleIcon, ClockIcon } from '@heroicons/react/24/outline';

function BiasMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [timeRange, setTimeRange] = useState('24h');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadBiasMetrics();
  }, [timeRange]);

  const loadBiasMetrics = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/admin/bias/dashboard?time_range=${timeRange}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to load bias metrics');
      
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getAlertStatusColor = (status) => {
    const colors = {
      green: 'text-green-600 bg-green-100',
      medium: 'text-yellow-600 bg-yellow-100',
      high: 'text-orange-600 bg-orange-100',
      critical: 'text-red-600 bg-red-100'
    };
    return colors[status] || colors.green;
  };

  const getAlertIcon = (status) => {
    if (status === 'green') return CheckCircleIcon;
    return ExclamationTriangleIcon;
  };

  if (loading) {
    return (
      
        
      
    );
  }

  if (error) {
    return (
      
        Error loading bias metrics: {error}
      
    );
  }

  if (!metrics) return null;

  const AlertIcon = getAlertIcon(metrics.summary.alert_status);

  return (
    
      {/* Header */}
      
        Bias Monitoring Dashboard
        
           setTimeRange(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            Last Hour
            Last 24 Hours
            Last 7 Days
            Last 30 Days
          
          
            Refresh
          
        
      

      {/* Summary Cards */}
      
        
          
            
              
            
            
              Alert Status
              
                {metrics.summary.alert_status}
              
            
          
        

        
          
            
              
                
              
            
            
              Total Assessments
              {metrics.summary.total_assessments.toLocaleString()}
            
          
        

        
          
            
              
                
              
            
            
              Bias Incidents
              {metrics.summary.bias_incidents}
            
          
        

        
          
            
              
                
              
            
            
              Fairness Score
              
                {(metrics.summary.overall_fairness_score * 100).toFixed(1)}%
              
            
          
        
      

      {/* Fairness Metrics */}
      
        Fairness Metrics
        
          
            
              Demographic Parity
              {(metrics.fairness_metrics.demographic_parity * 100).toFixed(1)}%
            
            
              
            
          

          
            
              Equalized Odds
              {(metrics.fairness_metrics.equalized_odds * 100).toFixed(1)}%
            
            
              
            
          

          
            
              Calibration
              {(metrics.fairness_metrics.calibration * 100).toFixed(1)}%
            
            
              
            
          
        
      

      {/* Demographic Breakdown */}
      
        {/* Gender Metrics */}
        
          Gender Distribution
          
            {Object.entries(metrics.demographic_metrics.gender).map(([gender, data]) => (
              
                
                  
                    {gender}
                    {data.assessments} assessments
                  
                  
                    Avg Risk: {data.avg_risk_score.toFixed(2)}
                    High Risk: {(data.high_risk_rate * 100).toFixed(1)}%
                  
                
              
            ))}
          
        

        {/* Ethnicity Metrics */}
        
          Ethnicity Distribution
          
            {Object.entries(metrics.demographic_metrics.ethnicity).map(([ethnicity, data]) => (
              
                
                  
                    {ethnicity}
                    {data.assessments} assessments
                  
                  
                    Avg Risk: {data.avg_risk_score.toFixed(2)}
                    High Risk: {(data.high_risk_rate * 100).toFixed(1)}%
                  
                
              
            ))}
          
        
      

      {/* Recent Incidents */}
      {metrics.recent_incidents && metrics.recent_incidents.length > 0 && (
        
          Recent Bias Incidents
          
            {metrics.recent_incidents.slice(0, 5).map((incident) => (
              
                
                  
                    
                      {incident.bias_type.replace('_', ' ').toUpperCase()} - {incident.severity.toUpperCase()}
                    
                    {incident.details}
                    
                      {new Date(incident.created_at).toLocaleString()}
                    
                  
                  
                    {incident.status}
                  
                
              
            ))}
          
        
      )}
    
  );
}

export default BiasMetrics;
```

#### **Step 5.7: Create Admin Dashboard Page (1 hour)**
Create: `frontend/src/pages/AdminDashboard.jsx`
```jsx
import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import BiasMetrics from '../components/BiasMetrics';
import { 
  ChartBarIcon, 
  ExclamationTriangleIcon, 
  UsersIcon, 
  CogIcon 
} from '@heroicons/react/24/outline';

function AdminDashboard() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('bias');

  if (!user || user.role !== 'admin') {
    return (
      
        
          
          Access Denied
          You need administrator privileges to access this page.
        
      
    );
  }

  const tabs = [
    {
      id: 'bias',
      name: 'Bias Monitoring',
      icon: ExclamationTriangleIcon,
      component: BiasMetrics
    },
    {
      id: 'analytics',
      name: 'System Analytics',
      icon: ChartBarIcon,
      component: () => Analytics coming soon...
    },
    {
      id: 'users',
      name: 'User Management',
      icon: UsersIcon,
      component: () => User management coming soon...
    },
    {
      id: 'settings',
      name: 'System Settings',
      icon: CogIcon,
      component: () => Settings coming soon...
    }
  ];

  const activeTabData = tabs.find(tab => tab.id === activeTab);
  const ActiveComponent = activeTabData?.component;

  return (
    
      
        
        {/* Header */}
        
          Administrator Dashboard
          Monitor system performance, bias metrics, and user activity.
        

        {/* Navigation Tabs */}
        
          
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                 setActiveTab(tab.id)}
                  className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  
                  {tab.name}
                
              );
            })}
          
        

        {/* Tab Content */}
        
          {ActiveComponent && }
        
      
    
  );
}

export default AdminDashboard;
```

### **📅 Evening Session (4 hours): 6:00 PM - 10:00 PM**

#### **Step 5.8: Update API Utils with Bias Endpoints (30 minutes)**
Update: `frontend/src/utils/api.js`
```javascript
// Add bias monitoring API endpoints
export const biasAPI = {
  getDashboard: async (timeRange = '24h', demographicFilter = null) => {
    const params = new URLSearchParams({ time_range: timeRange });
    if (demographicFilter) params.append('demographic_filter', demographicFilter);
    
    const response = await api.get(`/admin/bias/dashboard?${params}`);
    return response.data;
  },

  getIncidents: async (limit = 20, severity = null, status = null) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (severity) params.append('severity', severity);
    if (status) params.append('status_filter', status);
    
    const response = await api.get(`/admin/bias/incidents?${params}`);
    return response.data;
  },

  resolveIncident: async (incidentId, resolutionNotes) => {
    const response = await api.post(`/admin/bias/incidents/${incidentId}/resolve`, {
      resolution_notes: resolutionNotes
    });
    return response.data;
  },

  getLiveMetrics: async () => {
    const response = await api.get('/admin/bias/metrics/live');
    return response.data;
  },

  testAlert: async (alertType = 'demographic_parity', severity = 'medium') => {
    const response = await api.post('/admin/bias/test-alert', {
      alert_type: alertType,
      severity: severity
    });
    return response.data;
  }
};
```

#### **Step 5.9: Update Main App with Admin Routes (30 minutes)**
Update: `frontend/src/App.jsx`
```jsx
// Add import
import AdminDashboard from './pages/AdminDashboard';

// Add route in the Routes section (within protected routes)

      
    
  } 
/>
```

Update: `frontend/src/components/Navbar.jsx` (add admin link)
```jsx
// Add admin link for admin users
{isAuthenticated && user?.role === 'admin' && (
  
    Admin Dashboard
  
)}
```

#### **Step 5.10: Update Main App with Bias Routes (30 minutes)**
Update: `app.py`
```python
# Add this import
from api.admin.bias_routes import router as bias_router

# Add this line after other router includes
app.include_router(bias_router, prefix="/api/v1/admin", tags=["Bias Monitoring"])
```

#### **Step 5.11: Run Database Migration and Test (1 hour)**
```bash
# Run bias tables migration
cd Fairdoc/backend
python data/database/migrations/create_bias_tables.py

# Verify tables created
docker exec -it fairdoc_postgres psql -U fairdoc -d fairdoc_dev -c "\dt"

# Should show bias_incidents and bias_metrics tables
```

#### **Step 5.12: Test Bias Monitoring System (1.5 hours)**
Create: `tests/test_bias_monitoring.py`
```python
"""
Test bias monitoring functionality
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def get_admin_token():
    """Get admin token for testing"""
    login_data = {
        "email": "admin@fairdoc.ai",
        "password": "password",  # Default admin password
        "remember_me": False
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("Failed to get admin token")

def test_bias_dashboard():
    """Test bias dashboard endpoint"""
    print("🧪 Testing bias dashboard...")
    
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/admin/bias/dashboard", headers=headers)
    assert response.status_code == 200
    
    result = response.json()
    assert "summary" in result
    assert "demographic_metrics" in result
    assert "fairness_metrics" in result
    
    print("✅ Bias dashboard test passed")

def test_bias_incidents():
    """Test bias incidents endpoint"""
    print("🧪 Testing bias incidents...")
    
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/admin/bias/incidents", headers=headers)
    assert response.status_code == 200
    
    result = response.json()
    assert "incidents" in result
    assert "total_count" in result
    
    print("✅ Bias incidents test passed")

def test_create_test_alert():
    """Test creating test bias alert"""
    print("🧪 Testing test alert creation...")
    
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(
        f"{BASE_URL}/admin/bias/test-alert", 
        headers=headers,
        params={"alert_type": "demographic_parity", "severity": "medium"}
    )
    assert response.status_code == 200
    
    result = response.json()
    assert result["success"] == True
    assert "incident_id" in result
    
    print("✅ Test alert creation passed")

def test_live_metrics():
    """Test live bias metrics"""
    print("🧪 Testing live bias metrics...")
    
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/admin/bias/metrics/live", headers=headers)
    assert response.status_code == 200
    
    result = response.json()
    assert "real_time_scores" in result
    assert "alert_status" in result
    
    print("✅ Live metrics test passed")

if __name__ == "__main__":
    print("🚀 Starting bias monitoring tests...")
    
    try:
        test_bias_dashboard()
        test_bias_incidents()
        test_create_test_alert()
        test_live_metrics()
        
        print("🎉 All bias monitoring tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
```

#### **Step 5.13: Manual Frontend Test (1 hour)**
```bash
# Start backend with bias monitoring
cd Fairdoc/backend
./scripts/dev-start.sh

# Start frontend
cd Fairdoc/frontend
npm run dev

# Test bias monitoring:
# 1. Login as admin (admin@fairdoc.ai / password)
# 2. Navigate to http://localhost:3000/admin
# 3. View bias monitoring dashboard
# 4. Test different time ranges
# 5. Create test alerts
# 6. Verify metrics display correctly
```

### **📈 Day 5 Success Metrics**
- ✅ Bias monitoring service detects demographic disparities
- ✅ Real-time bias alerts trigger correctly
- ✅ Admin dashboard displays bias metrics
- ✅ Bias incidents logged to database
- ✅ Frontend bias dashboard functional
- ✅ Live metrics update in real-time

---

# **DAY 6: File Upload & Image Analysis** 📁
## **🕐 12 Hours | Goal: Multi-modal input processing**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 6.1: Create File Upload Models (1 hour)**
Create: `datamodels/file_models_extended.py`
```python
"""
Extended file models for multi-modal medical input
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from datamodels.base_models import BaseEntity, ValidationMixin, MetadataMixin

class FileType(str, Enum):
    """Supported file types for medical analysis"""
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"

class ImageCategory(str, Enum):
    """Medical image categories"""
    CHEST_XRAY = "chest_xray"
    SKIN_LESION = "skin_lesion"
    ECG = "ecg"
    MRI = "mri"
    CT_SCAN = "ct_scan"
    ULTRASOUND = "ultrasound"
    GENERAL_PHOTO = "general_photo"

class AudioCategory(str, Enum):
    """Medical audio categories"""
    HEART_SOUNDS = "heart_sounds"
    LUNG_SOUNDS = "lung_sounds"
    SPEECH_SAMPLE = "speech_sample"
    COUGH_RECORDING = "cough_recording"
    GENERAL_AUDIO = "general_audio"

class ProcessingStatus(str, Enum):
    """File processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"

class MedicalFile(BaseEntity, ValidationMixin, MetadataMixin):
    """Medical file upload with metadata"""
    
    file_id: UUID = Field(default_factory=uuid4)
    original_filename: str = Field(..., max_length=255)
    
    # File properties
    file_type: FileType
    file_size_bytes: int = Field(..., ge=0, le=50*1024*1024)  # Max 50MB
    mime_type: str = Field(..., max_length=100)
    
    # Medical categorization
    image_category: Optional[ImageCategory] = None
    audio_category: Optional[AudioCategory] = None
    
    # Storage information
    storage_path: str = Field(..., max_length=500)
    storage_bucket: Optional[str] = Field(None, max_length=100)
    
    # Processing status
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.UPLOADED)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    
    # Analysis results
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Security and compliance
    virus_scan_status: str = Field(default="pending")  # pending, clean, infected
    phi_detected: bool = Field(default=False)  # Personal Health Information
    
    # Association
    patient_id: Optional[UUID] = None
    assessment_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    
    @field_validator('file_size_bytes')
    @classmethod
    def validate_file_size(cls, v):
        if v > 50 * 1024 * 1024:  # 50MB
            raise ValueError('File size exceeds 50MB limit')
        return v

class FileUploadRequest(BaseModel):
    """File upload request"""
    filename: str = Field(..., max_length=255)
    file_type: FileType
    mime_type: str
    file_size: int = Field(..., ge=0)
    
    # Optional categorization
    image_category: Optional[ImageCategory] = None
    audio_category: Optional[AudioCategory] = None
    
    # Optional associations
    patient_id: Optional[UUID] = None
    assessment_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    
    # Description
    description: Optional[str] = Field(None, max_length=500)

class FileUploadResponse(BaseModel):
    """File upload response"""
    file_id: UUID
    upload_url: str
    file_key: str
    expires_at: datetime
    max_file_size: int = Field(default=50*1024*1024)

class FileAnalysisRequest(BaseModel):
    """Request for file analysis"""
    file_id: UUID
    analysis_type: str = Field(..., max_length=50)
    priority: Literal["low", "normal", "high"] = "normal"
    
    # Analysis parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ImageAnalysisResult(BaseModel):
    """Image analysis result"""
    file_id: UUID
    analysis_type: str
    
    # Detection results
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Medical interpretation
    medical_significance: str = Field(..., max_length=1000)
    urgency_level: str = Field(default="routine")
    recommended_action: str = Field(..., max_length=500)
    
    # Technical metadata
    image_quality: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    model_version: str
    
    # Bias monitoring
    bias_score: float = Field(default=0.0, ge=0.0, le=1.0)

class AudioAnalysisResult(BaseModel):
    """Audio analysis result"""
    file_id: UUID
    analysis_type: str
    
    # Audio processing results
    transcript: Optional[str] = None
    audio_features: Dict[str, float] = Field(default_factory=dict)
    
    # Medical analysis
    abnormalities_detected: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Clinical interpretation
    clinical_notes: str = Field(..., max_length=1000)
    follow_up_needed: bool = Field(default=False)
    
    # Technical metadata
    audio_quality: float = Field(..., ge=0.0, le=1.0)
    duration_seconds: float
    sample_rate: int
    processing_time_ms: float
    
    # Bias monitoring
    bias_score: float = Field(default=0.0, ge=0.0, le=1.0)
```

#### **Step 6.2: Create File Repository (1 hour)**
Create: `data/repositories/file_repository.py`
```python
"""
File repository for medical file storage and retrieval
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

class MedicalFileModel(Base):
    """Medical file database model"""
    __tablename__ = "medical_files"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid4)
    
    # File metadata
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Medical categorization
    image_category = Column(String(50), nullable=True)
    audio_category = Column(String(50), nullable=True)
    
    # Storage
    storage_path = Column(String(500), nullable=False)
    storage_bucket = Column(String(100), nullable=True)
    
    # Processing
    processing_status = Column(String(20), default="uploaded")
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Analysis results
    analysis_results = Column(JSON, default=dict)
    confidence_score = Column(Float, nullable=True)
    
    # Security
    virus_scan_status = Column(String(20), default="pending")
    phi_detected = Column(Boolean, default=False)
    
    # Associations
    patient_id = Column(PG_UUID(as_uuid=True), nullable=True)
    assessment_id = Column(PG_UUID(as_uuid=True), nullable=True)
    session_id = Column(PG_UUID(as_uuid=True), nullable=True)
    
    # Audit
    uploaded_by = Column(PG_UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class FileRepository:
    """Repository for medical file operations"""
    
    async def create_file_record(
        self,
        file_data: dict,
        uploaded_by: Optional[UUID] = None,
        db: Session = None
    ) -> MedicalFileModel:
        """Create a new file record"""
        try:
            file_record = MedicalFileModel(
                original_filename=file_data['filename'],
                file_type=file_data['file_type'],
                file_size_bytes=file_data['file_size'],
                mime_type=file_data['mime_type'],
                image_category=file_data.get('image_category'),
                audio_category=file_data.get('audio_category'),
                storage_path=file_data['storage_path'],
                storage_bucket=file_data.get('storage_bucket'),
                patient_id=file_data.get('patient_id'),
                assessment_id=file_data.get('assessment_id'),
                session_id=file_data.get('session_id'),
                uploaded_by=uploaded_by
            )
            
            db.add(file_record)
            db.commit()
            db.refresh(file_record)
            
            logger.info(f"File record created: {file_record.file_id}")
            return file_record
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create file record: {e}")
            raise
    
    async def get_file_by_id(self, file_id: UUID, db: Session) -> Optional[MedicalFileModel]:
        """Get file record by ID"""
        try:
            file_record = db.query(MedicalFileModel).filter(
                MedicalFileModel.file_id == file_id
            ).first()
            return file_record
        except Exception as e:
            logger.error(f"Failed to get file {file_id}: {e}")
            return None
    
    async def update_processing_status(
        self,
        file_id: UUID,
        status: str,
        error: Optional[str] = None,
        db: Session = None
    ) -> bool:
        """Update file processing status"""
        try:
            update_data = {
                "processing_status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status == "processing":
                update_data["processing_started_at"] = datetime.utcnow()
            elif status in ["completed", "failed"]:
                update_data["processing_completed_at"] = datetime.utcnow()
                if error:
                    update_data["processing_error"] = error
            
            db.query(MedicalFileModel).filter(
                MedicalFileModel.file_id == file_id
            ).update(update_data)
            
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update processing status for {file_id}: {e}")
            return False
    
    async def store_analysis_results(
        self,
        file_id: UUID,
        analysis_results: dict,
        confidence_score: Optional[float] = None,
        db: Session = None
    ) -> bool:
        """Store analysis results for a file"""
        try:
            update_data = {
                "analysis_results": analysis_results,
                "processing_status": "completed",
                "processing_completed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            if confidence_score is not None:
                update_data["confidence_score"] = confidence_score
            
            db.query(MedicalFileModel).filter(
                MedicalFileModel.file_id == file_id
            ).update(update_data)
            
            db.commit()
            logger.info(f"Analysis results stored for file: {file_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store analysis results for {file_id}: {e}")
            return False
    
    async def get_files_by_assessment(
        self,
        assessment_id: UUID,
        file_type: Optional[str] = None,
        db: Session = None
    ) -> List[MedicalFileModel]:
        """Get all files associated with an assessment"""
        try:
            query = db.query(MedicalFileModel).filter(
                MedicalFileModel.assessment_id == assessment_id
            )
            
            if file_type:
                query = query.filter(MedicalFileModel.file_type == file_type)
            
            files = query.order_by(MedicalFileModel.created_at.desc()).all()
            return files
            
        except Exception as e:
            logger.error(f"Failed to get files for assessment {assessment_id}: {e}")
            return []
    
    async def get_pending_analysis_files(
        self,
        file_type: Optional[str] = None,
        limit: int = 10,
        db: Session = None
    ) -> List[MedicalFileModel]:
        """Get files pending analysis"""
        try:
            query = db.query(MedicalFileModel).filter(
                MedicalFileModel.processing_status.in_(["uploaded", "processing"])
            )
            
            if file_type:
                query = query.filter(MedicalFileModel.file_type == file_type)
            
            files = query.order_by(MedicalFileModel.created_at.asc()).limit(limit).all()
            return files
            
        except Exception as e:
            logger.error(f"Failed to get pending analysis files: {e}")
            return []
    
    async def delete_file_record(self, file_id: UUID, db: Session) -> bool:
        """Delete file record (soft delete by updating status)"""
        try:
            db.query(MedicalFileModel).filter(
                MedicalFileModel.file_id == file_id
            ).update({
                "processing_status": "deleted",
                "updated_at": datetime.utcnow()
            })
            
            db.commit()
            logger.info(f"File record deleted: {file_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete file record {file_id}: {e}")
            return False
```

#### **Step 6.3: Create Image Analysis Service (1 hour)**
Create: `services/image_analysis_service.py`
```python
"""
Image analysis service for medical image processing
"""
import logging
import time
import base64
import io
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from PIL import Image
import numpy as np

from datamodels.file_models_extended import ImageAnalysisResult, ImageCategory

logger = logging.getLogger(__name__)

class MedicalImageAnalyzer:
    """Basic medical image analysis using rule-based detection"""
    
    def __init__(self):
        # Mock ML models for Day 6 (replace with real models later)
        self.models = {
            'chest_xray': self._analyze_chest_xray,
            'skin_lesion': self._analyze_skin_lesion,
            'ecg': self._analyze_ecg,
            'general_photo': self._analyze_general_photo
        }
        
        # Image processing parameters
        self.max_image_size = (1024, 1024)
        self.supported_formats = ['JPEG', 'PNG', 'BMP', 'TIFF']
    
    async def analyze_image(
        self,
        file_id: UUID,
        image_data: bytes,
        image_category: ImageCategory,
        patient_demographics: Optional[Dict] = None
    ) -> ImageAnalysisResult:
        """Analyze medical image and return results"""
        
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._load_image(image_data)
            processed_image = self._preprocess_image(image)
            
            # Calculate image quality
            quality_score = self._assess_image_quality(processed_image)
            
            # Perform analysis based on category
            analyzer_func = self.models.get(
                image_category.value,
                self._analyze_general_photo
            )
            
            findings, confidence, medical_significance, urgency, action = await analyzer_func(
                processed_image,
                patient_demographics
            )
            
            # Calculate bias score (mock implementation)
            bias_score = self._calculate_bias_score(findings, patient_demographics)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = ImageAnalysisResult(
                file_id=file_id,
                analysis_type=image_category.value,
                findings=findings,
                confidence_score=confidence,
                medical_significance=medical_significance,
                urgency_level=urgency,
                recommended_action=action,
                image_quality=quality_score,
                processing_time_ms=processing_time,
                model_version="rule_based_v1.0",
                bias_score=bias_score
            )
            
            logger.info(f"Image analysis completed for {file_id}: {image_category}")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed for {file_id}: {e}")
            
            # Return error result
            return ImageAnalysisResult(
                file_id=file_id,
                analysis_type=image_category.value,
                findings=[{"error": str(e)}],
                confidence_score=0.0,
                medical_significance="Analysis failed due to technical error",
                urgency_level="unknown",
                recommended_action="Please try uploading the image again or consult manually",
                image_quality=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version="rule_based_v1.0",
                bias_score=0.0
            )
    
    def _load_image(self, image_data: bytes) -> Image.Image:
        """Load image from bytes"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for analysis"""
        # Resize if too large
        if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _assess_image_quality(self, image: Image.Image) -> float:
        """Assess image quality (0.0 to 1.0)"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Simple quality metrics
            # 1. Check if image is too dark or too bright
            mean_brightness = np.mean(img_array)
            brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
            
            # 2. Check contrast (standard deviation)
           ```python
            contrast_score = np.std(img_array) / 255.0
            contrast_score = min(1.0, contrast_score * 4)  # Normalize
            
            # 3. Check sharpness (using Laplacian variance)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            laplacian_var = np.var(np.array([
                [0, -1, 0], [-1, 4, -1], [0, -1, 0]
            ]))  # Mock sharpness calculation
            sharpness_score = min(1.0, laplacian_var / 100)
            
            # Combined quality score
            quality_score = (brightness_score * 0.3 + contrast_score * 0.4 + sharpness_score * 0.3)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Image quality assessment failed: {e}")
            return 0.5  # Default medium quality
    
    async def _analyze_chest_xray(self, image: Image.Image, demographics: Optional[Dict]) -> Tuple:
        """Analyze chest X-ray images"""
        findings = []
        confidence = 0.75
        
        # Mock chest X-ray analysis
        img_array = np.array(image)
        
        # Check for potential pneumonia (dark regions)
        dark_ratio = np.sum(img_array  0.3:
            findings.append({
                "finding": "Possible infiltrate",
                "location": "bilateral lower lobes",
                "confidence": 0.65,
                "severity": "moderate"
            })
        
        # Check for cardiomegaly (heart enlargement)
        height, width = img_array.shape[:2]
        heart_region = img_array[height//3:2*height//3, width//3:2*width//3]
        heart_density = np.mean(heart_region)
        
        if heart_density > 150:
            findings.append({
                "finding": "Possible cardiomegaly",
                "confidence": 0.55,
                "recommendation": "ECG and echocardiogram recommended"
            })
        
        # Age-based risk adjustment
        if demographics and demographics.get('age', 0) > 65:
            confidence *= 0.9  # Slightly lower confidence for elderly
        
        if findings:
            medical_significance = f"Chest X-ray shows {len(findings)} concerning finding(s) that require medical evaluation"
            urgency = "urgent" if len(findings) > 1 else "routine"
            action = "Refer to pulmonologist for further evaluation and possible CT scan"
        else:
            findings.append({
                "finding": "No acute findings",
                "confidence": 0.8
            })
            medical_significance = "Chest X-ray appears normal with no acute abnormalities"
            urgency = "routine"
            action = "Continue routine care, repeat if symptoms persist"
        
        return findings, confidence, medical_significance, urgency, action
    
    async def _analyze_skin_lesion(self, image: Image.Image, demographics: Optional[Dict]) -> Tuple:
        """Analyze skin lesion images"""
        findings = []
        confidence = 0.70
        
        # Mock dermatology analysis using ABCDE criteria
        img_array = np.array(image)
        
        # A - Asymmetry (mock calculation)
        height, width = img_array.shape[:2]
        left_half = img_array[:, :width//2]
        right_half = img_array[:, width//2:]
        asymmetry_score = np.mean(np.abs(left_half.flatten()[:right_half.size] - right_half.flatten()))
        
        # B - Border irregularity
        edges = np.gradient(np.mean(img_array, axis=2))
        border_irregularity = np.std(edges)
        
        # C - Color variation
        color_std = np.std(img_array)
        
        # Risk assessment
        risk_factors = 0
        if asymmetry_score > 50:
            risk_factors += 1
            findings.append({
                "finding": "Asymmetric lesion",
                "criterion": "ABCDE - Asymmetry",
                "confidence": 0.6
            })
        
        if border_irregularity > 30:
            risk_factors += 1
            findings.append({
                "finding": "Irregular borders",
                "criterion": "ABCDE - Border",
                "confidence": 0.7
            })
        
        if color_std > 40:
            risk_factors += 1
            findings.append({
                "finding": "Color variation",
                "criterion": "ABCDE - Color",
                "confidence": 0.65
            })
        
        # Age and family history adjustments
        if demographics:
            age = demographics.get('age', 0)
            if age > 50:
                risk_factors += 0.5
                findings.append({
                    "finding": "Age-related risk factor",
                    "note": "Increased skin cancer risk with age"
                })
        
        if risk_factors >= 2:
            medical_significance = "Lesion shows concerning features requiring dermatological evaluation"
            urgency = "urgent"
            action = "Refer to dermatologist within 2 weeks for biopsy consideration"
            confidence = 0.8
        elif risk_factors >= 1:
            medical_significance = "Lesion shows some atypical features"
            urgency = "routine"
            action = "Monitor closely, photograph for changes, consider dermatology consultation"
            confidence = 0.7
        else:
            findings.append({
                "finding": "Benign-appearing lesion",
                "confidence": 0.75
            })
            medical_significance = "Lesion appears benign with no concerning features"
            urgency = "routine"
            action = "Continue skin self-examination, routine follow-up"
        
        return findings, confidence, medical_significance, urgency, action
    
    async def _analyze_ecg(self, image: Image.Image, demographics: Optional[Dict]) -> Tuple:
        """Analyze ECG/EKG images"""
        findings = []
        confidence = 0.65  # Lower confidence for complex ECG interpretation
        
        # Mock ECG analysis
        img_array = np.array(image)
        
        # Check for rhythm abnormalities (mock - would use signal processing in real implementation)
        signal_variance = np.var(img_array)
        
        if signal_variance > 1000:
            findings.append({
                "finding": "Irregular rhythm pattern",
                "type": "Possible atrial fibrillation",
                "confidence": 0.6
            })
        
        # Check for ST elevation (mock)
        mean_intensity = np.mean(img_array)
        if mean_intensity > 180:
            findings.append({
                "finding": "Possible ST elevation",
                "leads": "II, III, aVF",
                "confidence": 0.7,
                "clinical_significance": "Possible myocardial infarction"
            })
        
        # Age-specific considerations
        if demographics:
            age = demographics.get('age', 0)
            if age > 65 and findings:
                medical_significance = "ECG abnormalities in elderly patient require urgent cardiology evaluation"
                urgency = "urgent"
                action = "Immediate cardiology consultation, cardiac enzymes, continuous monitoring"
            elif findings:
                medical_significance = "ECG shows abnormalities requiring further cardiac evaluation"
                urgency = "urgent"
                action = "Urgent cardiology referral, consider stress testing"
            else:
                findings.append({
                    "finding": "Normal sinus rhythm",
                    "confidence": 0.8
                })
                medical_significance = "ECG within normal limits"
                urgency = "routine"
                action = "Continue routine cardiac care"
        
        return findings, confidence, medical_significance, urgency, action
    
    async def _analyze_general_photo(self, image: Image.Image, demographics: Optional[Dict]) -> Tuple:
        """Analyze general medical photos"""
        findings = []
        confidence = 0.60  # Lower confidence for general images
        
        # Basic image analysis
        img_array = np.array(image)
        
        # Check for obvious abnormalities
        red_channel = img_array[:, :, 0] if len(img_array.shape) == 3 else img_array
        red_intensity = np.mean(red_channel)
        
        if red_intensity > 180:
            findings.append({
                "finding": "Increased redness/inflammation",
                "confidence": 0.5,
                "recommendation": "Consider inflammatory condition"
            })
        
        # Check for swelling (brightness variations)
        brightness_var = np.var(np.mean(img_array, axis=2))
        if brightness_var > 500:
            findings.append({
                "finding": "Possible swelling or edema",
                "confidence": 0.45
            })
        
        if findings:
            medical_significance = "Image shows possible abnormalities requiring clinical correlation"
            urgency = "routine"
            action = "Clinical examination recommended, correlate with symptoms"
        else:
            findings.append({
                "finding": "No obvious abnormalities detected",
                "note": "Limited by image quality and type"
            })
            medical_significance = "No clear abnormalities identified in this general medical image"
            urgency = "routine"
            action = "Continue monitoring, seek professional evaluation if symptoms persist"
        
        return findings, confidence, medical_significance, urgency, action
    
    def _calculate_bias_score(self, findings: List[Dict], demographics: Optional[Dict]) -> float:
        """Calculate bias score for image analysis"""
        bias_score = 0.0
        
        if not demographics:
            return bias_score
        
        # Mock bias calculation
        # Check for potential skin tone bias in dermatology
        if any("skin" in str(finding).lower() for finding in findings):
            ethnicity = demographics.get('ethnicity', '').lower()
            if ethnicity in ['black', 'asian', 'mixed']:
                # Acknowledge potential bias in skin lesion detection for darker skin tones
                bias_score = 0.15
        
        # Age bias in cardiac conditions
        age = demographics.get('age', 0)
        if any("cardiac" in str(finding).lower() for finding in findings):
            if age 
    file_type: "image" | "audio" | "document"
    image_category: "chest_xray" | "skin_lesion" | "ecg" | "general_photo"
    description: "Optional description of the file"
    ```
    
    **Response:**
    ```
    {
        "success": true,
        "file_id": "uuid4",
        "filename": "uploaded_filename.jpg",
        "file_type": "image",
        "processing_status": "processing",
        "analysis_results": {...}
    }
    ```
    """
    try:
        # Validate file size
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file selected"
            )
        
        # Read file content to check size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
            )
        
        # Validate file type
        content_type = file.content_type
        if file_type == FileType.IMAGE and content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image type: {content_type}"
            )
        elif file_type == FileType.AUDIO and content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio type: {content_type}"
            )
        
        # Generate unique filename
        file_id = uuid.uuid4()
        file_extension = os.path.splitext(file.filename)[1]
        storage_filename = f"{file_id}{file_extension}"
        storage_path = os.path.join(UPLOAD_DIRECTORY, storage_filename)
        
        # Save file to disk
        with open(storage_path, "wb") as f:
            f.write(file_content)
        
        # Create file record in database
        file_repo = FileRepository()
        file_data = {
            'filename': file.filename,
            'file_type': file_type.value,
            'file_size': len(file_content),
            'mime_type': content_type,
            'image_category': image_category.value if image_category else None,
            'audio_category': audio_category.value if audio_category else None,
            'storage_path': storage_path,
            'patient_id': uuid.UUID(patient_id) if patient_id else None,
            'assessment_id': uuid.UUID(assessment_id) if assessment_id else None,
            'session_id': uuid.UUID(session_id) if session_id else None
        }
        
        file_record = await file_repo.create_file_record(
            file_data=file_data,
            uploaded_by=current_user.id if current_user else None,
            db=db
        )
        
        # Start analysis if it's an image
        analysis_results = None
        if file_type == FileType.IMAGE and image_category:
            try:
                # Update status to processing
                await file_repo.update_processing_status(
                    file_record.file_id, 
                    ProcessingStatus.PROCESSING.value,
                    db=db
                )
                
                # Get patient demographics for bias monitoring
                demographics = None
                if current_user:
                    demographics = {
                        'age': 35,  # Mock - would get from patient record
                        'gender': 'unknown',
                        'ethnicity': 'unknown'
                    }
                
                # Perform image analysis
                analysis_result = await medical_image_analyzer.analyze_image(
                    file_id=file_record.file_id,
                    image_data=file_content,
                    image_category=image_category,
                    patient_demographics=demographics
                )
                
                # Store analysis results
                await file_repo.store_analysis_results(
                    file_record.file_id,
                    analysis_result.dict(),
                    analysis_result.confidence_score,
                    db
                )
                
                analysis_results = analysis_result.dict()
                
                logger.info(f"Image analysis completed for file: {file_record.file_id}")
                
            except Exception as analysis_error:
                logger.error(f"Image analysis failed: {analysis_error}")
                await file_repo.update_processing_status(
                    file_record.file_id,
                    ProcessingStatus.FAILED.value,
                    error=str(analysis_error),
                    db=db
                )
        
        response_data = {
            "success": True,
            "file_id": str(file_record.file_id),
            "filename": file.filename,
            "file_type": file_type.value,
            "file_size": len(file_content),
            "processing_status": ProcessingStatus.COMPLETED.value if analysis_results else ProcessingStatus.UPLOADED.value,
            "upload_time": file_record.created_at,
            "analysis_results": analysis_results
        }
        
        logger.info(f"File uploaded successfully: {file.filename} ({file_type.value})")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )

@router.get("/files/{file_id}")
async def get_file_info(
    file_id: str,
    current_user: Optional[UserModel] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Get file information and analysis results"""
    try:
        file_uuid = uuid.UUID(file_id)
        
        file_repo = FileRepository()
        file_record = await file_repo.get_file_by_id(file_uuid, db)
        
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return {
            "file_id": str(file_record.file_id),
            "filename": file_record.original_filename,
            "file_type": file_record.file_type,
            "file_size": file_record.file_size_bytes,
            "mime_type": file_record.mime_type,
            "image_category": file_record.image_category,
            "audio_category": file_record.audio_category,
            "processing_status": file_record.processing_status,
            "analysis_results": file_record.analysis_results,
            "confidence_score": file_record.confidence_score,
            "virus_scan_status": file_record.virus_scan_status,
            "created_at": file_record.created_at,
            "updated_at": file_record.updated_at
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file ID format"
        )
    except Exception as e:
        logger.error(f"Failed to get file info {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file information"
        )

@router.post("/files/{file_id}/reanalyze")
async def reanalyze_file(
    file_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Trigger reanalysis of an uploaded file"""
    try:
        file_uuid = uuid.UUID(file_id)
        
        file_repo = FileRepository()
        file_record = await file_repo.get_file_by_id(file_uuid, db)
        
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        if file_record.file_type != "image":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files can be reanalyzed currently"
            )
        
        # Read file from storage
        with open(file_record.storage_path, "rb") as f:
            file_content = f.read()
        
        # Update status to processing
        await file_repo.update_processing_status(
            file_uuid,
            ProcessingStatus.PROCESSING.value,
            db=db
        )
        
        # Perform analysis
        demographics = {
            'age': 35,  # Mock demographics
            'gender': 'unknown',
            'ethnicity': 'unknown'
        }
        
        image_category = ImageCategory(file_record.image_category)
        analysis_result = await medical_image_analyzer.analyze_image(
            file_id=file_uuid,
            image_data=file_content,
            image_category=image_category,
            patient_demographics=demographics
        )
        
        # Store updated results
        await file_repo.store_analysis_results(
            file_uuid,
            analysis_result.dict(),
            analysis_result.confidence_score,
            db
        )
        
        return {
            "success": True,
            "message": "File reanalyzed successfully",
            "analysis_results": analysis_result.dict()
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file ID format"
        )
    except Exception as e:
        logger.error(f"Failed to reanalyze file {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File reanalysis failed"
        )

@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a file and its analysis results"""
    try:
        file_uuid = uuid.UUID(file_id)
        
        file_repo = FileRepository()
        file_record = await file_repo.get_file_by_id(file_uuid, db)
        
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Delete physical file
        try:
            if os.path.exists(file_record.storage_path):
                os.remove(file_record.storage_path)
        except Exception as file_error:
            logger.warning(f"Failed to delete physical file: {file_error}")
        
        # Delete database record
        success = await file_repo.delete_file_record(file_uuid, db)
        
        if success:
            return {
                "success": True,
                "message": "File deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete file"
            )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file ID format"
        )
    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File deletion failed"
        )

@router.get("/files")
async def list_files(
    file_type: Optional[FileType] = None,
    processing_status: Optional[ProcessingStatus] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: Optional[UserModel] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """List uploaded files with optional filtering"""
    try:
        file_repo = FileRepository()
        
        # For now, return pending analysis files as a demo
        files = await file_repo.get_pending_analysis_files(
            file_type=file_type.value if file_type else None,
            limit=limit,
            db=db
        )
        
        files_data = []
        for file_record in files:
            files_data.append({
                "file_id": str(file_record.file_id),
                "filename": file_record.original_filename,
                "file_type": file_record.file_type,
                "file_size": file_record.file_size_bytes,
                "processing_status": file_record.processing_status,
                "confidence_score": file_record.confidence_score,
                "created_at": file_record.created_at,
                "has_analysis": bool(file_record.analysis_results)
            })
        
        return {
            "files": files_data,
            "total_count": len(files_data),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file list"
        )
```

### **📅 Afternoon Session (4 hours): 12:00 PM - 4:00 PM**

#### **Step 6.5: Create Database Migration for File Tables (30 minutes)**
Create: `data/database/migrations/create_file_tables.py`
```python
"""
Database migration for file storage tables
"""
from sqlalchemy import text
from data.database.connection_manager import db_manager

def upgrade():
    """Create file storage tables"""
    
    engine = db_manager.engine
    
    with engine.connect() as conn:
        # Drop existing table if exists
        conn.execute(text("DROP TABLE IF EXISTS medical_files CASCADE;"))
        
        # Create medical files table
        conn.execute(text("""
            CREATE TABLE medical_files (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                file_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(20) NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                mime_type VARCHAR(100) NOT NULL,
                image_category VARCHAR(50),
                audio_category VARCHAR(50),
                storage_path VARCHAR(500) NOT NULL,
                storage_bucket VARCHAR(100),
                processing_status VARCHAR(20) DEFAULT 'uploaded',
                processing_started_at TIMESTAMP WITH TIME ZONE,
                processing_completed_at TIMESTAMP WITH TIME ZONE,
                processing_error TEXT,
                analysis_results JSONB DEFAULT '{}',
                confidence_score FLOAT,
                virus_scan_status VARCHAR(20) DEFAULT 'pending',
                phi_detected BOOLEAN DEFAULT false,
                patient_id UUID,
                assessment_id UUID,
                session_id UUID,
                uploaded_by UUID,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))
        
        # Create indexes
        conn.execute(text("""
            CREATE INDEX idx_medical_files_file_id ON medical_files(file_id);
            CREATE INDEX idx_medical_files_type ON medical_files(file_type);
            CREATE INDEX idx_medical_files_status ON medical_files(processing_status);
            CREATE INDEX idx_medical_files_created_at ON medical_files(created_at);
            CREATE INDEX idx_medical_files_patient_id ON medical_files(patient_id);
            CREATE INDEX idx_medical_files_assessment_id ON medical_files(assessment_id);
            CREATE INDEX idx_medical_files_uploaded_by ON medical_files(uploaded_by);
        """))
        
        conn.commit()
        print("✅ File storage tables created successfully")

if __name__ == "__main__":
    upgrade()
```

#### **Step 6.6: Update Main App with File Routes (30 minutes)**
Update: `app.py`
```python
# Add this import
from api.files.upload_routes import router as files_router

# Add this line after other router includes
app.include_router(files_router, prefix="/api/v1/files", tags=["File Upload"])
```

#### **Step 6.7: Create Frontend File Upload Components (2 hours)**
Create: `frontend/src/components/FileUpload.jsx`
```jsx
import React, { useState, useCallback } from 'react';
import { PhotoIcon, DocumentIcon, SpeakerWaveIcon } from '@heroicons/react/24/outline';

const FILE_TYPES = {
  image: {
    label: 'Medical Image',
    icon: PhotoIcon,
    accept: 'image/jpeg,image/png,image/bmp,image/tiff',
    categories: [
      { value: 'chest_xray', label: 'Chest X-Ray' },
      { value: 'skin_lesion', label: 'Skin Lesion' },
      { value: 'ecg', label: 'ECG/EKG' },
      { value: 'general_photo', label: 'General Medical Photo' }
    ]
  },
  audio: {
    label: 'Audio Recording',
    icon: SpeakerWaveIcon,
    accept: 'audio/wav,audio/mp3,audio/m4a,audio/ogg',
    categories: [
      { value: 'heart_sounds', label: 'Heart Sounds' },
      { value: 'lung_sounds', label: 'Lung Sounds' },
      { value: 'speech_sample', label: 'Speech Sample' },
      { value: 'cough_recording', label: 'Cough Recording' }
    ]
  },
  document: {
    label: 'Medical Document',
    icon: DocumentIcon,
    accept: 'application/pdf,.doc,.docx',
    categories: []
  }
};

function FileUpload({ onUploadComplete, onUploadError, assessmentId }) {
  const [selectedType, setSelectedType] = useState('image');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [description, setDescription] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [selectedType, selectedCategory]);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;

    // Validate file size (50MB limit)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      onUploadError?.('File size exceeds 50MB limit');
      return;
    }

    // Validate file type
    const selectedFileType = FILE_TYPES[selectedType];
    const acceptedTypes = selectedFileType.accept.split(',');
    const isValidType = acceptedTypes.some(type => {
      if (type.startsWith('.')) {
        return file.name.toLowerCase().endsWith(type);
      }
      return file.type === type;
    });

    if (!isValidType) {
      onUploadError?.(`Invalid file type. Accepted types: ${selectedFileType.accept}`);
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('file_type', selectedType);
      
      if (selectedCategory) {
        if (selectedType === 'image') {
          formData.append('image_category', selectedCategory);
        } else if (selectedType === 'audio') {
          formData.append('audio_category', selectedCategory);
        }
      }
      
      if (description) {
        formData.append('description', description);
      }
      
      if (assessmentId) {
        formData.append('assessment_id', assessmentId);
      }

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch('/api/v1/files/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      
      // Add to uploaded files list
      setUploadedFiles(prev => [...prev, result]);
      
      // Reset form
      setDescription('');
      setSelectedCategory('');
      
      onUploadComplete?.(result);
      
    } catch (error) {
      console.error('Upload failed:', error);
      onUploadError?.(error.message);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const selectedFileType = FILE_TYPES[selectedType];
  const Icon = selectedFileType.icon;

  return (
    
      {/* File Type Selection */}
      
        
          File Type
        
        
          {Object.entries(FILE_TYPES).map(([type, config]) => {
            const TypeIcon = config.icon;
            return (
               {
                  setSelectedType(type);
                  setSelectedCategory('');
                }}
                className={`p-3 border rounded-lg flex flex-col items-center space-y-2 ${
                  selectedType === type
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                
                {config.label}
              
            );
          })}
        
      

      {/* Category Selection */}
      {selectedFileType.categories.length > 0 && (
        
          
            Category
          
           setSelectedCategory(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Select category...
            {selectedFileType.categories.map(category => (
              
                {category.label}
              
            ))}
          
        
      )}

      {/* Description */}
      
        
          Description (Optional)
        
         setDescription(e.target.value)}
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Describe the medical file (symptoms, location, etc.)"
        />
      

      {/* Upload Area */}
      
        
        
        {uploading ? (
          
            Uploading and analyzing...
            
              
            
            {uploadProgress}%
          
        ) : (
          <>
            
              Drop your {selectedFileType.label.toLowerCase()} here
            
            
              or click to browse files
            
            
            
            
            
              Choose File
            
            
            
              Max file size: 50MB
            
          
        )}
      

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        
          Recently Uploaded
          
            {uploadedFiles.map((file) => (
              
                
                  
                    {file.filename}
                    
                      {file.file_type} • {(file.file_size / 1024 / 1024).toFixed(1)} MB
                    
                    
                      Status: {file.processing_status}
                    
                  
                  
                  {file.analysis_results && (
                    
                      
                        Analysis Complete
                      
                    
                  )}
                
                
                {file.analysis_results && (
                  
                    Analysis Results:
                    
                      {file.analysis_results.medical_significance}
                    
                    
                      
                        Confidence: {(file.analysis_results.confidence_score * 100).toFixed(1)}%
                      
                      
                        Urgency: {file.analysis_results.urgency_level}
                      
                    
                  
                )}
              
            ))}
          
        
      )}
    
  );
}

export default FileUpload;
```

#### **Step 6.8: Create File Upload Page (1 hour)**
Create: `frontend/src/pages/FileUpload.jsx`
```jsx
import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';

function FileUploadPage() {
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState(''); // 'success' or 'error'

  const handleUploadComplete = (result) => {
    setMessage(`File uploaded successfully! ${result.analysis_results ? 'Analysis completed.' : 'Processing...'}`);
    setMessageType('success');
    
    // Clear message after 5 seconds
    setTimeout(() => setMessage(''), 5000);
  };

  const handleUploadError = (error) => {
    setMessage(`Upload failed: ${error}`);
    setMessageType('error');
    
    // Clear message after 5 seconds
    setTimeout(() => setMessage(''), 5000);
  };

  return (
    
      
        
        {/* Header */}
        
          
            Medical File Upload & Analysis
          
          
            Upload medical images, audio recordings, and documents for AI analysis
          
        

        {/* Success/Error Message */}
        {message && (
          
            
              {messageType === 'success' ? (
                
              ) : (
                
              )}
              
                {message}
              
            
          
        )}

        {/* Upload Component */}
        
          
        

        {/* Information Section */}
        
          
            
              
                
                  
                
              
              Medical Images
              
                Upload X-rays, skin photos, ECGs, and other medical images for AI-powered analysis and diagnosis assistance.
              
            
          

          
            
              
                
                  
                
              
              Audio Analysis
              
                Record heart sounds, lung sounds, cough patterns, and speech samples for audio-based medical assessment.
              
            
          

          
            
              
                
                  
                
              
              Instant Results
              
                Get immediate AI analysis with confidence scores, medical significance, and recommended next steps.
              
            
          
        

        {/* Safety Notice */}
        
          
            
              
                
              
            
            
              Privacy & Security Notice
              
                
                  All uploaded files are encrypted and processed securely. Medical images are analyzed using AI for educational 
                  and triage purposes only. This does not replace professional medical consultation. For medical emergencies, 
                  contact emergency services immediately.
                
              
            
          
        
      
    
  );
}

export default FileUploadPage;
```

### **📅 Evening Session (4 hours): 6:00 PM - 10:00 PM**

#### **Step 6.9: Add File Upload Route to App (30 minutes)**
Update: `frontend/src/App.jsx`
```jsx
// Add import
import FileUploadPage from './pages/FileUpload';

// Add route in the Routes section
} 
/>
```

Update: `frontend/src/components/Navbar.jsx` (add upload link)
```jsx
// Add upload link in navigation

  File Upload

```

#### **Step 6.10: Create Audio Processing Service (1 hour)**
Create: `services/audio_analysis_service.py`
```python
"""
Audio analysis service for medical audio processing
"""
import logging
import time
import wave
import io
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from datamodels.file_models_extended import AudioAnalysisResult, AudioCategory

logger = logging.getLogger(__name__)

class MedicalAudioAnalyzer:
    """Basic medical audio analysis using signal processing"""
    
    def __init__(self):
        # Mock audio models for Day 6
        self.models = {
            'heart_sounds': self._analyze_heart_sounds,
            'lung_sounds': self._analyze_lung_sounds,
            'speech_sample': self._analyze_speech,
            'cough_recording': self._analyze_cough,
            'general_audio': self._analyze_general_audio
        }
    
    async def analyze_audio(
        self,
        file_id: UUID,
        audio_data: bytes,
        audio_category: AudioCategory,
        patient_demographics: Optional[Dict] = None
    ) -> AudioAnalysisResult:
        """Analyze medical audio and return results"""
        
        start_time = time.time()
        
        try:
            # Basic audio processing
            audio_info = self._extract_audio_info(audio_data)
            
            # Perform analysis based on category
            analyzer_func = self.models.get(
                audio_category.value,
                self._analyze_general_audio
            )
            
            transcript, features, abnormalities, confidence, clinical_notes, follow_up = await analyzer_func(
                audio_data,
                audio_info,
                patient_demographics
            )
            
            # Calculate bias score
            bias_score = self._calculate_audio_bias_score(abnormalities, patient_demographics)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = AudioAnalysisResult(
                file_id=file_id,
                analysis_type=audio_category.value,
                transcript=transcript,
                audio_features=features,
                abnormalities_detected=abnormalities,
                confidence_score=confidence,
                clinical_notes=clinical_notes,
                follow_up_needed=follow_up,
                audio_quality=audio_info['quality'],
                duration_seconds=audio_info['duration'],
                sample_rate=audio_info['sample_rate'],
                processing_time_ms=processing_time,
                bias_score=bias_score
            )
            
            logger.info(f"Audio analysis completed for {file_id}: {audio_category}")
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis failed for {file_id}: {e}")
            
            # Return error result
            return AudioAnalysisResult(
                file_id=file_id,
                analysis_type=audio_category.value,
                transcript=None,
                audio_features={},
                abnormalities_detected=[],
                confidence_score=0.0,
                clinical_notes=f"Audio analysis failed: {str(e)}",
                follow_up_needed=True,
                audio_quality=0.0,
                duration_seconds=0.0,
                sample_rate=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                bias_score=0.0
            )
    
    def _extract_audio_info(self, audio_data: bytes) -> Dict:
        """Extract basic audio information"""
        try:
            # Mock audio info extraction
            # In real implementation, would use librosa or similar
            duration = 10.5  # Mock duration
            sample_rate = 44100  # Mock sample rate
            quality = 0.75  # Mock quality score
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'quality': quality,
                'channels': 1,
                'bit_depth': 16
            }
            
        except Exception as e:
            logger.error(f"Audio info extraction failed: {e}")
            return {
                'duration': 0.0,
                'sample_rate': 0,
                'quality': 0.0,
                'channels': 0,
                'bit_depth': 0
            }
    
    async def _analyze_heart_sounds(self, audio_data: bytes, audio_info: Dict, demographics: Optional[Dict]) -> Tuple:
        """Analyze heart sound recordings"""
        transcript = None  # Heart sounds don't have transcripts
        features = {
            'heart_rate_bpm': 72,  # Mock heart rate
            'rhythm_regularity': 0.85,  # Mock regularity score
            'murmur_detected': False,
            'gallop_detected': False
        }
        
        abnormalities = []
        confidence = 0.70
        
        # Mock heart sound analysis
        if features['heart_rate_bpm'] > 100:
            abnormalities.append("Tachycardia detected")
        elif features['heart_rate_bpm']  65:
            if not abnormalities:
                abnormalities.append("Age-related changes in heart sounds")
            confidence *= 0.9
        
        if abnormalities:
            clinical_notes = f"Heart sound analysis reveals {len(abnormalities)} finding(s). Consider ECG and echocardiogram for further evaluation."
            follow_up = True
        else:
            clinical_notes = "Heart sounds appear normal with regular rhythm and rate."
            follow_up = False
        
        return transcript, features, abnormalities, confidence, clinical_notes, follow_up
    
    async def _analyze_lung_sounds(self, audio_data: bytes, audio_info: Dict, demographics: Optional[Dict]) -> Tuple:
        """Analyze lung sound recordings"""
        transcript = None
        features = {
            'breathing_rate': 16,  # Mock breathing rate
            'wheeze_detected': False,
            'crackles_detected': False,
            'stridor_detected': False,
            'breath_sound_intensity': 0.8
        }
        
        abnormalities = []
        confidence = 0.65
        
        # Mock lung sound analysis
        if features['breathing_rate'] > 24:
            abnormalities.append("Tachypnea detected")
        
        # Simulate detection based on audio characteristics
        if audio_info['duration'] > 15:  # Longer recordings might capture abnormalities
            features['wheeze_detected'] = True
            abnormalities.append("Expiratory wheeze detected")
        
        if audio_info['quality']  Tuple:
        """Analyze speech patterns for neurological assessment"""
        # Mock speech-to-text
        transcript = "The patient spoke clearly for approximately 10 seconds during the recording."
        
        features = {
            'speech_rate_wpm': 150,  # Words per minute
            'articulation_clarity': 0.85,
            'voice_tremor': 0.1,
            'pause_frequency': 0.2,
            'volume_consistency': 0.9
        }
        
        abnormalities = []
        confidence = 0.60  # Lower confidence for speech analysis
        
        # Mock speech analysis
        if features['speech_rate_wpm']  200:
            abnormalities.append("Rapid speech rate detected")
        
        if features['articulation_clarity']  0.3:
            abnormalities.append("Voice tremor detected")
        
        if abnormalities:
            clinical_notes = f"Speech analysis reveals {len(abnormalities)} finding(s). Consider neurological evaluation."
            follow_up = True
        else:
            clinical_notes = "Speech patterns appear normal with clear articulation and appropriate rate."
            follow_up = False
        
        return transcript, features, abnormalities, confidence, clinical_notes, follow_up
    
    async def _analyze_cough(self, audio_data: bytes, audio_info: Dict, demographics: Optional[Dict]) -> Tuple:
        """Analyze cough recordings"""
        transcript = "Cough recording analyzed for pattern and characteristics."
        
        features = {
            'cough_frequency': 8,  # Coughs per minute
            'cough_type': 'dry',
            'intensity': 0.7,
            'wheeze_present': False,
            'productive': False
        }
        
        abnormalities = []
        confidence = 0.65
        
        # Mock cough analysis
        if features['cough_frequency'] > 10:
            abnormalities.append("Frequent coughing detected")
        
        if features['intensity'] > 0.8:
            abnormalities.append("Severe cough intensity")
            features['productive'] = True
        
        if audio_info['duration'] > 30:  # Long recording suggests persistent cough
            abnormalities.append("Persistent cough pattern")
        
        if abnormalities:
            clinical_notes = f"Cough analysis reveals {len(abnormalities)} concerning feature(s). Consider respiratory evaluation."
            follow_up = True
        else:
            clinical_notes = "Cough pattern appears normal with low frequency and intensity."
            follow_up = False
        
        return transcript, features, abnormalities, confidence, clinical_notes, follow_up
    
    async def _analyze_general_audio(self, audio_data: bytes, audio_info: Dict, demographics: Optional[Dict]) -> Tuple:
        """Analyze general medical audio"""
        transcript = "General audio recording processed."
        
        features = {
            'audio_type': 'general',
            'clarity': audio_info['quality'],
            'duration': audio_info['duration']
        }
        
        abnormalities = []
        confidence = 0.50  # Lower confidence for general audio
        
        clinical_notes = "General audio recording analyzed. Specific medical interpretation requires clinical context."
        follow_up = False
        
        return transcript, features, abnormalities, confidence, clinical_notes, follow_up
    
    def _calculate_audio_bias_score(self, abnormalities: List[str], demographics: Optional[Dict]) -> float:
        """Calculate bias score for audio analysis"""
        bias_score = 0.0
        
        if not demographics:
            return bias_score
        
        # Mock bias calculation for audio analysis
        # Account for potential accent or language bias in speech analysis
        if any("speech" in abnormality.lower() for abnormality in abnormalities):
            ethnicity = demographics.get('ethnicity', '').lower()
            if ethnicity not in ['white', 'unknown']:
                bias_score = 0.1  # Potential bias in speech recognition
        
        return min(1.0, bias_score)

# Global audio analyzer instance
medical_audio_analyzer = MedicalAudioAnalyzer()
```

#### **Step 6.11: Run Database Migration and Test (1 hour)**
```bash
# Run file tables migration
cd Fairdoc/backend
python data/database/migrations/create_file_tables.py

# Verify tables created
docker exec -it fairdoc_postgres psql -U fairdoc -d fairdoc_dev -c "\dt"

# Should show medical_files table
```

#### **Step 6.12: Test File Upload System (1.5 hours)**
Create: `tests/test_file_upload.py`
```python
"""
Test file upload and analysis functionality
"""
import requests
import base64
import io
from PIL import Image

BASE_URL = "http://localhost:8000/api/v1"

def get_test_token():
    """Get test token for file upload"""
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "remember_me": False
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("Failed to get test token")

def create_test_image():
    """Create a test medical image"""
    # Create a simple test image
    img = Image.new('RGB', (512, 512), color='white')
    
    # Add some mock medical content (simple shapes)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a mock chest X-ray outline
    draw.ellipse([100, 100, 400, 400], outline='gray', width=3)
    draw.rectangle([200, 150, 300, 350], outline='black', width=2)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_image_upload():
    """Test image upload and analysis"""
    print("🧪 Testing image upload and analysis...")
    
    token = get_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create test image
    image_data = create_test_image()
    
    # Prepare file upload
    files = {
        'file': ('test_xray.jpg', image_data, 'image/jpeg')
    }
    
    data = {
        'file_type': 'image',
        'image_category': 'chest_xray',
        'description': 'Test chest X-ray upload'
    }
    
    response = requests.post(
        f"{BASE_URL}/files/upload",
        headers=headers,
        files=files,
        data=data
    )
    
    print(f"Upload Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    assert "file_id" in result
    
    print("✅ Image upload test passed")
    return result["file_id"]

def test_file_info_retrieval(file_id):
    """Test file information retrieval"""
    print(f"\n🧪 Testing file info retrieval for {file_id}...")
    
    token = get_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/files/files/{file_id}", headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert "analysis_results" in result
    
    print("✅ File info retrieval test passed")

def test_file_list():
    """Test file listing"""
    print("\n🧪 Testing file listing...")
    
    token = get_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/files/files", headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert "files" in result
    
    print("✅ File listing test passed")

def test_file_reanalysis(file_id):
    """Test file reanalysis"""
    print(f"\n🧪 Testing file reanalysis for {file_id}...")
    
    token = get_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(f"{BASE_URL}/files/files/{file_id}/reanalyze", headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    
    print("✅ File reanalysis test passed")

if __name__ == "__main__":
    print("🚀 Starting file upload tests...")
    
    try:
        file_id = test_image_upload()
        test_file_info_retrieval(file_id)
        test_file_list()
        test_file_reanalysis(file_id)
        
        print("\n🎉 All file upload tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
```

#### **Step 6.13: Manual Frontend Test (1 hour)**
```bash
# Start backend with file upload support
cd Fairdoc/backend
./scripts/dev-start.sh

# Start frontend
cd Fairdoc/frontend
npm run dev

# Test file upload:
# 1. Navigate to http://localhost:3000/upload
# 2. Upload test medical images
# 3. Verify analysis results display
# 4. Test different file types and categories
# 5. Check error handling for invalid files
```

### **📈 Day 6 Success Metrics**
- ✅ File upload API accepts multiple formats (images, audio)
- ✅ Image analysis returns medical findings and recommendations
- ✅ Audio processing extracts features and abnormalities
- ✅ Frontend file upload interface functional
- ✅ Analysis results display with confidence scores
- ✅ File storage and retrieval working correctly
- ✅ Bias monitoring integrated into analysis pipeline

---

# **DAY 7: NHS EHR Integration & Production Setup** 🏥
## **🕐 12 Hours | Goal: NHS integration and deployment ready**

### **📅 Morning Session (4 hours): 6:00 AM - 10:00 AM**

#### **Step 7.1: Create NHS EHR Integration Service (1.5 hours)**
Create: `services/nhs_ehr_service.py`
```python
"""
NHS EHR integration service for Fairdoc AI
Handles FHIR R4 data exchange and GP Connect integration
"""
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
import aiohttp

from core.config import settings
from datamodels.nhs_ehr_models import NHSPatient, FHIRBundle, GPConnectRecord

logger = logging.getLogger(__name__)

class NHSEHRService:
    """NHS Electronic Health Record integration service"""
    
    def __init__(self):
        self.nhs_api_base = settings.NHS_API_BASE_URL
        self.client_id = settings.NHS_CLIENT_ID
        self.client_secret = settings.NHS_CLIENT_SECRET
        self.access_token = None
        self.token_expires_at = None
        
        # Mock data for development (remove in production)
        self.mock_mode = True
        self.mock_patients = self._generate_mock_patients()
    
    async def authenticate(self) -> str:
        """Authenticate with NHS Digital APIs"""
        try:
            if self.mock_mode:
                # Return mock token for development
                self.access_token = "mock_nhs_token_12345"
                self.token_expires_at = datetime.utcnow() + timedelta(hours=1)
                return self.access_token
            
            # Real NHS Digital OAuth2 authentication
            auth_url = f"{self.nhs_api_base}/oauth2/token"
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'patient:read observation:write'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=auth_data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data['access_token']
                        expires_in = token_data.get('expires_in', 3600)
                        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                        
                        logger.info("NHS API authentication successful")
                        return self.access_token
                    else:
                        error_text = await response.text()
                        raise Exception(f"NHS authentication failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"NHS authentication error: {e}")
            # Fallback to mock mode for development
            self.mock_mode = True
            return await self.authenticate()
    
    async def validate_nhs_number(self, nhs_number: str) -> bool:
        """Validate NHS number using Modulus 11 algorithm"""
        try:
            # Remove spaces and convert to string
            nhs_clean = str(nhs_number).replace(' ', '')
            
            if len(nhs_clean) != 10 or not nhs_clean.isdigit():
                return False
            
            # Modulus 11 check
            check_digit = int(nhs_clean[9])
            sum_digits = sum(int(nhs_clean[i]) * (10 - i) for i in range(9))
            remainder = sum_digits % 11
            
            if remainder == 0:
                return check_digit == 0
            elif remainder == 1:
                return False  # Invalid NHS number
            else:
                return check_digit == (11 - remainder)
                
        except Exception as e:
            logger.error(f"NHS number validation error: {e}")
            return False
    
    async def get_patient_record(self, nhs_number: str) -> Optional[Dict]:
        """Fetch patient record from NHS systems"""
        try:
            # Validate NHS number first
            if not await self.validate_nhs_number(nhs_number):
                raise ValueError("Invalid NHS number format")
            
            if self.mock_mode:
                return self._get_mock_patient_record(nhs_number)
            
            # Ensure we have valid authentication
            if not self.access_token or datetime.utcnow() >= self.token_expires_at:
                await self.authenticate()
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/fhir+json',
                'X-Request-ID': str(uuid4())
            }
            
            # Fetch patient demographics
            patient_url = f"{self.nhs_api_base}/fhir/Patient"
            params = {'identifier': f'https://fhir.nhs.uk/Id/nhs-number|{nhs_number}'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(patient_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        patient_data = await response.json()
                        
                        # Fetch additional data in parallel
                        tasks = [
                            self._fetch_patient_observations(nhs_number, headers, session),
                            self._fetch_patient_medications(nhs_number, headers, session),
                            self._fetch_patient_conditions(nhs_number, headers, session),
                            self._fetch_patient_allergies(nhs_number, headers, session)
                        ]
                        
                        observations, medications, conditions, allergies = await asyncio.gather(
                            *tasks, return_exceptions=True
                        )
                        
                        # Compile comprehensive record
                        record = {
                            'patient': patient_data,
                            'observations': observations if not isinstance(observations, Exception) else [],
                            'medications': medications if not isinstance(medications, Exception) else [],
                            'conditions': conditions if not isinstance(conditions, Exception) else [],
                            'allergies': allergies if not isinstance(allergies, Exception) else [],
                            'retrieved_at': datetime.utcnow().isoformat(),
                            'source': 'nhs_digital'
                        }
                        
                        logger.info(f"NHS patient record retrieved: {nhs_number}")
                        return record
                    
                    elif response.status == 404:
                        logger.warning(f"Patient not found: {nhs_number}")
                        return None
                    else:
                        error_text = await response.text()
                        raise Exception(f"NHS API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to fetch NHS patient record: {e}")
            return None
    
    async def submit_assessment_to_nhs(self, nhs_number: str, assessment_data: Dict) -> bool:
        """Submit AI assessment results to NHS systems"""
        try:
            if not await self.validate_nhs_number(nhs_number):
                raise ValueError("Invalid NHS number")
            
            if self.mock_mode:
                logger.info(f"Mock: Assessment submitted for {nhs_number}")
                return True
            
            # Create FHIR Observation resource
            observation = {
                "resourceType": "Observation",
                "id": str(uuid4()),
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "survey",
                        "display": "Survey"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "225399008",
                        "display": "AI-assisted medical triage"
                    }]
                },
                "subject": {
                    "reference": f"Patient/{nhs_number}",
                    "identifier": {
                        "system": "https://fhir.nhs.uk/Id/nhs-number",
                        "value": nhs_number
                    }
                },
                "effectiveDateTime": datetime.utcnow().isoformat(),
                "valueString": json.dumps(assessment_data),
                "component": [
                    {
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": "225336008",
                                "display": "Risk assessment"
                            }]
                        },
                        "valueString": assessment_data.get('risk_level', 'unknown')
                    },
                    {
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct", 
                                "code": "182836005",
                                "display": "Recommended action"
                            }]
                        },
                        "valueString": assessment_data.get('recommended_action', 'unknown')
                    }
                ],
                "extension": [
                    {
                        "url": "http://fairdoc.ai/fhir/extensions/bias-score",
                        "valueDecimal": assessment_data.get('bias_score', 0.0)
                    },
                    {
                        "url": "http://fairdoc.ai/fhir/extensions/confidence-score", 
                        "valueDecimal": assessment_data.get('confidence_score', 0.0)
                    }
                ]
            }
            
            # Submit to NHS via GP Connect
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/fhir+json',
                'X-Request-ID': str(uuid4())
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.nhs_api_base}/fhir/Observation",
                    headers=headers,
                    json=observation
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(f"Assessment submitted to NHS for patient {nhs_number}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to submit to NHS: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"NHS submission error: {e}")
            return False
    
    def _generate_mock_patients(self) -> Dict:
        """Generate mock patient data for development"""
        return {
            "1234567890": {
                "nhs_number": "1234567890",
                "name": "John Smith",
                "date_of_birth": "1980-05-15",
                "gender": "male",
                "address": "123 Main St, London, SW1A 1AA",
                "gp_practice": "City Medical Centre",
                "medical_history": [
                    "Hypertension",
                    "Type 2 Diabetes"
                ],
                "current_medications": [
                    "Lisinopril 10mg daily",
                    "Metformin 500mg twice daily"
                ],
                "allergies": ["Penicillin"],
                "last_blood_pressure": "140/90 mmHg",
                "last_hba1c": "7.2%",
                "last_updated": "2024-01-10"
            },
            "9876543210": {
                "nhs_number": "9876543210", 
                "name": "Sarah Johnson",
                "date_of_birth": "1992-08-22",
                "gender": "female",
                "address": "456 Oak Road, Manchester, M1 2AB",
                "gp_practice": "Riverside Surgery",
                "medical_history": [
                    "Asthma",
                    "Eczema"
                ],
                "current_medications": [
                    "Salbutamol inhaler PRN",
                    "Beclometasone inhaler twice daily"
                ],
                "allergies": ["Nuts", "Shellfish"],
                "last_peak_flow": "380 L/min",
                "last_updated": "2024-01-08"
            }
        }
    
    def _get_mock_patient_record(self, nhs_number: str) -> Optional[Dict]:
        """Get mock patient record for development"""
        patient = self.mock_patients.get(nhs_number)
        if not patient:
            return None
        
        return {
            "patient": {
                "resourceType": "Patient",
                "id": nhs_number,
                "identifier": [{
                    "system": "https://fhir.nhs.uk/Id/nhs-number",
                    "value": nhs_number
                }],
                "name": [{
                    "use": "official",
                    "text": patient["name"]
                }],
                "gender": patient["gender"],
                "birthDate": patient["date_of_birth"],
                "address": [{
                    "use": "home",
                    "text": patient["address"]
                }]
            },
            "observations": [
                {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "75367002",
                            "display": "Blood pressure"
                        }]
                    },
                    "valueString": patient.get("last_blood_pressure", "Unknown"),
                    "effectiveDateTime": "2024-01-10T10:00:00Z"
                }
            ],
            "medications": [
                {
                    "resourceType": "MedicationStatement",
                    "medicationCodeableConcept": {
                        "text": med
                    },
                    "status": "active"
                } for med in patient.get("current_medications", [])
            ],
            "conditions": [
                {
                    "resourceType": "Condition",
                    "code": {
                        "text": condition
                    },
                    "clinicalStatus": {
                        "coding": [{
                            "code": "active"
                        }]
                    }
                } for condition in patient.get("medical_history", [])
            ],
            "allergies": [
                {
                    "resourceType": "AllergyIntolerance",
                    "code": {
                        "text": allergy
                    },
                    "clinicalStatus": {
                        "coding": [{
                            "code": "active"
                        }]
                    }
                } for allergy in patient.get("allergies", [])
            ],
            "retrieved_at": datetime.utcnow().isoformat(),
            "source": "mock_nhs_data"
        }
    
    async def _fetch_patient_observations(self, nhs_number: str, headers: Dict, session) -> List:
        """Fetch patient observations from NHS"""
        try:
            obs_url = f"{self.nhs_api_base}/fhir/Observation"
            params = {'patient': nhs_number, '_count': 50}
            
            async with session.get(obs_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('entry', [])
                return []
        except Exception as e:
            logger.error(f"Failed to fetch observations: {e}")
            return []
    
    async def _fetch_patient_medications(self, nhs_number: str, headers: Dict, session) -> List:
        """Fetch patient medications from NHS"""
        try:
            med_url = f"{self.nhs_api_base}/fhir/MedicationStatement"
            params = {'patient': nhs_number, '_count': 50}
            
            async with session.get(med_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('entry', [])
                return []
        except Exception as e:
            logger.error(f"Failed to fetch medications: {e}")
            return []
    
    async def _fetch_patient_conditions(self, nhs_number: str, headers: Dict, session) -> List:
        """Fetch patient conditions from NHS"""
        try:
            cond_url = f"{self.nhs_api_base}/fhir/Condition"
            params = {'patient': nhs_number, '_count': 50}
            
            async with session.get(cond_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('entry', [])
                return []
        except Exception as e:
            logger.error(f"Failed to fetch conditions: {e}")
            return []
    
    async def _fetch_patient_allergies(self, nhs_number: str, headers: Dict, session) -> List:
        """Fetch patient allergies from NHS"""
        try:
            allergy_url = f"{self.nhs_api_base}/fhir/AllergyIntolerance"
            params = {'patient': nhs_number, '_count': 50}
            
            async with session.get(allergy_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('entry', [])
                return []
        except Exception as e:
            logger.error(f"Failed to fetch allergies: {e}")
            return []

# Global NHS EHR service instance
nhs_ehr_service = NHSEHRService()
```

#### **Step 7.2: Create NHS Integration API Routes (1 hour)**
Create: `api/nhs/integration_routes.py`
```python
"""
NHS integration API routes for Fairdoc AI
Handles NHS EHR data exchange and patient record management
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional

from core.dependencies import get_db, get_current_active_user
from services.nhs_ehr_service import nhs_ehr_service
from data.repositories.auth_repository import UserModel

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/patient/{nhs_number}")
async def get_nhs_patient_record(
    nhs_number: str,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get NHS patient record by NHS number
    
    **Parameters:**
    - nhs_number: 10-digit NHS number (e.g., "1234567890")
    
    **Response:**
    ```
    {
        "patient": {
            "nhs_number": "1234567890",
            "name": "John Smith",
            "date_of_birth": "1980-05-15",
            "gender": "male",
            "address": "123 Main St, London, SW1A 1AA"
        },
        "medical_history": [...],
        "current_medications": [...],
        "allergies": [...],
        "recent_observations": [...],
        "last_updated": "2024-01-10T10:00:00Z"
    }
    ```
    """
    try:
        # Validate NHS number format
        if not await nhs_ehr_service.validate_nhs_number(nhs_number):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid NHS number format"
            )
        
        # Fetch patient record from NHS systems
        patient_record = await nhs_ehr_service.get_patient_record(nhs_number)
        
        if not patient_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found in NHS systems"
            )
        
        # Log access for audit trail
        logger.info(f"NHS patient record accessed: {nhs_number} by user {current_user.email}")
        
        return {
            "success": True,
            "patient_record": patient_record,
            "accessed_by": current_user.email,
            "accessed_at": datetime.utcnow().isoformat()
        }

```