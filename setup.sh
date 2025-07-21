#!/bin/bash

# Medical AI Triage System - Complete Setup Script
# This script initializes the entire project with all dependencies

set -e

PROJECT_NAME="medical-ai-triage"
echo "🏥 Initializing Medical AI Triage System..."

# Initialize project with uv
echo "📦 Setting up Python project with uv..."
uv init $PROJECT_NAME --python 3.11
cd $PROJECT_NAME

# Create project structure
echo "📁 Creating project structure..."
mkdir -p {src/{app/{api,core,models,services,utils},tests},docker,scripts,docs,data/{raw,processed,models},logs}

# Create main application structure
mkdir -p src/app/{api/v1,core/context,models/database,services/{ai,chat,routing},utils}
mkdir -p src/tests/{unit,integration,e2e}

# Create Docker-related directories
mkdir -p docker/{postgres,redis,minio,ollama}

echo "✅ Project structure created!"

# Create pyproject.toml with all dependencies
cat > pyproject.toml << 'EOF'
[project]
name = "medical-ai-triage"
version = "0.1.0"
description = "AI-powered medical triage system for emergency healthcare"
authors = [{name = "Medical AI Team", email = "team@medical-ai.com"}]
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
    # Core Web Framework
    "fastapi[all]>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "starlette>=0.27.0",
    
    # Database & ORM
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "asyncpg>=0.29.0",
    "psycopg2-binary>=2.9.0",
    
    # Redis & Caching
    "redis>=5.0.0",
    "aioredis>=2.0.0",
    
    # AI & ML Dependencies
    "dspy-ai>=2.4.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.13",
    "langchain-openai>=0.0.5",
    "openai>=1.3.0",
    "transformers>=4.35.0",
    "torch>=2.1.0",
    "sentence-transformers>=2.2.0",
    
    # Vector Database & RAG
    "chromadb>=0.4.15",
    "faiss-cpu>=1.7.4",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    
    # Object Storage
    "minio>=7.2.0",
    "boto3>=1.34.0",
    
    # Authentication & Security
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.6",
    
    # Monitoring & Logging
    "structlog>=23.2.0",
    "prometheus-client>=0.19.0",
    "sentry-sdk[fastapi]>=1.38.0",
    
    # Validation & Serialization
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    
    # HTTP Client & API Integration
    "httpx>=0.25.0",
    "requests>=2.31.0",
    
    # Background Tasks
    "celery>=5.3.0",
    "kombu>=5.3.0",
    
    # Medical & Healthcare
    "pydicom>=2.4.0",
    "nibabel>=5.1.0",
    "scikit-learn>=1.3.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "rich>=13.6.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "httpx>=0.25.0",
    "factory-boy>=3.3.0",
    
    # Development Tools
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
    
    # Documentation
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.4.0",
    
    # Performance
    "locust>=2.17.0",
]

production = [
    "gunicorn>=21.2.0",
    "prometheus-fastapi-instrumentator>=6.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["src/tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/app",
    "--cov-report=html",
    "--cov-report=term-missing:skip-covered",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
disallow_any_generics = true
disallow_incomplete_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
EOF

echo "✅ pyproject.toml created with all dependencies!"

# Install all dependencies
echo "📦 Installing all dependencies..."
uv sync --all-extras

echo "✅ Dependencies installed!"
EOF

chmod +x setup.sh
