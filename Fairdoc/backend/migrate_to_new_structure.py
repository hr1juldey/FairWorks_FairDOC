#!/usr/bin/env python3
"""
# migrate_to_new_structure.py
# Script to migrate current Fairdoc backend to versioned structure
# v1/ - existing backend structure
# v2/ - new PostgreSQL/ChromaDB separated structure
"""

import os
import glob
import shutil
from pathlib import Path

def get_script_directory():
    """Get the directory where this script is located"""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return script_dir

def set_working_directory_to_script_location():
    """Change working directory to where the script is located (backend directory)"""
    script_dir = get_script_directory()
    
    print("🔍 Script location detection:")
    print("   Script file:", __file__)
    print("   Script absolute path:", os.path.abspath(__file__))
    print("   Script directory:", script_dir)
    print("   Current working directory (before):", os.getcwd())
    
    # Change to script directory
    os.chdir(script_dir)
    
    print("   Current working directory (after):", os.getcwd())
    print("   ✅ Changed working directory to script location")
    
    return script_dir

def find_backend_directory_from_hack():
    """Find backend directory when running from Hack directory"""
    current_dir = os.getcwd()
    
    print("🔍 Searching for backend directory from Hack location...")
    print("   Current directory:", current_dir)
    
    # Possible paths to backend directory from Hack
    possible_paths = [
        os.path.join(current_dir, "FairWorks_FairDOC", "Fairdoc", "backend"),
        os.path.join(current_dir, "Fairdoc", "backend"),
        os.path.join(current_dir, "backend"),
    ]
    
    # Also check script directory and relative paths
    script_dir = get_script_directory()
    possible_paths.extend([
        script_dir,  # Script is in backend directory
        os.path.join(script_dir, "..", "..", "Fairdoc", "backend"),
        os.path.join(script_dir, "..", "backend"),
        os.path.dirname(script_dir)  # Parent of script directory
    ])
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        app_py_path = os.path.join(abs_path, "app.py")
        
        print(f"   Checking: {abs_path}")
        
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            if os.path.exists(app_py_path) and os.path.isfile(app_py_path):
                print(f"   ✅ Found backend directory: {abs_path}")
                return abs_path
            else:
                print("   ❌ Directory exists but no app.py found")
        else:
            print("   ❌ Directory does not exist")
    
    return None

def debug_current_directory():
    """Debug current directory and list contents"""
    current_dir = os.getcwd()
    print("🔍 Debug Information:")
    print("   Current working directory:", current_dir)
    print("   Directory name:", os.path.basename(current_dir))
    
    # List all files and directories in current directory
    print("   Contents of current directory:")
    try:
        items = os.listdir(current_dir)
        for item in sorted(items):
            item_path = os.path.join(current_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"     📄 {item} ({size} bytes)")
            elif os.path.isdir(item_path):
                print(f"     📁 {item}/")
    except Exception as e:
        print(f"     ❌ Error listing directory: {e}")
    
    # Specifically check for app.py
    app_py_path = os.path.join(current_dir, "app.py")
    print(f"   Checking for app.py at: {app_py_path}")
    print(f"   app.py exists: {os.path.exists(app_py_path)}")
    print(f"   app.py is file: {os.path.isfile(app_py_path) if os.path.exists(app_py_path) else 'N/A'}")
    if os.path.exists(app_py_path):
        print(f"   app.py readable: {os.access(app_py_path, os.R_OK)}")
        print(f"   app.py size: {os.path.getsize(app_py_path)} bytes")

def verify_backend_directory():
    """Verify we're in the correct backend directory"""
    current_dir = os.getcwd()
    
    # Check multiple indicators that we're in the backend directory
    required_files = ["app.py"]
    required_dirs = ["core", "datamodels", "services", "MLmodels"]
    
    print("🔍 Verifying backend directory...")
    
    # Debug current directory
    debug_current_directory()
    
    missing_files = []
    missing_dirs = []
    
    # Check required files
    for req_file in required_files:
        file_path = os.path.join(current_dir, req_file)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            missing_files.append(req_file)
        else:
            print(f"   ✅ Found required file: {req_file}")
    
    # Check required directories
    for req_dir in required_dirs:
        dir_path = os.path.join(current_dir, req_dir)
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            missing_dirs.append(req_dir)
        else:
            print(f"   ✅ Found required directory: {req_dir}")
    
    if missing_files or missing_dirs:
        print("❌ Not in correct backend directory")
        print("   Current directory:", current_dir)
        if missing_files:
            print("   Missing files:", ", ".join(missing_files))
        if missing_dirs:
            print("   Missing directories:", ", ".join(missing_dirs))
        return False
    
    print("✅ Verified: Running from correct backend directory")
    return True

def create_versioned_directory_structure():
    """Create versioned directory structure with v1 and v2"""
    
    current_dir = os.getcwd()
    
    # Create main versioned backend directory
    versioned_backend_path = os.path.join(current_dir, "fairdoc-backend-versioned")
    
    # Remove existing directory if it exists
    if os.path.exists(versioned_backend_path):
        print("🗑️ Removing existing versioned directory:", versioned_backend_path)
        shutil.rmtree(versioned_backend_path)
    
    os.makedirs(versioned_backend_path, exist_ok=True)
    print("📁 Created versioned backend directory:", versioned_backend_path)
    
    # Create v1 directory (for existing backend)
    v1_path = os.path.join(versioned_backend_path, "v1")
    os.makedirs(v1_path, exist_ok=True)
    print("📁 Created v1 directory:", v1_path)
    
    # Create v2 directory structure (new structure)
    v2_path = os.path.join(versioned_backend_path, "v2")
    
    # Define the new v2 structure
    v2_structure = {
        "api": [
            "auth", "medical", "chat", "admin", "files", "nhs", "doctors", "rag"
        ],
        "core": [],
        "datamodels": [],
        "services": [],
        "MLmodels": [
            "classifiers", "embeddings", "ollama_models", "rag"
        ],
        "data": [
            "database", "repositories/postgres", "repositories/chromadb",
            "schemas/postgres", "schemas/chromadb", "schemas/redis",
            "migrations/postgres", "migrations/chromadb"
        ],
        "rag": [
            "indexing", "retrieval", "generation"
        ],
        "utils": [],
        "tools": [
            "data_generators", "testing", "deployment", "monitoring"
        ],
        "bkdocs": []
    }
    
    os.makedirs(v2_path, exist_ok=True)
    print("📁 Created v2 directory:", v2_path)
    
    # Create v2 directory structure
    for main_dir, subdirs in v2_structure.items():
        main_path = os.path.join(v2_path, main_dir)
        os.makedirs(main_path, exist_ok=True)
        
        # Create __init__.py in main directory
        init_file = os.path.join(main_path, "__init__.py")
        with open(init_file, 'w') as f:
            f.write(f"# {main_dir} module - v2\n")
        
        # Create subdirectories
        for subdir in subdirs:
            sub_path = os.path.join(main_path, subdir)
            os.makedirs(sub_path, exist_ok=True)
            sub_init_file = os.path.join(sub_path, "__init__.py")
            with open(sub_init_file, 'w') as f:
                f.write(f"# {subdir} module - v2\n")
    
    # Create shared directories at root level
    shared_dirs = ["tests", "docs", "scripts", "docker"]
    for shared_dir in shared_dirs:
        shared_path = os.path.join(versioned_backend_path, shared_dir)
        os.makedirs(shared_path, exist_ok=True)
        init_file = os.path.join(shared_path, "__init__.py")
        with open(init_file, 'w') as f:
            f.write(f"# {shared_dir} - shared across versions\n")
    
    return Path(versioned_backend_path), Path(v1_path), Path(v2_path)

def scan_existing_files_comprehensive():
    """Comprehensive file scanning using multiple methods"""
    
    current_dir = os.getcwd()
    existing_files = {}
    
    print("🔍 Starting comprehensive file scan...")
    print("   Working directory:", current_dir)
    
    # Directories to skip during scanning
    skip_dirs = {
        '__pycache__', '.git', '.pytest_cache', '.mypy_cache',
        'node_modules', '.venv', 'venv', 'env', '.env_backup',
        'fairdoc-backend-new', 'fairdoc-backend-versioned'  # Skip any existing output directories
    }
    
    # Method 1: Scan specific directories with os.walk
    scan_dirs = ['api', 'core', 'datamodels', 'services', 'MLmodels', 'data', 'utils', 'tools', 'bkdocs']
    
    for scan_dir in scan_dirs:
        dir_path = os.path.join(current_dir, scan_dir)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"   📂 Scanning directory: {scan_dir}")
            
            # Use os.walk for recursive scanning with proper dirs handling
            for root, dirs, files in os.walk(dir_path):
                # Get relative path of current directory
                rel_root = os.path.relpath(root, current_dir)
                rel_root = rel_root.replace(os.sep, '/')
                
                # Print current directory being processed
                print(f"     📁 Processing directory: {rel_root}")
                
                # Modify dirs in-place to skip unwanted directories
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                
                # Print subdirectories that will be processed
                if dirs:
                    print(f"     📁 Subdirectories to scan: {', '.join(dirs)}")
                
                # Process files in current directory
                if files:
                    print(f"     📄 Files in {rel_root}: {len(files)} files")
                    for file in files:
                        # Skip certain file types
                        if file.endswith(('.pyc', '.pyo', '.pyd', '.so', '.dll')):
                            print(f"       ⏭️ Skipping compiled file: {file}")
                            continue
                        
                        file_path = os.path.join(root, file)
                        # Get relative path from current directory
                        rel_path = os.path.relpath(file_path, current_dir)
                        # Normalize path separators for cross-platform compatibility
                        rel_path = rel_path.replace(os.sep, '/')
                        existing_files[rel_path] = file_path
                        print(f"       📄 Found: {rel_path}")
                else:
                    print(f"     📭 No files in {rel_root}")
        else:
            print(f"   ⚠️ Directory not found or not accessible: {scan_dir}")
    
    # Method 2: Scan root level files explicitly
    root_files = [
        '.env', '.env.local', '.env.prod', '.env.testing',
        'app.py', '__init__.py', 'requirements.txt', 'docker-compose.yml'
    ]
    
    print("   📂 Scanning root files...")
    for root_file in root_files:
        file_path = os.path.join(current_dir, root_file)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            existing_files[root_file] = file_path
            print(f"     📄 Found root file: {root_file}")
    
    # Method 3: Use glob patterns for additional coverage
    print("   📂 Using glob patterns for additional files...")
    glob_patterns = [
        '*.py', '*.yml', '*.yaml', '*.txt', '*.md', '*.json', '*.toml'
    ]
    
    for pattern in glob_patterns:
        for file_path in glob.glob(os.path.join(current_dir, pattern)):
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                # Skip if already found or if it's a file we want to skip
                if (filename not in existing_files and
                    not filename.endswith(('.pyc', '.pyo')) and
                    'migrate_to_new_structure.py' not in filename):
                    existing_files[filename] = file_path
                    print(f"     📄 Found with glob: {filename}")
    
    print(f"✅ Total files found: {len(existing_files)}")
    
    # Print summary of directory structure scanned
    unique_dirs = set()
    for rel_path in existing_files.keys():
        if '/' in rel_path:
            dir_part = '/'.join(rel_path.split('/')[:-1])
            unique_dirs.add(dir_part)
    
    print(f"📊 Summary: Scanned {len(unique_dirs)} directories, found {len(existing_files)} files")
    
    return existing_files

def copy_existing_files_to_v1(existing_files: dict, v1_path: Path):
    """Copy existing files to v1 directory (preserve original structure)"""
    
    print("📋 Copying existing files to v1 (original structure)...")
    
    copied_count = 0
    skipped_count = 0
    
    for old_rel_path, old_abs_path in existing_files.items():
        try:
            # Skip the migration script itself
            if 'migrate_to_new_structure.py' in old_rel_path:
                print(f"     ⏭️ Skipping migration script: {old_rel_path}")
                skipped_count += 1
                continue
            
            # Copy to v1 with same structure
            new_abs_path = os.path.join(str(v1_path), old_rel_path)
            
            # Create parent directories if they don't exist
            new_parent_dir = os.path.dirname(new_abs_path)
            os.makedirs(new_parent_dir, exist_ok=True)
            
            # Copy file using shutil
            shutil.copy2(old_abs_path, new_abs_path)
            
            print(f"     📄 Copied to v1: {old_rel_path}")
            copied_count += 1
            
        except Exception as e:
            print(f"     ⚠️ Error copying {old_rel_path}: {e}")
            skipped_count += 1
    
    print(f"✅ Copied {copied_count} files to v1, skipped {skipped_count} files")

def determine_new_location_v2(old_path: str) -> str:
    """Determine where an existing file should go in the v2 structure"""
    
    # File renaming rules for v2
    rename_rules = {
        'datamodels/chatmodels.py': 'datamodels/chat_models.py',
        'datamodels/medical_model.py': 'datamodels/medical_models.py',
    }
    
    # Normalize path separators
    old_path_normalized = old_path.replace('\\', '/')
    
    # Check if file needs renaming
    if old_path_normalized in rename_rules:
        return rename_rules[old_path_normalized]
    
    # For most files, the location stays the same in v2
    return old_path_normalized

def copy_selected_files_to_v2(existing_files: dict, v2_path: Path):
    """Copy selected existing files to v2 directory with new structure"""
    
    print("📋 Copying selected files to v2 (new structure)...")
    
    # Files to copy to v2 (core files that should be migrated)
    files_to_migrate_to_v2 = [
        'app.py',
        '.env', '.env.local', '.env.prod', '.env.testing',
        'requirements.txt',
        'docker-compose.yml'
    ]
    
    # Directories to migrate (will be restructured)
    dirs_to_migrate = ['core', 'datamodels', 'services', 'MLmodels', 'data', 'utils']
    
    copied_count = 0
    skipped_count = 0
    
    for old_rel_path, old_abs_path in existing_files.items():
        try:
            should_copy = False
            
            # Check if it's a root file we want to migrate
            if old_rel_path in files_to_migrate_to_v2:
                should_copy = True
            
            # Check if it's in a directory we want to migrate
            for migrate_dir in dirs_to_migrate:
                if old_rel_path.startswith(migrate_dir + '/') or old_rel_path.startswith(migrate_dir + '\\'):
                    should_copy = True
                    break
            
            if not should_copy:
                continue
            
            # Determine new location in v2
            new_rel_path = determine_new_location_v2(old_rel_path)
            new_abs_path = os.path.join(str(v2_path), new_rel_path)
            
            # Create parent directories if they don't exist
            new_parent_dir = os.path.dirname(new_abs_path)
            os.makedirs(new_parent_dir, exist_ok=True)
            
            # Copy file using shutil
            shutil.copy2(old_abs_path, new_abs_path)
            
            if old_rel_path != new_rel_path:
                print(f"     📄 Copied to v2 (renamed): {old_rel_path} -> {new_rel_path}")
            else:
                print(f"     📄 Copied to v2: {old_rel_path}")
            
            copied_count += 1
            
        except Exception as e:
            print(f"     ⚠️ Error copying {old_rel_path} to v2: {e}")
            skipped_count += 1
    
    print(f"✅ Copied {copied_count} files to v2, skipped {skipped_count} files")

def generate_v2_specific_files(v2_path: Path):
    """Generate v2-specific files with new structure"""
    
    print("🔧 Generating v2-specific files...")
    
    files_to_generate = {
        # Core files
        "core/websocket_manager.py": "# WebSocket connection management for real-time communications",
        "core/security.py": "# JWT, OAuth, and encryption utilities",
        "core/exceptions.py": "# Custom exception classes for Fairdoc backend",
        "core/dependencies.py": "# FastAPI dependency injection utilities",
        
        # Additional data models
        "datamodels/doctor_models.py": "# Doctor network and availability models",
        "datamodels/nice_models.py": "# NICE guidelines and disease classification models",
        "datamodels/rag_models.py": "# RAG search request and response models",
        
        # Services
        "services/auth_service.py": "# User authentication and session management",
        "services/medical_ai_service.py": "# AI orchestration for medical triage",
        "services/bias_detection_service.py": "# Real-time bias monitoring and correction",
        "services/chat_orchestrator.py": "# Multi-modal chat flow management",
        "services/ollama_service.py": "# Local LLM integration and routing",
        "services/notification_service.py": "# Real-time notification system",
        "services/nhs_ehr_service.py": "# NHS EHR integration and FHIR operations",
        "services/doctor_network_service.py": "# Doctor availability and consultation routing",
        "services/nice_service.py": "# NICE guidelines integration service",
        "services/rag_service.py": "# RAG search and document retrieval orchestration",
        
        # API routes
        "api/auth/routes.py": "# Authentication API endpoints",
        "api/medical/routes.py": "# Medical triage API endpoints",
        "api/chat/routes.py": "# WebSocket chat API endpoints",
        "api/admin/routes.py": "# Admin and monitoring API endpoints",
        "api/files/routes.py": "# File upload and processing API endpoints",
        "api/nhs/routes.py": "# NHS EHR integration API endpoints",
        "api/doctors/routes.py": "# Doctor network API endpoints",
        "api/rag/routes.py": "# RAG search and retrieval API endpoints",
        
        # ML Models
        "MLmodels/classifiers/triage_classifier.py": "# Primary medical triage classification model",
        "MLmodels/classifiers/risk_classifier.py": "# Patient risk assessment classifier",
        "MLmodels/classifiers/bias_classifier.py": "# Bias detection and classification models",
        "MLmodels/embeddings/medical_embeddings.py": "# Medical text embedding generation",
        "MLmodels/embeddings/embedding_generator.py": "# Vector embedding generation for RAG",
        "MLmodels/embeddings/similarity_search.py": "# ChromaDB similarity search operations",
        "MLmodels/ollama_models/clinical_model.py": "# Clinical reasoning LLM interface",
        "MLmodels/ollama_models/chat_model.py": "# Conversational AI model interface",
        "MLmodels/ollama_models/classification_model.py": "# Text classification LLM interface",
        "MLmodels/rag/retrieval_model.py": "# Document retrieval model for RAG",
        "MLmodels/rag/ranking_model.py": "# Result ranking and scoring model",
        "MLmodels/rag/context_fusion.py": "# Context combination and fusion logic",
        "MLmodels/model_manager.py": "# ML model loading, caching, and lifecycle management",
        
        # Database managers
        "data/database/postgres_manager.py": "# PostgreSQL connection and session management",
        "data/database/chromadb_manager.py": "# ChromaDB vector database operations",
        "data/database/redis_manager.py": "# Redis cache and session management",
        
        # PostgreSQL repositories
        "data/repositories/postgres/auth_repository.py": "# User authentication data access layer",
        "data/repositories/postgres/medical_repository.py": "# Medical assessment data access layer",
        "data/repositories/postgres/chat_repository.py": "# Chat history data access layer",
        "data/repositories/postgres/bias_repository.py": "# Bias metrics data access layer",
        "data/repositories/postgres/nhs_ehr_repository.py": "# NHS EHR data access layer",
        "data/repositories/postgres/doctor_repository.py": "# Doctor records data access layer",
        "data/repositories/postgres/nice_repository.py": "# NICE guidelines data access layer",
        
        # ChromaDB repositories
        "data/repositories/chromadb/rag_repository.py": "# RAG document storage and retrieval",
        "data/repositories/chromadb/embedding_repository.py": "# Vector embeddings management",
        "data/repositories/chromadb/similarity_repository.py": "# Similarity search operations",
        
        # Database schemas
        "data/schemas/postgres/user_schemas.py": "# PostgreSQL user table schemas",
        "data/schemas/postgres/medical_schemas.py": "# PostgreSQL medical table schemas",
        "data/schemas/postgres/nhs_ehr_schemas.py": "# PostgreSQL NHS EHR table schemas",
        "data/schemas/postgres/doctor_schemas.py": "# PostgreSQL doctor table schemas",
        "data/schemas/postgres/nice_schemas.py": "# PostgreSQL NICE data table schemas",
        "data/schemas/chromadb/rag_collections.py": "# ChromaDB RAG document collections",
        "data/schemas/chromadb/medical_knowledge_collections.py": "# ChromaDB medical knowledge vectors",
        "data/schemas/chromadb/conversation_collections.py": "# ChromaDB chat context vectors",
        "data/schemas/chromadb/similarity_collections.py": "# ChromaDB similarity search collections",
        "data/schemas/redis/cache_schemas.py": "# Redis cache key patterns and schemas",
        "data/schemas/redis/session_schemas.py": "# Redis session management schemas",
        
        # Migrations
        "data/migrations/postgres/001_initial_tables.py": "# Initial PostgreSQL database tables creation",
        "data/migrations/postgres/002_nhs_ehr_tables.py": "# NHS EHR integration tables",
        "data/migrations/postgres/003_doctor_tables.py": "# Doctor network tables",
        "data/migrations/postgres/004_nice_tables.py": "# NICE guidelines tables",
        "data/migrations/chromadb/init_collections.py": "# Initialize ChromaDB collections for RAG",
        "data/migrations/chromadb/setup_embeddings.py": "# Setup embedding models and indexes",
        
        # RAG components
        "rag/indexing/document_processor.py": "# Document processing and preparation for RAG indexing",
        "rag/indexing/chunk_splitter.py": "# Text chunking strategies for optimal retrieval",
        "rag/indexing/metadata_extractor.py": "# Extract and structure document metadata",
        "rag/retrieval/vector_retriever.py": "# Vector-based document retrieval",
        "rag/retrieval/hybrid_retriever.py": "# Hybrid search combining vector and keyword",
        "rag/retrieval/context_retriever.py": "# Context-aware document retrieval",
        "rag/generation/prompt_templates.py": "# RAG prompt templates for different use cases",
        "rag/generation/context_formatter.py": "# Format retrieved context for LLM consumption",
        "rag/generation/response_synthesizer.py": "# Synthesize final responses from context and queries",
        "rag/rag_pipeline.py": "# Main RAG orchestration and pipeline management",
        
        # Utilities
        "utils/medical_utils.py": "# Medical data processing and validation utilities",
        "utils/text_processing.py": "# NLP preprocessing and text analysis utilities",
        "utils/image_processing.py": "# Medical image analysis and processing utilities",
        "utils/validation_utils.py": "# Data validation and sanitization utilities",
        "utils/monitoring_utils.py": "# Logging, monitoring, and metrics utilities",
        "utils/nhs_utils.py": "# NHS data formatting and integration utilities",
        "utils/nice_utils.py": "# NICE guidelines processing utilities",
        "utils/rag_utils.py": "# RAG helper functions and utilities",
        
        # Tools and testing
        "tools/data_generators/postgres_seed_data.py": "# Generate seed data for PostgreSQL development",
        "tools/data_generators/chromadb_seed_data.py": "# Generate seed data for ChromaDB development",
        "tools/testing/postgres_fixtures.py": "# PostgreSQL test fixtures and helpers",
        "tools/testing/chromadb_fixtures.py": "# ChromaDB test fixtures and helpers",
        "tools/testing/rag_test_utils.py": "# RAG testing utilities and helpers",
        "tools/deployment/postgres_setup.sh": "# PostgreSQL deployment and configuration script",
        "tools/deployment/chromadb_setup.sh": "# ChromaDB deployment and configuration script",
        "tools/deployment/rag_index_builder.py": "# Build and maintain RAG indexes for production",
        "tools/monitoring/postgres_monitor.py": "# PostgreSQL performance monitoring",
        "tools/monitoring/chromadb_monitor.py": "# ChromaDB performance monitoring",
        "tools/monitoring/rag_performance_monitor.py": "# RAG system performance monitoring",
        
        # V2-specific config files
        "requirements-rag.txt": "# RAG-specific Python dependencies for v2",
        "docker-compose.rag.yml": "# Docker compose for RAG services (ChromaDB, embeddings)",
        "README-v2.md": "# Fairdoc v2 Backend - PostgreSQL/ChromaDB Architecture"
    }
    
    generated_count = 0
    
    # Generate files only if they don't already exist
    for file_path, description in files_to_generate.items():
        full_path = os.path.join(str(v2_path), file_path)
        
        # Only generate if file doesn't exist
        if not os.path.exists(full_path):
            # Create parent directories
            parent_dir = os.path.dirname(full_path)
            os.makedirs(parent_dir, exist_ok=True)
            
            # Generate file content
            filename = os.path.basename(file_path)
            if file_path.endswith('.md'):
                content = f"# {file_path}\n\n{description}\n\n## TODO\n\nImplement {filename} functionality\n"
            elif file_path.endswith('.txt'):
                content = f"# {description}\n"
            elif file_path.endswith('.yml'):
                content = f"# {description}\nversion: '3.8'\n# TODO: Add service definitions\n"
            elif file_path.endswith('.sh'):
                content = f"#!/bin/bash\n# {file_path}\n# {description}\n\n# TODO: Implement setup script\necho 'Setup script not implemented yet'\n"
            else:
                content = f'"""\n{file_path}\n{description}\n"""\n\n# TODO: Implement {filename} functionality\npass\n'
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"     🆕 Generated v2: {file_path}")
            generated_count += 1
        else:
            print(f"     ⏭️ Skipped v2 (exists): {file_path}")
    
    print(f"✅ Generated {generated_count} new v2 files")

def create_version_documentation(versioned_backend_path: Path):
    """Create documentation for the versioned structure"""
    
    print("📝 Creating version documentation...")
    
    # Main README
    main_readme = os.path.join(str(versioned_backend_path), "README.md")
    readme_content = """# Fairdoc AI Backend - Versioned Architecture

This directory contains versioned backend implementations for Fairdoc AI.

## Structure

```
fairdoc-backend-versioned/
├── v1/                     # Original backend structure
│   ├── api/
│   ├── core/
│   ├── datamodels/
│   ├── services/
│   ├── MLmodels/
│   ├── data/
│   ├── utils/
│   ├── tools/
│   └── app.py
├── v2/                     # New PostgreSQL/ChromaDB structure
│   ├── api/
│   ├── core/
│   ├── datamodels/
│   ├── services/
│   ├── MLmodels/
│   ├── data/
│   ├── rag/               # New RAG components
│   ├── utils/
│   ├── tools/
│   └── app.py
├── tests/                  # Shared tests
├── docs/                   # Shared documentation
├── scripts/                # Shared scripts
└── docker/                 # Shared Docker configurations
```

## Version Differences

### v1 (Original)
- Original backend structure
- All data in PostgreSQL
- Basic ML models
- Simple file structure

### v2 (New Architecture)
- Separated PostgreSQL and ChromaDB
- RAG components for document processing
- Enhanced AI orchestration
- Bias monitoring system
- NHS EHR integration
- Doctor network services

## Usage

### Running v1
```
cd v1/
python app.py
```

### Running v2
```
cd v2/
python app.py
```

## Migration Notes

This structure was generated by the migration script to preserve the original v1 backend while introducing the new v2 architecture.

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(main_readme, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # V1 README
    v1_readme = os.path.join(str(versioned_backend_path), "v1", "README-v1.md")
    v1_content = """# Fairdoc AI v1 Backend

This is the original backend structure preserved during migration.

## Features
- Original API structure
- Basic medical AI
- PostgreSQL data storage
- Simple file upload
- Basic authentication

## Usage
```
python app.py
```

**Note:** This is the legacy version. New features are developed in v2.
"""
    
    with open(v1_readme, 'w', encoding='utf-8') as f:
        f.write(v1_content)
    
    print("✅ Version documentation created")

def validate_versioned_migration(versioned_backend_path: Path, v1_path: Path, v2_path: Path, existing_files: dict):
    """Validate the versioned migration was successful"""
    
    print("🔍 Validating versioned migration...")
    
    validation_passed = True
    
    # Check v1 structure
    print("   📋 Validating v1 structure...")
    v1_critical_files = ['app.py', 'core/config.py', 'datamodels/__init__.py']
    
    for critical_file in v1_critical_files:
        v1_file_path = os.path.join(str(v1_path), critical_file)
        if not os.path.exists(v1_file_path):
            print(f"     ❌ v1 critical file missing: {critical_file}")
            validation_passed = False
        else:
            print(f"     ✅ v1 critical file found: {critical_file}")
    
    # Check v2 structure
    print("   📋 Validating v2 structure...")
    v2_expected_dirs = ['api', 'core', 'datamodels', 'services', 'MLmodels', 'data', 'rag', 'utils', 'tools']
    
    for expected_dir in v2_expected_dirs:
        v2_dir_path = os.path.join(str(v2_path), expected_dir)
        if not os.path.exists(v2_dir_path) or not os.path.isdir(v2_dir_path):
            print(f"     ❌ v2 expected directory missing: {expected_dir}")
            validation_passed = False
        else:
            print(f"     ✅ v2 directory found: {expected_dir}")
    
    # Check shared directories
    print("   📋 Validating shared structure...")
    shared_dirs = ['tests', 'docs', 'scripts', 'docker']
    
    for shared_dir in shared_dirs:
        shared_path = os.path.join(str(versioned_backend_path), shared_dir)
        if not os.path.exists(shared_path):
            print(f"     ❌ Shared directory missing: {shared_dir}")
            validation_passed = False
        else:
            print(f"     ✅ Shared directory found: {shared_dir}")
    
    # Validate file counts
    v1_files = len([f for f in existing_files.keys() if os.path.exists(os.path.join(str(v1_path), f))])
    print(f"     📊 Files in v1: {v1_files}/{len(existing_files)}")
    
    # Count v2 files
    v2_file_count = 0
    for _root, _dirs, files in os.walk(str(v2_path)):
        v2_file_count += len(files)
    print(f"     📊 Files in v2: {v2_file_count}")
    
    if validation_passed:
        print("🎉 Versioned migration validation passed!")
    else:
        print("⚠️ Versioned migration validation failed - please review")
    
    return validation_passed

def main():
    """Main migration function with versioned structure creation"""
    print("🚀 Starting Fairdoc backend VERSIONED structure migration...")
    print("🎯 Creating v1/ (existing) and v2/ (new PostgreSQL/ChromaDB structure)")
    print("🔍 Detecting execution environment...")
    
    current_dir = os.getcwd()
    script_dir = get_script_directory()
    
    print(f"   Current working directory: {current_dir}")
    print(f"   Script location: {script_dir}")
    print(f"   Current directory name: {os.path.basename(current_dir)}")
    
    backend_dir = None
    
    # Strategy 1: Check if we're already in backend directory
    if verify_backend_directory():
        print("✅ Already in backend directory")
        backend_dir = current_dir
    else:
        print("❌ Not in backend directory, searching...")
        
        # Strategy 2: Try to find backend directory from current location
        found_backend = find_backend_directory_from_hack()
        
        if found_backend:
            backend_dir = found_backend
            print(f"✅ Found backend directory: {backend_dir}")
            os.chdir(backend_dir)
            print(f"📁 Changed working directory to: {os.getcwd()}")
        else:
            # Strategy 3: Use script location as fallback
            print("⚠️ Could not find backend directory, trying script location...")
            try:
                set_working_directory_to_script_location()
                if verify_backend_directory():
                    backend_dir = os.getcwd()
                    print("✅ Script location is backend directory")
            except Exception as e:
                print(f"❌ Script location strategy failed: {e}")
    
    # Final verification
    if not backend_dir or not verify_backend_directory():
        print("❌ Could not locate backend directory automatically")
        print("\n🔍 Manual path detection:")
        print("   Current directory:", os.getcwd())
        print("   Script directory:", script_dir)
        print("\n💡 Possible solutions:")
        print("   1. Copy the script to Fairdoc/backend/ directory")
        print("   2. Navigate to backend directory before running:")
        print("      cd FairWorks_FairDOC/Fairdoc/backend/")
        print("      python migrate_to_new_structure.py")
        return
    
    print(f"✅ Ready to migrate from: {os.getcwd()}")
    
    try:
        # Scan existing files
        print("\n🔍 Scanning existing files...")
        existing_files = scan_existing_files_comprehensive()
        
        if not existing_files:
            print("❌ No files found to migrate!")
            return
        
        # Create versioned directory structure
        print("\n📁 Creating versioned directory structure...")
        versioned_backend_path, v1_path, v2_path = create_versioned_directory_structure()
        
        # Copy existing files to v1 (preserve original structure)
        print("\n📋 Copying existing files to v1...")
        copy_existing_files_to_v1(existing_files, v1_path)
        
        # Copy selected files to v2 (new structure)
        print("\n📋 Copying selected files to v2...")
        copy_selected_files_to_v2(existing_files, v2_path)
        
        # Generate v2-specific files
        print("\n🔧 Generating v2-specific files...")
        generate_v2_specific_files(v2_path)
        
        # Create documentation
        print("\n📝 Creating documentation...")
        create_version_documentation(versioned_backend_path)
        
        # Validate migration
        print("\n🔍 Validating versioned migration...")
        validation_success = validate_versioned_migration(versioned_backend_path, v1_path, v2_path, existing_files)
        
        print("\n✅ VERSIONED MIGRATION COMPLETED!")
        print("📂 Versioned backend structure created in:", str(versioned_backend_path))
        print("📊 Total existing files found:", len(existing_files))
        print("📝 Structure created:")
        print(f"   📁 v1/ - Original backend ({len([f for f in existing_files.keys()])} files)")
        print("   📁 v2/ - New PostgreSQL/ChromaDB structure")
        print("   📁 tests/ - Shared testing")
        print("   📁 docs/ - Shared documentation")
        print("   📁 scripts/ - Shared scripts")
        print("   📁 docker/ - Shared Docker configs")
        
        print("\n🔄 Next steps:")
        print("   1. Review the versioned structure:")
        print(f"      cd {versioned_backend_path.name}")
        print("   2. Test v1 backend:")
        print("      cd v1/ && python app.py")
        print("   3. Test v2 backend:")
        print("      cd v2/ && python app.py")
        print("   4. Replace original backend when ready:")
        print("      mv backend backend-backup")
        print(f"      mv {versioned_backend_path.name} backend")
        
        if validation_success:
            print("\n🎉 VERSIONED MIGRATION COMPLETED SUCCESSFULLY!")
            print("\n📋 Summary:")
            print("   ✅ v1/ contains your original backend (preserved)")
            print("   ✅ v2/ contains new PostgreSQL/ChromaDB architecture")
            print("   ✅ Shared directories created for tests, docs, scripts")
            print("   ✅ Documentation generated")
            print("   ✅ All validations passed")
        else:
            print("\n⚠️ Please review validation errors before proceeding")
        
    except Exception as e:
        print(f"\n❌ Versioned migration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
