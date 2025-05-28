#!/usr/bin/env python3
"""
# migrate_to_new_structure.py
# Script to migrate current Fairdoc backend to versioned structure with unified API access
# v1/ - existing backend structure (independent)
# v2/ - new PostgreSQL/ChromaDB separated structure (independent)
# Root - unified API with shared configs
"""

import os
import glob
import shutil
from pathlib import Path
from datetime import datetime

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

def create_unified_backend_structure():
    """Create unified backend structure with v1, v2, and root app"""
    
    current_dir = os.getcwd()
    
    # Create backup of current backend
    backup_dir = os.path.join(current_dir, "backend-backup-original")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    # Copy current backend to backup
    shutil.copytree(current_dir, backup_dir, ignore=shutil.ignore_patterns('backend-backup-*', '__pycache__', '*.pyc'))
    print(f"📁 Created backup at: {backup_dir}")
    
    # Create v1 directory (for existing backend)
    v1_path = os.path.join(current_dir, "v1")
    if os.path.exists(v1_path):
        shutil.rmtree(v1_path)
    os.makedirs(v1_path, exist_ok=True)
    print(f"📁 Created v1 directory: {v1_path}")
    
    # Create v2 directory structure (new structure)
    v2_path = os.path.join(current_dir, "v2")
    if os.path.exists(v2_path):
        shutil.rmtree(v2_path)
    
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
    print(f"📁 Created v2 directory: {v2_path}")
    
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
    
    return Path(current_dir), Path(v1_path), Path(v2_path)

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
        'v1', 'v2', 'backend-backup-original'  # Skip version directories
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
    
    print(f"✅ Total files found: {len(existing_files)}")
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
            
            # Skip shared config files (keep in root)
            if old_rel_path in ['.env', '.env.local', '.env.prod', '.env.testing', 'docker-compose.yml', 'requirements.txt']:
                print(f"     ⏭️ Keeping in root: {old_rel_path}")
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
    
    # Directories to migrate (will be restructured)
    dirs_to_migrate = ['core', 'datamodels', 'services', 'MLmodels', 'data', 'utils']
    
    copied_count = 0
    skipped_count = 0
    
    for old_rel_path, old_abs_path in existing_files.items():
        try:
            should_copy = False
            
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

def create_unified_app_files(backend_root: Path, v1_path: Path, v2_path: Path):
    """Create unified app files that enable both independent and mounted access"""
    
    print("🔧 Creating unified app structure...")
    
    # 1. Create main app.py (mounts both v1 and v2)
    main_app_content = '''"""
Main Fairdoc Backend Application
Serves both v1 and v2 APIs with unified access
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import version-specific apps
from v1.app_v1 import app as app_v1
from v2.app_v2 import app as app_v2

# Create main application
app = FastAPI(
    title="Fairdoc AI - Healthcare Triage Platform",
    description="Unified API serving both v1 (legacy) and v2 (modern) endpoints",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount version-specific applications
app.mount("/api/v1", app_v1)
app.mount("/api/v2", app_v2)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fairdoc AI Healthcare Triage Platform",
        "versions": {
            "v1": {
                "docs": "/api/v1/docs",
                "description": "Legacy API with original structure"
            },
            "v2": {
                "docs": "/api/v2/docs",
                "description": "Modern API with PostgreSQL/ChromaDB architecture"
            }
        },
        "unified_docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "versions": ["v1", "v2"],
        "services": ["api", "database", "ai_models"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    
    # Write main app.py
    with open(os.path.join(str(backend_root), "app.py"), 'w') as f:
        f.write(main_app_content)
    
    # 2. Create v1/app_v1.py (independent v1 app)
    v1_app_content = '''"""
Fairdoc v1 Backend Application
Original backend structure preserved
Can run independently or as part of unified app
"""
from fastapi import FastAPI

# Create v1 application
app = FastAPI(
    title="Fairdoc AI v1",
    description="Legacy API with original backend structure",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Import v1-specific routes
# from api.auth.routes import router as auth_router
# from api.medical.routes import router as medical_router

# Include routers
# app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# app.include_router(medical_router, prefix="/medical", tags=["Medical"])

@app.get("/")
async def v1_root():
    return {
        "message": "Fairdoc AI v1 - Legacy API",
        "version": "1.0.0",
        "structure": "original",
        "features": [
            "Basic medical triage",
            "PostgreSQL storage",
            "Original data models"
        ]
    }

@app.get("/health")
async def v1_health():
    return {
        "status": "healthy",
        "version": "v1",
        "database": "postgresql",
        "ai_models": "basic"
    }

# For independent development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
'''
    
    # Write v1/app_v1.py
    with open(os.path.join(str(v1_path), "app_v1.py"), 'w') as f:
        f.write(v1_app_content)
    
    # 3. Create v2/app_v2.py (independent v2 app)
    v2_app_content = '''"""
Fairdoc v2 Backend Application
Modern architecture with PostgreSQL/ChromaDB separation
Can run independently or as part of unified app
"""
from fastapi import FastAPI

# Create v2 application
app = FastAPI(
    title="Fairdoc AI v2",
    description="Modern API with PostgreSQL/ChromaDB separated architecture",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Import v2-specific routes
# from api.auth.routes import router as auth_router
# from api.medical.routes import router as medical_router
# from api.rag.routes import router as rag_router
# from api.nhs.routes import router as nhs_router

# Include routers
# app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# app.include_router(medical_router, prefix="/medical", tags=["Medical"])
# app.include_router(rag_router, prefix="/rag", tags=["RAG Search"])
# app.include_router(nhs_router, prefix="/nhs", tags=["NHS Integration"])

@app.get("/")
async def v2_root():
    return {
        "message": "Fairdoc AI v2 - Modern Architecture",
        "version": "2.0.0",
        "structure": "postgresql_chromadb_separated",
        "features": [
            "Advanced AI triage",
            "PostgreSQL + ChromaDB",
            "RAG document processing",
            "NHS EHR integration",
            "Real-time bias monitoring",
            "Doctor network services"
        ]
    }

@app.get("/health")
async def v2_health():
    return {
        "status": "healthy",
        "version": "v2",
        "databases": ["postgresql", "chromadb", "redis"],
        "ai_models": ["specialized_classifiers", "ollama_llm", "rag_embeddings"],
        "integrations": ["nhs_ehr", "doctor_network"]
    }

# For independent development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
'''
    
    # Write v2/app_v2.py
    with open(os.path.join(str(v2_path), "app_v2.py"), 'w') as f:
        f.write(v2_app_content)
    
    print("✅ Created unified app structure")

def create_development_scripts(backend_root: Path):
    """Create development scripts for easy management"""
    
    print("🔧 Creating development scripts...")
    
    # Create dev_v1.py - REMOVE EMOJIS FOR WINDOWS COMPATIBILITY
    dev_v1_content = '''#!/usr/bin/env python3
"""
Development script for running v1 independently
"""
import os
import sys
import uvicorn

# Add v1 to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'v1'))

if __name__ == "__main__":
    print("Starting Fairdoc v1 (Independent Development)")
    print("API Documentation: http://localhost:8001/docs")
    uvicorn.run("v1.app_v1:app", host="0.0.0.0", port=8001, reload=True)
'''
    
    # Create dev_v2.py - REMOVE EMOJIS FOR WINDOWS COMPATIBILITY
    dev_v2_content = '''#!/usr/bin/env python3
"""
Development script for running v2 independently
"""
import os
import sys
import uvicorn

# Add v2 to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'v2'))

if __name__ == "__main__":
    print("Starting Fairdoc v2 (Independent Development)")
    print("API Documentation: http://localhost:8002/docs")
    uvicorn.run("v2.app_v2:app", host="0.0.0.0", port=8002, reload=True)
'''
    
    # Create dev_unified.py - REMOVE EMOJIS FOR WINDOWS COMPATIBILITY
    dev_unified_content = '''#!/usr/bin/env python3
"""
Development script for running unified app (v1 + v2)
"""
import uvicorn

if __name__ == "__main__":
    print("Starting Fairdoc Unified API (v1 + v2)")
    print("Unified Documentation: http://localhost:8000/docs")
    print("v1 Documentation: http://localhost:8000/api/v1/docs")
    print("v2 Documentation: http://localhost:8000/api/v2/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
'''
    
    # Write development scripts with explicit UTF-8 encoding
    try:
        with open(os.path.join(str(backend_root), "dev_v1.py"), 'w', encoding='utf-8') as f:
            f.write(dev_v1_content)
        
        with open(os.path.join(str(backend_root), "dev_v2.py"), 'w', encoding='utf-8') as f:
            f.write(dev_v2_content)
        
        with open(os.path.join(str(backend_root), "dev_unified.py"), 'w', encoding='utf-8') as f:
            f.write(dev_unified_content)
        
        print("✅ Created development scripts")
        
    except UnicodeEncodeError as e:
        print(f"⚠️ Encoding error when creating scripts: {e}")
        print("   Creating ASCII-only versions...")
        
        # Fallback: Create simpler ASCII-only versions
        simple_v1 = 'import uvicorn\nif __name__ == "__main__":\n    uvicorn.run("v1.app_v1:app", host="0.0.0.0", port=8001, reload=True)\n'
        simple_v2 = 'import uvicorn\nif __name__ == "__main__":\n    uvicorn.run("v2.app_v2:app", host="0.0.0.0", port=8002, reload=True)\n'
        simple_unified = 'import uvicorn\nif __name__ == "__main__":\n    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)\n'
        
        with open(os.path.join(str(backend_root), "dev_v1.py"), 'w', encoding='utf-8') as f:
            f.write(simple_v1)
        with open(os.path.join(str(backend_root), "dev_v2.py"), 'w', encoding='utf-8') as f:
            f.write(simple_v2)
        with open(os.path.join(str(backend_root), "dev_unified.py"), 'w', encoding='utf-8') as f:
            f.write(simple_unified)
        
        print("✅ Created simplified development scripts")


def create_enhanced_documentation(backend_root: Path):
    """Create enhanced documentation for the unified structure"""
    
    print("📝 Creating enhanced documentation...")
    
    readme_content = f'''# Fairdoc AI Backend - Unified Architecture

This backend supports both legacy (v1) and modern (v2) API versions with shared configuration.

## 🏗️ Structure

```
backend/
├── app.py                 # Main unified app (mounts v1 + v2)
├── dev_v1.py             # Run v1 independently
├── dev_v2.py             # Run v2 independently
├── dev_unified.py        # Run unified app
├── .env                  # Shared environment config
├── docker-compose.yml    # Shared Docker services
├── requirements.txt      # Shared dependencies
├── v1/                   # Legacy backend structure
│   ├── app_v1.py        # Independent v1 app
│   ├── api/
│   ├── core/
│   ├── datamodels/
│   ├── services/
│   └── ...
└── v2/                   # Modern backend structure
    ├── app_v2.py        # Independent v2 app
    ├── api/
    ├── core/
    ├── datamodels/
    ├── services/
    ├── rag/             # RAG components
    └── ...
```

## 🚀 Development Usage

### Independent Development

```
# Run v1 independently (port 8001)
python dev_v1.py

# Run v2 independently (port 8002)
python dev_v2.py
```

### Unified API

```
# Run unified app with both versions (port 8000)
python dev_unified.py
# or
python app.py
```

## 📖 API Documentation

### Unified Access (Production)
- **Main Docs**: http://localhost:8000/docs
- **v1 API**: http://localhost:8000/api/v1/
- **v2 API**: http://localhost:8000/api/v2/
- **v1 Docs**: http://localhost:8000/api/v1/docs
- **v2 Docs**: http://localhost:8000/api/v2/docs

### Independent Access (Development)
- **v1 Docs**: http://localhost:8001/docs
- **v2 Docs**: http://localhost:8002/docs

## 🐳 Docker Usage

Shared Docker services (PostgreSQL, Redis, ChromaDB) are configured in the root:

```
# Start shared services
docker-compose up -d

# Services available to both v1 and v2:
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
# - ChromaDB: localhost:8001
```

## 🔧 Configuration

All configuration files are shared and located in the root:
- `.env` - Environment variables
- `.env.local` - Local development
- `.env.prod` - Production settings
- `docker-compose.yml` - Docker services

Both v1 and v2 use the same configuration, ensuring consistency.

## 📊 Version Differences

### v1 (Legacy)
- Original backend structure
- PostgreSQL only
- Basic ML models
- Simple triage logic

### v2 (Modern)
- PostgreSQL + ChromaDB separation
- RAG document processing
- Advanced AI orchestration
- NHS EHR integration
- Real-time bias monitoring
- Doctor network services

## 🔄 Migration Path

1. **Development**: Use independent scripts to develop each version
2. **Testing**: Use unified app to test version compatibility
3. **Production**: Deploy unified app with version routing

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
    
    # Write README.md
    with open(os.path.join(str(backend_root), "README.md"), 'w') as f:
        f.write(readme_content)
    
    print("✅ Enhanced documentation created")

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
        
        # Database managers
        "data/database/postgres_manager.py": "# PostgreSQL connection and session management",
        "data/database/chromadb_manager.py": "# ChromaDB vector database operations",
        "data/database/redis_manager.py": "# Redis cache and session management",
        
        # V2-specific config files
        "requirements-v2.txt": "# v2-specific Python dependencies",
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
            else:
                content = f'"""\n{file_path}\n{description}\n"""\n\n# TODO: Implement {filename} functionality\npass\n'
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"     🆕 Generated v2: {file_path}")
            generated_count += 1
        else:
            print(f"     ⏭️ Skipped v2 (exists): {file_path}")
    
    print(f"✅ Generated {generated_count} new v2 files")

def main():
    """Main migration function with unified structure creation"""
    print("🚀 Starting Fairdoc backend UNIFIED structure migration...")
    print("🎯 Creating unified backend with independent v1/v2 + shared configs")
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
        return
    
    print(f"✅ Ready to migrate from: {os.getcwd()}")
    
    try:
        # Scan existing files
        print("\n🔍 Scanning existing files...")
        existing_files = scan_existing_files_comprehensive()
        
        if not existing_files:
            print("❌ No files found to migrate!")
            return
        
        # Create unified directory structure
        print("\n📁 Creating unified directory structure...")
        backend_root, v1_path, v2_path = create_unified_backend_structure()
        
        # Copy existing files to v1 (preserve original structure)
        print("\n📋 Copying existing files to v1...")
        copy_existing_files_to_v1(existing_files, v1_path)
        
        # Copy selected files to v2 (new structure)
        print("\n📋 Copying selected files to v2...")
        copy_selected_files_to_v2(existing_files, v2_path)
        
        # Generate v2-specific files
        print("\n🔧 Generating v2-specific files...")
        generate_v2_specific_files(v2_path)
        
        # Create unified app files
        print("\n🔧 Creating unified app structure...")
        create_unified_app_files(backend_root, v1_path, v2_path)
        
        # Create development scripts
        print("\n🔧 Creating development scripts...")
        create_development_scripts(backend_root)
        
        # Create documentation
        print("\n📝 Creating enhanced documentation...")
        create_enhanced_documentation(backend_root)
        
        print("\n✅ UNIFIED MIGRATION COMPLETED!")
        print("📂 Unified backend structure created in current directory")
        print("📊 Total existing files found:", len(existing_files))
        print("📝 Structure created:")
        print("   📁 v1/ - Legacy backend (independent)")
        print("   📁 v2/ - Modern backend (independent)")
        print("   📄 app.py - Unified API (mounts v1 + v2)")
        print("   📄 dev_*.py - Development scripts")
        print("   📄 .env, docker-compose.yml - Shared configs")
        
        print("\n🚀 Usage Instructions:")
        print("\n**Independent Development:**")
        print("   python dev_v1.py     # v1 only (port 8001)")
        print("   python dev_v2.py     # v2 only (port 8002)")
        
        print("\n**Unified API:**")
        print("   python dev_unified.py  # Both versions (port 8000)")
        print("   python app.py          # Same as above")
        
        print("\n**API Access:**")
        print("   Unified: http://localhost:8000/docs")
        print("   v1 API:  http://localhost:8000/api/v1/")
        print("   v2 API:  http://localhost:8000/api/v2/")
        print("   v1 Docs: http://localhost:8000/api/v1/docs")
        print("   v2 Docs: http://localhost:8000/api/v2/docs")
        
        print("\n🎉 UNIFIED MIGRATION COMPLETED SUCCESSFULLY!")
        print("\n📋 Summary:")
        print("   ✅ v1/ - Legacy backend preserved and independent")
        print("   ✅ v2/ - Modern PostgreSQL/ChromaDB architecture")
        print("   ✅ Unified app.py - Mounts both versions")
        print("   ✅ Independent development scripts")
        print("   ✅ Shared configuration (.env, docker)")
        print("   ✅ Enhanced documentation")
        
    except Exception as e:
        print(f"\n❌ Unified migration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
