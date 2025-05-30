"""
Fairdoc V1 API Package
Complete medical triage system with NHS compliance
"""

import sys
import os
from pathlib import Path

# Ensure proper path resolution for all imports
current_dir = Path(__file__).parent

# Add v1 directory to path
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Add backend directory to path
backend_dir = current_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Add project root to path
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__api_version__ = "v1"

# Import main components for easy access
try:
    from .app_v1 import app
    from .core.config import get_v1_settings
    
    __all__ = ["app", "get_v1_settings"]
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Could not import main components: {e}")
    __all__ = []



"""
Fairdoc AI - Medical Triage System
Root package initialization
"""

__version__ = "1.0.0"
__author__ = "Fairdoc AI Team"
