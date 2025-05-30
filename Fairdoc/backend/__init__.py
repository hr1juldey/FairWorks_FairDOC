"""
Fairdoc Backend Package
"""

import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Add the project root to Python path
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


"""
Fairdoc AI - Medical Triage System
Root package initialization
"""

__version__ = "1.0.0"
__author__ = "Fairdoc AI Team"
