# tests/training_poc/__init__.py
"""
Training POC Package Initialization

This __init__.py file sets up the Python path and import resolution
to make the POC modules work with pytest without modifying the source files.

It handles:
1. Adding src directory to Python path for app2 imports
2. Setting up sys.path for relative imports within the POC
3. Making the POC work as both a standalone package and pytest module
"""

import sys
import os
from pathlib import Path

# Get the project root directory (fairdoc_ai_triage/src/)
current_file = Path(__file__).resolve()
tests_training_poc_dir = current_file.parent  # tests/training_poc/
tests_dir = tests_training_poc_dir.parent     # tests/
src_dir = tests_dir.parent                    # src/
project_root = src_dir                        # This is our src/ directory

# Add src directory to Python path for app2 imports
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add the training_poc directory itself for relative imports
if str(tests_training_poc_dir) not in sys.path:
    sys.path.insert(0, str(tests_training_poc_dir))

# Ensure the tests directory is also in path (for conftest.py and other test utilities)
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Set environment variables that might be needed
os.environ.setdefault('PYTHONPATH', str(src_dir))

# Optional: Print debug info (uncomment for debugging)
# print(f"[POC Init] Added to sys.path:")
# print(f"  src_dir: {src_dir}")
# print(f"  tests_dir: {tests_dir}")
# print(f"  training_poc_dir: {tests_training_poc_dir}")

try:
    # Test that we can import app2 modules (this validates our path setup)
    from app2.core.dspy_config_v2 import ensure_dspy_configured
    # print("[POC Init] ✅ Successfully can import app2 modules")
except ImportError as e:
    print(f"[POC Init] ⚠️  Warning: Could not import app2 modules: {e}")
    print(f"[POC Init] Current sys.path: {sys.path[:5]}...")  # Show first 5 entries

# Make the package importable
__all__ = [
    'external_trainer',
    'patient_generator', 
    'optimizer_comparison',
    'evaluation_suite',
    'time_cost_analysis',
    'run_poc'
]

# Optional: Pre-import modules to catch any issues early
try:
    # These imports will work now because we've set up the path correctly
    from . import patient_generator
    from . import external_trainer
    from . import optimizer_comparison
    from . import evaluation_suite
    from . import time_cost_analysis
    from . import run_poc
    
    # print("[POC Init] ✅ Successfully imported all POC modules")
    
except ImportError as e:
    print(f"[POC Init] ⚠️  Warning: Could not pre-import POC modules: {e}")

# Export key classes for easier importing
try:
    from .patient_generator import PersonaGenerator, PatientScenario
    from .external_trainer import ExternalTrainer
    from .optimizer_comparison import OptimizerBenchmark
    from .evaluation_suite import MedicalEvaluationSuite
    from .time_cost_analysis import TrainingCostAnalyzer
    
    __all__.extend([
        'PersonaGenerator',
        'PatientScenario', 
        'ExternalTrainer',
        'OptimizerBenchmark',
        'MedicalEvaluationSuite',
        'TrainingCostAnalyzer'
    ])
    
except ImportError as e:
    # If imports fail, don't crash the whole package
    print(f"[POC Init] ⚠️  Could not export main classes: {e}")
    pass
