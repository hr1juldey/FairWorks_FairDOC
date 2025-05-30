"""
Universal Import Configuration for V1 API
Ensures all modules can find each other
"""

import sys
import os
from pathlib import Path

def setup_import_paths():
    """Setup import paths for the entire V1 API"""
    
    # Get the directory containing this file
    config_dir = Path(__file__).parent
    
    # Define all relevant paths
    paths = [
        config_dir,                      # v1/
        config_dir.parent,               # backend/
        config_dir.parent.parent,        # Fairdoc/
        config_dir / "api",              # v1/api/
        config_dir / "core",             # v1/core/
        config_dir / "data",             # v1/data/
        config_dir / "datamodels",       # v1/datamodels/
        config_dir / "utils",            # v1/utils/
        config_dir / "services",         # v1/services/
        config_dir / "MLmodels",         # v1/MLmodels/
        config_dir / "tools",            # v1/tools/
    ]
    
    # Add paths to sys.path
    for path in paths:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [str(p.resolve()) for p in paths if str(p.resolve()) not in current_pythonpath]
    
    if new_paths:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = os.pathsep.join([current_pythonpath] + new_paths)
        else:
            os.environ['PYTHONPATH'] = os.pathsep.join(new_paths)


# Call setup when module is imported
setup_import_paths()

# Test import function
def test_imports():
    """Test that all critical imports work"""
    
    test_modules = [
        ("core.config", "get_v1_settings"),
        ("core.security", "get_current_active_user"),
        ("data.database", "get_db"),
        ("datamodels.auth_models", "UserDB"),
        ("datamodels.file_models", "FileCategory"),
        ("utils.file_utils", "validate_medical_file"),
    ]
    
    results = {}
    
    for module_name, item_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[item_name])
            item = getattr(module, item_name)
            print(item)
            results[f"{module_name}.{item_name}"] = "✅ SUCCESS"
        except Exception as e:
            results[f"{module_name}.{item_name}"] = f"❌ FAILED: {e}"
    
    return results


if __name__ == "__main__":
    # Test imports when run directly
    print("Testing V1 API imports...")
    results = test_imports()
    for test, result in results.items():
        print(f"{test}: {result}")
