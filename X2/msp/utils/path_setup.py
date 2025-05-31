import sys
import os
from pathlib import Path

_APP_ROOT_SETUP_DONE = False

def setup_project_paths():
    global _APP_ROOT_SETUP_DONE
    if _APP_ROOT_SETUP_DONE:
        return

    # Assuming this file is in msp/utils/
    # Project root is two levels up from here (msp/)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    paths_to_add = [
        str(project_root),  # Adds 'msp' directory to sys.path
    ]
    
    for path_str in paths_to_add:
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            print(f"Added to sys.path: {path_str}")

    _APP_ROOT_SETUP_DONE = True

# Call setup when this module is imported for the first time
setup_project_paths()
