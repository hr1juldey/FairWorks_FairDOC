"""
Smart Import Management for V1 API
Handles both relative and absolute imports automatically
"""

import sys
import os
import importlib
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class ImportManager:
    """Manages imports across the V1 API package"""
    
    def __init__(self):
        self._setup_paths()
        self._imported_modules = {}
    
    def _setup_paths(self):
        """Setup all necessary paths for imports"""
        
        # Get current directory (core)
        current_dir = Path(__file__).parent
        
        # Get v1 directory
        v1_dir = current_dir.parent
        
        # Get backend directory
        backend_dir = v1_dir.parent
        
        # Get project root
        project_root = backend_dir.parent
        
        # Paths to add
        paths_to_add = [
            str(project_root),           # Fairdoc/
            str(backend_dir),            # Fairdoc/backend/
            str(v1_dir),                 # Fairdoc/backend/v1/
            str(current_dir),            # Fairdoc/backend/v1/core/
            str(v1_dir / "api"),         # Fairdoc/backend/v1/api/
            str(v1_dir / "data"),        # Fairdoc/backend/v1/data/
            str(v1_dir / "datamodels"),  # Fairdoc/backend/v1/datamodels/
            str(v1_dir / "utils"),       # Fairdoc/backend/v1/utils/
            str(v1_dir / "services"),    # Fairdoc/backend/v1/services/
        ]
        
        # Add paths to sys.path if not already present
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
                logger.debug(f"Added to sys.path: {path}")
    
    def safe_import(self, module_name: str, package: Optional[str] = None) -> Optional[Any]:
        """Safely import a module with fallback options"""
        
        # Cache check
        cache_key = f"{package}.{module_name}" if package else module_name
        if cache_key in self._imported_modules:
            return self._imported_modules[cache_key]
        
        # Try different import strategies
        import_strategies = [
            # Absolute import
            lambda: importlib.import_module(module_name, package),
            
            # Try with v1 prefix
            lambda: importlib.import_module(f"v1.{module_name}"),
            
            # Try with backend.v1 prefix
            lambda: importlib.import_module(f"backend.v1.{module_name}"),
            
            # Try with Fairdoc.backend.v1 prefix
            lambda: importlib.import_module(f"Fairdoc.backend.v1.{module_name}"),
            
            # Try relative import if package is provided
            lambda: importlib.import_module(f".{module_name}", package) if package else None,
        ]
        
        for strategy in import_strategies:
            try:
                if strategy:
                    module = strategy()
                    if module:
                        self._imported_modules[cache_key] = module
                        logger.debug(f"Successfully imported: {cache_key}")
                        return module
            except (ImportError, ValueError, TypeError) as e:
                logger.debug(f"Import strategy failed for {cache_key}: {e}")
                continue
        
        logger.error(f"All import strategies failed for: {cache_key}")
        return None
    
    def import_from_module(self, module_name: str, item_name: str, package: Optional[str] = None) -> Optional[Any]:
        """Import a specific item from a module"""
        
        module = self.safe_import(module_name, package)
        if module and hasattr(module, item_name):
            return getattr(module, item_name)
        
        logger.error(f"Could not import {item_name} from {module_name}")
        return None


# Global import manager instance
import_manager = ImportManager()

# Convenience functions
def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """Safely import a module"""
    return import_manager.safe_import(module_name, package)

def import_from(module_name: str, item_name: str, package: Optional[str] = None) -> Optional[Any]:
    """Import a specific item from a module"""
    return import_manager.import_from_module(module_name, item_name, package)

# Common imports for easy access
def get_database():
    """Get database session"""
    return import_from("data.database", "get_db")

def get_current_user():
    """Get current user dependency"""
    return import_from("core.security", "get_current_active_user")

def get_config():
    """Get configuration"""
    return import_from("core.config", "get_v1_settings")
