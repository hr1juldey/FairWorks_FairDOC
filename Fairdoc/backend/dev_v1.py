#!/usr/bin/env python3
"""
V1 API Development Runner
Ensures proper import paths and starts the development server
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add backend to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set up environment
os.environ.setdefault('ENVIRONMENT', 'development')

def main():
    """Main entry point for V1 API"""
    
    try:
        # Import and run the V1 app
        from v1.app_v1 import app
        import uvicorn
        
        print("🚀 Starting Fairdoc AI V1 API...")
        print(f"📁 Project root: {project_root}")
        print(f"📁 Backend dir: {backend_dir}")
        print(f"🐍 Python path: {sys.path[:3]}...")
        
        # Run with uvicorn
        uvicorn.run(
            "v1.app_v1:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Checking import paths...")
        
        # Debug import paths
        print("\n📋 Current sys.path:")
        for i, path in enumerate(sys.path[:10]):
            print(f"  {i}: {path}")
        
        # Try to import v1 module step by step
        try:
            import v1
            print("✅ v1 module imported successfully")
        except ImportError as e2:
            print(f"❌ Cannot import v1 module: {e2}")
        
        return 1
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
