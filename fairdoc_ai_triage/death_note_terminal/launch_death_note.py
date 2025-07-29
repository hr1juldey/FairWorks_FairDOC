#!/usr/bin/env python3
"""
Death Note Terminal - One-Click Launcher

Launches the Death Note Terminal control panel with automatic:
- Dependency installation
- Backend server startup 
- Frontend deployment
- Browser auto-launch

Single responsibility: Application orchestration and deployment
File: ./launch_death_note.py
"""

import asyncio
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional, List

# Configuration
DEATH_NOTE_DIR = Path(__file__).parent / "death_note_terminal"
PROJECT_ROOT = Path(__file__).parent
CONTROL_PANEL_URL = "http://localhost:8999"
BACKEND_HEALTH_URL = "http://localhost:8000/api/v2/health"
MAIN_BACKEND = "http://localhost:8000/"
class DeathNoteLauncher:
    """One-click launcher for Death Note Terminal system"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.death_note_process: Optional[subprocess.Popen] = None
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        try:
            import fastapi
            import uvicorn
            import psutil
            print("✅ Core dependencies found")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Death Note Terminal dependencies"""
        requirements_file = DEATH_NOTE_DIR / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found in death_note_terminal/")
            return False
        
        print("📦 Installing Death Note Terminal dependencies...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ], capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def check_backend_health(self) -> bool:
        """Check if main backend is already running"""
        try:
            import requests
            response = requests.get(BACKEND_HEALTH_URL, timeout=2)
            if response.status_code == 200:
                print("✅ Main backend already running")
                return True
        except Exception:
            pass
        return False
    
    def start_main_backend(self) -> bool:
        """Start the main V2 backend if not running"""
        if self.check_backend_health():
            return True
            
        print("🚀 Starting main V2 backend...")
        try:
            backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "src.app2.main_v2:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=PROJECT_ROOT)
            
            self.processes.append(backend_process)
            
            # Wait for backend to start
            print("⏳ Waiting for backend to initialize...")
            for i in range(10):
                time.sleep(2)
                if self.check_backend_health():
                    print("✅ Main backend started successfully")
                    return True
                print(f"   Attempt {i + 1}/10...")
            
            print("⚠️ Backend may still be starting in background")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_death_note_terminal(self) -> bool:
        """Start the Death Note Terminal control panel"""
        main_py = DEATH_NOTE_DIR / "main.py"
        
        if not main_py.exists():
            print("❌ Death Note Terminal main.py not found")
            return False
        
        print("🎭 Starting Death Note Terminal...")
        try:
            self.death_note_process = subprocess.Popen([
                sys.executable, str(main_py)
            ], cwd=DEATH_NOTE_DIR)
            
            # Wait for terminal to start
            print("⏳ Initializing Death Note Terminal...")
            time.sleep(3)
            
            print("✅ Death Note Terminal started on port 8999")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Death Note Terminal: {e}")
            return False
    
    def open_browser(self) -> None:
        """Open the control panel in default browser"""
        print(f"🌐 Opening browser: {CONTROL_PANEL_URL}")
        try:
            webbrowser.open(CONTROL_PANEL_URL)
            print("✅ Browser launched successfully")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")
            print(f"   Please manually open: {CONTROL_PANEL_URL}")
    
    def display_info(self) -> None:
        """Display running services information"""
        
        print("\n" + "=" * 60)
        print("🎭 DEATH NOTE TERMINAL - ACTIVE SERVICES")
        print("=" * 60)
        print(f"📱 Control Panel:    {CONTROL_PANEL_URL}")
        print(f"🏥 Main Backend:     {MAIN_BACKEND}")
        print(f"📚 API Docs:         {MAIN_BACKEND}/docs")
        print(f"🔧 Terminal Docs:    {CONTROL_PANEL_URL}/docs")
        print("=" * 60)
        print("💡 Use the web interface to:")
        print("   • Start/stop V1/V2 servers")
        print("   • Run unit/integration/e2e tests")
        print("   • View terminal outputs")
        print("   • Get AI analysis from Ollama")
        print("=" * 60)
        print("⚠️  Press Ctrl+C to shutdown all services")
        print("=" * 60 + "\n")
    
    def cleanup(self) -> None:
        """Clean shutdown of all processes"""
        print("\n🧹 Shutting down Death Note Terminal...")
        
        # Stop Death Note Terminal
        if self.death_note_process and self.death_note_process.poll() is None:
            self.death_note_process.terminate()
            try:
                self.death_note_process.wait(timeout=5)
                print("✅ Death Note Terminal stopped")
            except subprocess.TimeoutExpired:
                self.death_note_process.kill()
                print("🔥 Force killed Death Note Terminal")
        
        # Stop other processes
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("👋 All services stopped")
    
    def run(self) -> None:
        """Main execution flow"""
        print("🎭 Death Note Terminal - One-Click Launcher")
        print("=" * 50)
        
        try:
            # Step 1: Check/install dependencies
            if not self.check_dependencies():
                if not self.install_dependencies():
                    sys.exit(1)
            
            # Step 2: Start main backend
            if not self.start_main_backend():
                print("⚠️ Continuing without main backend...")
            
            # Step 3: Start Death Note Terminal
            if not self.start_death_note_terminal():
                sys.exit(1)
            
            # Step 4: Open browser
            self.open_browser()
            
            # Step 5: Display info and wait
            self.display_info()
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
                    # Check if Death Note Terminal is still running
                    if (self.death_note_process and 
                        self.death_note_process.poll() is not None):
                        print("❌ Death Note Terminal stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested...")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()

def main():
    """Entry point for the launcher"""
    launcher = DeathNoteLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
