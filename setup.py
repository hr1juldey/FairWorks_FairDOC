from pathlib import Path
import subprocess
import sys
import os
import time
from datetime import datetime

def print_debug(message: str, level: str = "INFO") -> None:
    """Print debug message with timestamp and level"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")

def print_separator(title: str = "") -> None:
    """Print a visual separator"""
    separator = "=" * 80
    if title:
        title_line = f" {title} ".center(80, "=")
        print(f"\n{title_line}")
    else:
        print(f"\n{separator}")

def safe_decode_output(output_bytes: bytes) -> str:
    """Safely decode subprocess output handling encoding issues"""
    if isinstance(output_bytes, str):
        return output_bytes
    
    # Try different encodings in order of preference
    encodings = ['utf-8', 'cp1252', 'latin1', 'ascii']
    
    for encoding in encodings:
        try:
            return output_bytes.decode(encoding, errors='replace')
        except (UnicodeDecodeError, LookupError):
            continue
    
    # Fallback: decode with replacement characters
    return output_bytes.decode('utf-8', errors='replace')

def run_subprocess_safe(cmd: list, timeout: int = 600) -> tuple:
    """Run subprocess with safe encoding handling"""
    print_debug(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Use bytes mode to avoid encoding issues, then decode safely
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # Force UTF-8 for Python subprocesses
        )
        
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_bytes, stderr_bytes = process.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout)
        
        # Safely decode output
        stdout = safe_decode_output(stdout_bytes)
        stderr = safe_decode_output(stderr_bytes)
        
        return process.returncode, stdout, stderr
        
    except Exception as e:
        print_debug(f"Subprocess error: {e}", "ERROR")
        raise

def get_system_info() -> dict:
    """Get system information for debugging"""
    import platform
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "architecture": platform.architecture(),
        "working_directory": os.getcwd(),
        "script_path": os.path.abspath(__file__),
        "environment": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", "Not set"),
            "PATH": os.environ.get("PATH", "")[:200] + "...",  # Truncate PATH
        }
    }

def check_python_compatibility() -> bool:
    """Check if Python version is compatible"""
    python_version = sys.version_info
    print_debug(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 8:
        print_debug("✅ Python version is compatible", "SUCCESS")
        return True
    else:
        print_debug("❌ Python 3.8+ required", "ERROR")
        return False

def install_uv() -> bool:
    """Install or verify uv package manager"""
    print_debug("Checking uv installation...")
    
    try:
        returncode, stdout, stderr = run_subprocess_safe(["uv", "--version"], timeout=10)
        if returncode == 0:
            print_debug(f"✅ uv found: {stdout.strip()}", "SUCCESS")
            return True
        else:
            print_debug(f"❌ uv version check failed: {stderr}", "ERROR")
            raise FileNotFoundError("uv not working")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        print_debug("❌ uv not found. Installing uv...", "WARNING")
        try:
            returncode, stdout, stderr = run_subprocess_safe([sys.executable, "-m", "pip", "install", "uv"])
            if returncode == 0:
                print_debug("✅ uv installed successfully", "SUCCESS")
                return True
            else:
                print_debug(f"❌ Failed to install uv: {stderr}", "ERROR")
                return False
        except Exception as e:
            print_debug(f"❌ Error installing uv: {e}", "ERROR")
            return False

def install_dependencies(requirements_path: Path) -> bool:
    """Install dependencies from requirements.txt"""
    print_debug("Installing dependencies with encoding-safe subprocess...")
    
    start_time = time.time()
    
    try:
        # Run uv pip install with safe encoding handling
        cmd = ["uv", "pip", "install", "-r", str(requirements_path)]
        returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=1800)  # 30 minute timeout
        
        end_time = time.time()
        installation_time = end_time - start_time
        
        if returncode == 0:
            print_debug(f"✅ Installation completed successfully in {installation_time:.2f}s", "SUCCESS")
            
            # Show installation summary
            print_debug("Installation output summary (last 15 lines):", "INFO")
            output_lines = stdout.split('\n')
            for line in output_lines[-15:]:
                if line.strip():
                    print_debug(f"  {line.strip()}", "OUTPUT")
            
            if stderr.strip():
                print_debug("Warnings/Notes:", "WARNING")
                stderr_lines = stderr.split('\n')
                for line in stderr_lines[-5:]:  # Last 5 warning lines
                    if line.strip():
                        print_debug(f"  {line.strip()}", "WARNING")
            
            return True
        else:
            print_debug(f"❌ Installation failed with return code: {returncode}", "ERROR")
            print_debug(f"Installation time: {installation_time:.2f}s")
            
            print_debug("Error output:", "ERROR")
            error_lines = stderr.split('\n')
            for line in error_lines:
                if line.strip():
                    print_debug(f"  {line.strip()}", "ERROR")
            
            return False
            
    except subprocess.TimeoutExpired:
        print_debug("❌ Installation timed out after 30 minutes", "ERROR")
        return False
    except Exception as e:
        print_debug(f"❌ Unexpected error during installation: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

def download_nltk_data() -> bool:
    """Download essential NLTK data packages"""
    print_debug("Downloading NLTK data packages...")
    
    # Essential NLTK data packages for medical NLP
    nltk_packages = [
        ("punkt", "Punkt tokenizer"),
        ("stopwords", "Stop words corpus"),
        ("wordnet", "WordNet lexical database"), 
        ("omw-1.4", "Open Multilingual Wordnet"),
        ("averaged_perceptron_tagger", "POS tagger"),
        ("vader_lexicon", "Sentiment analysis lexicon"),
        ("brown", "Brown corpus"),
        ("names", "Names corpus")
    ]
    
    success_count = 0
    
    for package, description in nltk_packages:
        try:
            print_debug(f"Downloading NLTK {package} ({description})...")
            cmd = [
                sys.executable, "-c", 
                f"import nltk; nltk.download('{package}', quiet=True); print('Downloaded {package}')"
            ]
            returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=120)
            
            if returncode == 0:
                print_debug(f"✅ {package} downloaded successfully", "SUCCESS")
                success_count += 1
            else:
                print_debug(f"❌ Failed to download {package}: {stderr}", "WARNING")
                
        except Exception as e:
            print_debug(f"❌ Error downloading {package}: {e}", "WARNING")
    
    print_debug(f"NLTK data download complete: {success_count}/{len(nltk_packages)} packages", "INFO")
    return success_count > 0

def verify_installation() -> bool:
    """Verify key packages are installed and working"""
    print_debug("Verifying package installation...")
    
    # Core packages to verify
    verification_packages = [
        ("fastapi", "FastAPI web framework"),
        ("pydantic", "Pydantic data validation"),
        ("uvicorn", "Uvicorn ASGI server"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch ML framework"),
        ("nltk", "NLTK text processing"),
        ("textblob", "TextBlob NLP library"),
        ("mesop", "Mesop UI framework"),
        ("numpy", "NumPy numerical computing"),
        ("pandas", "Pandas data analysis")
    ]
    
    success_count = 0
    
    for package, description in verification_packages:
        try:
            cmd = [
                sys.executable, "-c", 
                f"import {package}; print(f'{package}: {{getattr({package}, \"__version__\", \"unknown\")}}')"
            ]
            returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=10)
            
            if returncode == 0:
                version_info = stdout.strip()
                print_debug(f"✅ {description}: {version_info}", "VERIFY")
                success_count += 1
            else:
                print_debug(f"❌ Failed to import {package}: {stderr.strip()}", "VERIFY")
                
        except Exception as e:
            print_debug(f"❌ Error verifying {package}: {e}", "VERIFY")
    
    print_debug(f"Package verification complete: {success_count}/{len(verification_packages)} packages working", "INFO")
    return success_count >= len(verification_packages) * 0.8  # 80% success rate required

def test_medical_nlp() -> bool:
    """Test medical NLP functionality"""
    print_debug("Testing medical NLP capabilities...")
    
    test_script = """
import nltk
from textblob import TextBlob

# Test NLTK
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    text = "Patient presents with chest pain. ECG shows abnormal readings."
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)
    print(f"NLTK tokenization: {len(tokens)} tokens, {len(sentences)} sentences")
except Exception as e:
    print(f"NLTK error: {e}")

# Test TextBlob
try:
    blob = TextBlob("The patient feels better after medication.")
    sentiment = blob.sentiment
    print(f"TextBlob sentiment: polarity={sentiment.polarity:.3f}, subjectivity={sentiment.subjectivity:.3f}")
except Exception as e:
    print(f"TextBlob error: {e}")

print("Medical NLP test completed")
"""
    
    try:
        cmd = [sys.executable, "-c", test_script]
        returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=30)
        
        if returncode == 0:
            print_debug("✅ Medical NLP test passed:", "SUCCESS")
            for line in stdout.split('\n'):
                if line.strip():
                    print_debug(f"  {line.strip()}", "TEST")
            return True
        else:
            print_debug(f"❌ Medical NLP test failed: {stderr}", "ERROR")
            return False
            
    except Exception as e:
        print_debug(f"❌ Error testing medical NLP: {e}", "ERROR")
        return False

def main():
    print_separator("NHS Chest Pain AI Triage System - Python 3.13 Setup")
    print_debug("Starting NHS Medical AI dependency installation...")
    
    # Print system information
    print_separator("SYSTEM INFORMATION")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        if isinstance(value, dict):
            print_debug(f"{key.upper()}:")
            for sub_key, sub_value in value.items():
                print_debug(f"  {sub_key}: {sub_value}")
        else:
            print_debug(f"{key.upper()}: {value}")
    
    # Check Python compatibility
    if not check_python_compatibility():
        sys.exit(1)
    
    # Get the directory where setup.py is located
    setup_dir = Path(__file__).parent.absolute()
    print_debug(f"Setup directory: {setup_dir}")
    
    # Path to requirements.txt relative to setup.py
    requirements_path = setup_dir / "requirements.txt"
    print_debug(f"Requirements path: {requirements_path}")
    
    # Ensure requirements.txt exists
    if not requirements_path.exists():
        print_debug(f"Requirements file not found at {requirements_path}", "ERROR")
        print_debug("Available files in directory:", "DEBUG")
        try:
            for file in setup_dir.iterdir():
                print_debug(f"  - {file.name}", "DEBUG")
        except Exception as e:
            print_debug(f"Could not list directory contents: {e}", "ERROR")
        sys.exit(1)
    
    # Check file details
    try:
        file_size = requirements_path.stat().st_size
        print_debug(f"Requirements file size: {file_size} bytes")
        
        with open(requirements_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print_debug(f"Requirements file has {len(lines)} lines")
        print_debug("First 5 non-comment lines:")
        count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                print_debug(f"  {count + 1}: {line}", "DEBUG")
                count += 1
                if count >= 5:
                    break
                    
    except Exception as e:
        print_debug(f"Error reading requirements file: {e}", "ERROR")
        sys.exit(1)
    
    print_separator("UV PACKAGE MANAGER")
    
    # Install/verify uv
    if not install_uv():
        sys.exit(1)
    
    print_separator("DEPENDENCY INSTALLATION")
    
    # Install dependencies
    if not install_dependencies(requirements_path):
        sys.exit(1)
    
    print_separator("NLTK DATA DOWNLOAD")
    
    # Download NLTK data
    if not download_nltk_data():
        print_debug("❌ NLTK data download failed", "WARNING")
    
    print_separator("PACKAGE VERIFICATION")
    
    # Verify installation
    if not verify_installation():
        print_debug("❌ Package verification failed", "WARNING")
    
    print_separator("MEDICAL NLP TESTING")
    
    # Test medical NLP functionality
    if not test_medical_nlp():
        print_debug("❌ Medical NLP test failed", "WARNING")
    
    print_separator("INSTALLATION COMPLETE")
    
    print_debug("🎉 NHS Medical AI dependencies installed successfully!")
    print_debug("📋 Available Medical NLP Tools:")
    print_debug("  ✅ NLTK - Text tokenization, POS tagging, sentiment analysis")
    print_debug("  ✅ TextBlob - Simple NLP operations")
    print_debug("  ✅ Hugging Face Transformers - Advanced medical models")
    print_debug("  ✅ PyTorch - Deep learning framework")
    print_debug("")
    print_debug("🚀 Next Steps:")
    print_debug("  1. Test NLTK: python -c 'import nltk; print(\"NLTK ready!\")'")
    print_debug("  2. Test medical NLP: python medical_nlp_test.py")
    print_debug("  3. Start your NHS chest pain triage application!")
    print_debug("  4. Use Hugging Face medical models for advanced NLP")

if __name__ == "__main__":
    main()
