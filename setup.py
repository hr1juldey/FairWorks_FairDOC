from pathlib import Path
import subprocess
import sys
import os

os.path.abspath(__file__)

def main() -> None:
    # Get the directory where setup.py is located
    setup_dir = Path(__file__).parent.absolute()
    
    # Path to requirements.txt relative to setup.py
    requirements_path = setup_dir / "requirements.txt"
    
    # Ensure requirements.txt exists
    if not requirements_path.exists():
        print(f"Error: Requirements file not found at {requirements_path}")
        sys.exit(1)
    
    print(f"Installing dependencies from {requirements_path}...")
    
    try:
        # Run uv pip install with the requirements file
        result = subprocess.run(
            ["uv", "pip", "install", "-r", str(requirements_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print("Installation output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
        print("Installation complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'uv' command not found. Please install uv first.")
        print("You can install it with: pip install uv")
        sys.exit(1)


if __name__ == "__main__":
    main()
