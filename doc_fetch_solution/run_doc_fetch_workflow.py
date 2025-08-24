import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
import re
from urllib.parse import urlparse

# Configuration
VENV_PYTHON = "../fairdoc_ai_triage/.venv/bin/python"
SITEMAP_URLS_FILE = "fetch/urls.txt"
FETCH_PAGES_DIR = "fetch/pages"
FETCH_DOCS_DIR = "fetch/docs"
ERRORS_LOG_FILE = "fetch/errors.log"
PROGRESS_LOG_FILE = "fetch/progress.log"
SCRIPTS_DIR = "."  # Assuming scripts are in the same directory as run_doc_fetch_workflow

def log_message(message, level="INFO"):
    """Logs messages to progress.log and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    with open(PROGRESS_LOG_FILE, "a") as f:
        f.write(log_entry + "\n")
    print(log_entry)

def run_command(command, requires_approval=False, description="Executing command"):
    """Executes a shell command."""
    log_message(f"{description}: {' '.join(command)}")
    try:
        # Use subprocess.run for better control and error handling
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stdout:
            log_message(f"Command stdout:\n{result.stdout}", level="DEBUG")
        if result.stderr:
            log_message(f"Command stderr:\n{result.stderr}", level="WARNING")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"Command failed with exit code {e.returncode}:\n{e.stderr}", level="ERROR")
        return False
    except FileNotFoundError:
        log_message(f"Command not found: {command[0]}. Please ensure it's installed and in PATH.", level="ERROR")
        return False
    except Exception as e:
        log_message(f"An unexpected error occurred: {e}", level="ERROR")
        return False

def setup_directories():
    """Ensures fetch directories exist."""
    log_message("Checking and setting up directories...")
    if not os.path.exists("fetch"):
        os.makedirs("fetch")
        log_message("Created fetch directory.")
    if not os.path.exists(FETCH_PAGES_DIR):
        os.makedirs(FETCH_PAGES_DIR)
        log_message(f"Created {FETCH_PAGES_DIR} directory.")
    if not os.path.exists(FETCH_DOCS_DIR):
        os.makedirs(FETCH_DOCS_DIR)
        log_message(f"Created {FETCH_DOCS_DIR} directory.")
    
    # Ensure log files exist
    for f in [SITEMAP_URLS_FILE, ERRORS_LOG_FILE, PROGRESS_LOG_FILE]:
        if not os.path.exists(f):
            with open(f, 'w') as fp:
                pass  # Create empty file
            log_message(f"Created empty file: {fp}")
    log_message("Directory setup complete.")

def install_dependencies():
    """Installs required Python dependencies using uv pip."""
    log_message("Checking and installing Python dependencies...")
    # Check if beautifulsoup4 is installed
    try:
        subprocess.run([VENV_PYTHON, "-c", "import bs4"], check=True, capture_output=True)
        log_message("beautifulsoup4 is already installed.")
    except subprocess.CalledProcessError:
        log_message("beautifulsoup4 not found. Installing...")
        if not run_command(["uv", "pip", "install", "beautifulsoup4"], description="Installing beautifulsoup4"):
            log_message("Failed to install beautifulsoup4. Exiting.", level="ERROR")
            sys.exit(1)
    log_message("Python dependencies checked.")

def extract_urls_from_sitemap(sitemap_url):
    """Extracts URLs from a sitemap XML and writes to urls.txt."""
    log_message(f"Extracting URLs from sitemap: {sitemap_url}")
    command = ["curl", "-s", sitemap_url]
    try:
        curl_output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        urls = re.findall(r'<loc>(.*?)</loc>', curl_output)
        if not urls:
            log_message(f"No URLs found in sitemap: {sitemap_url}", level="WARNING")
            return False
        with open(SITEMAP_URLS_FILE, "w") as f:
            for url in urls:
                f.write(url + "\n")
        log_message(f"Extracted {len(urls)} URLs to {SITEMAP_URLS_FILE}.")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to fetch sitemap {sitemap_url}: {e.stderr}", level="ERROR")
        return False
    except Exception as e:
        log_message(f"An error occurred during sitemap parsing: {e}", level="ERROR")
        return False

def write_urls_from_links(site_links):
    """Writes provided site links to urls.txt."""
    log_message(f"Writing {len(site_links)} provided URLs to {SITEMAP_URLS_FILE}")
    with open(SITEMAP_URLS_FILE, "w") as f:
        for url in site_links:
            f.write(url + "\n")
    log_message(f"URLs written to {SITEMAP_URLS_FILE}.")

def main():
    parser = argparse.ArgumentParser(description="Automated documentation fetching workflow.")
    parser.add_argument("--sitemap_url", type=str, help="The XML sitemap URL to process (e.g., https://dspy.ai/sitemap.xml)")
    parser.add_argument("--site_links", nargs='*', help="A list of individual site links to process (space-separated)")
    
    args = parser.parse_args()

    if not args.sitemap_url and not args.site_links:
        log_message("Error: Either --sitemap_url or --site_links must be provided.", level="ERROR")
        parser.print_help()
        sys.exit(1)

    setup_directories()
    install_dependencies()

    if args.sitemap_url:
        if not extract_urls_from_sitemap(args.sitemap_url):
            log_message("Failed to extract URLs from sitemap. Exiting.", level="ERROR")
            sys.exit(1)
    elif args.site_links:
        write_urls_from_links(args.site_links)
    
    # Execute fetch_docs.py
    log_message("Executing fetch_docs.py...")
    if not run_command([VENV_PYTHON, os.path.join(SCRIPTS_DIR, "fetch_docs.py")], description="Running fetch_docs.py"):
        log_message("fetch_docs.py failed. Continuing to conversion with available files.", level="WARNING")
    time.sleep(5)  # Adhere to rate limit request (even for local script execution)

    # Execute convert_to_md.py
    log_message("Executing convert_to_md.py...")
    if not run_command([VENV_PYTHON, os.path.join(SCRIPTS_DIR, "convert_to_md.py")], description="Running convert_to_md.py"):
        log_message("convert_to_md.py failed. Continuing to cleanup.", level="WARNING")
    time.sleep(5)  # Adhere to rate limit request

    # Execute cleanup_report.py
    log_message("Executing cleanup_report.py...")
    # Pass the sitemap_url to cleanup_report.py if it was provided
    cleanup_cmd = [VENV_PYTHON, os.path.join(SCRIPTS_DIR, "cleanup_report.py")]
    if args.sitemap_url:
        # This would require cleanup_report.py to accept an argument, which it currently doesn't.
        # For now, it will use its default. If this was a real project, I'd modify cleanup_report.py
        # to accept the sitemap_url as an argument.
        pass 
    if not run_command(cleanup_cmd, description="Running cleanup_report.py"):
        log_message("cleanup_report.py failed.", level="WARNING")
    time.sleep(5)  # Adhere to rate limit request

    log_message("Documentation fetching workflow complete.")

if __name__ == "__main__":
    main()
