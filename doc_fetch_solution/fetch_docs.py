import os
import re
import subprocess
import time
from datetime import datetime

# Configuration
SITEMAP_URLS_FILE = "./fetch/urls.txt"
FETCH_PAGES_DIR = "./fetch/pages"
ERRORS_LOG_FILE = "./fetch/errors.log"
PROGRESS_LOG_FILE = "./fetch/progress.log"
MAX_RETRIES = 3
TIMEOUT = 30 # seconds

def sanitize_filename(url):
    """
    Converts a URL to a safe filesystem filename.
    Replaces invalid characters and limits length.
    """
    filename = url.replace("https://", "").replace("http://", "")
    filename = re.sub(r"[/?&:]", "_", filename)
    filename = filename.strip("_") # Remove leading/trailing underscores
    if len(filename) > 200:
        filename = filename[:200]
    return filename + ".txt"

def log_progress(message, level="info"):
    """Logs messages to progress.log and optionally to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level.upper()}] {message}"
    with open(PROGRESS_LOG_FILE, "a") as f:
        f.write(log_message + "\n")
    print(log_message)

def log_error(url, message):
    """Logs errors to errors.log and progress.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"[{timestamp}] [ERROR] Failed to fetch {url}: {message}"
    with open(ERRORS_LOG_FILE, "a") as f:
        f.write(error_message + "\n")
    log_progress(f"Failed to fetch {url}: {message}", level="error")

def fetch_with_curl(url, filepath):
    """Fetches content using curl."""
    try:
        command = ["curl", "-L", "-s", "--max-time", str(TIMEOUT), "-o", filepath, url]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return True
        else:
            log_progress(f"Curl failed for {url}: {result.stderr or 'No output'}", level="warning")
            return False
    except Exception as e:
        log_progress(f"Curl exception for {url}: {e}", level="warning")
        return False

def fetch_with_wget(url, filepath):
    """Fetches content using wget."""
    try:
        command = ["wget", "-q", f"--timeout={TIMEOUT}", "-O", filepath, url]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return True
        else:
            log_progress(f"Wget failed for {url}: {result.stderr or 'No output'}", level="warning")
            return False
    except Exception as e:
        log_progress(f"Wget exception for {url}: {e}", level="warning")
        return False

# Placeholder for browser_action - this will need to be handled by the agent directly
# The Python script cannot directly call browser_action.
# For now, this will just return False.
def fetch_with_browser_tools(url, filepath):
    """Placeholder for fetching content using browser tools."""
    log_progress(f"Attempting to fetch {url} with browser tools (requires manual agent intervention).", level="warning")
    # In a real scenario, the agent would pause the script, use browser_action,
    # and then resume or manually save the content.
    # For this automated script, we'll simulate failure.
    return False

def main():
    if not os.path.exists(SITEMAP_URLS_FILE):
        log_error("", f"Sitemap URLs file not found: {SITEMAP_URLS_FILE}")
        return

    with open(SITEMAP_URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    total_urls = len(urls)
    success_count = 0
    error_count = 0

    log_progress(f"Starting content fetching for {total_urls} URLs.")

    for i, url in enumerate(urls):
        filename = sanitize_filename(url)
        filepath = os.path.join(FETCH_PAGES_DIR, filename)
        
        log_progress(f"Processing {i+1}/{total_urls}: {url}")

        fetched = False
        # Try curl
        if fetch_with_curl(url, filepath):
            fetched = True
            log_progress(f"Successfully fetched with curl: {url}")
        
        # If curl failed, try wget
        if not fetched:
            if fetch_with_wget(url, filepath):
                fetched = True
                log_progress(f"Successfully fetched with wget: {url}")
        
        # If wget failed, try browser tools (placeholder)
        if not fetched:
            # This part would require direct agent interaction for browser_action
            # For now, it will just fail.
            if fetch_with_browser_tools(url, filepath):
                fetched = True
                log_progress(f"Successfully fetched with browser tools: {url}")

        if fetched:
            success_count += 1
        else:
            error_count += 1
            log_error(url, "All fetching methods failed.")
        
        log_progress(f"Progress: {i+1}/{total_urls} ({success_count} success, {error_count} errors)")

    log_progress(f"Content fetching complete. Total URLs: {total_urls}, Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
