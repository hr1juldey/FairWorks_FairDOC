import os
import glob
from datetime import datetime

# Configuration
SITEMAP_URLS_FILE = "./fetch/urls.txt"
FETCH_PAGES_DIR = "./fetch/pages"
FETCH_DOCS_DIR = "./fetch/docs"
ERRORS_LOG_FILE = "./fetch/errors.log"
PROGRESS_LOG_FILE = "./fetch/progress.log"

def log_progress(message, level="info"):
    """Logs messages to progress.log and optionally to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level.upper()}] {message}"
    with open(PROGRESS_LOG_FILE, "a") as f:
        f.write(log_message + "\n")
    print(log_message)

def remove_empty_files(directory):
    """Removes empty files from a given directory."""
    removed_count = 0
    for filepath in glob.glob(os.path.join(directory, "*")):
        if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
            os.remove(filepath)
            log_progress(f"Removed empty file: {filepath}", level="debug")
            removed_count += 1
    return removed_count

def generate_summary_report(sitemap_url="https://dspy.ai/sitemap.xml"):
    """Generates a summary report of the documentation fetching workflow."""
    total_urls = 0
    if os.path.exists(SITEMAP_URLS_FILE):
        with open(SITEMAP_URLS_FILE, "r") as f:
            total_urls = len([line.strip() for line in f if line.strip()])

    success_fetched_count = len(glob.glob(os.path.join(FETCH_PAGES_DIR, "*.txt")))
    converted_count = len(glob.glob(os.path.join(FETCH_DOCS_DIR, "*.md")))
    
    error_count = 0
    if os.path.exists(ERRORS_LOG_FILE):
        with open(ERRORS_LOG_FILE, "r") as f:
            error_count = len([line.strip() for line in f if line.strip()])

    # Heuristic for conversion errors, as the convert_to_md.py script logs them
    # but doesn't provide a direct count to this script.
    # We'll assume any discrepancy between fetched and converted is a conversion error.
    conversion_errors = success_fetched_count - converted_count
    if conversion_errors < 0:  # Should not happen if logic is correct
        conversion_errors = 0

    total_errors = error_count + conversion_errors

    success_rate = (success_fetched_count / total_urls * 100) if total_urls > 0 else 0
    conversion_rate = (converted_count / success_fetched_count * 100) if success_fetched_count > 0 else 0

    report = f"""## Documentation Fetching Complete

**Sitemap:** {sitemap_url}
**Total URLs:** {total_urls}
**Successfully Fetched:** {success_fetched_count}
**Conversion Success:** {converted_count}
**Errors:** {total_errors}

**Success Rate:** {success_rate:.2f}%
**Conversion Rate:** {conversion_rate:.2f}%

### Files Generated:
- Raw content: {FETCH_PAGES_DIR}/ ({success_fetched_count} files)
- Markdown docs: {FETCH_DOCS_DIR}/ ({converted_count} files)
- Error log: {ERRORS_LOG_FILE}
- URL list: {SITEMAP_URLS_FILE}
"""
    log_progress("Generated Summary Report:\n" + report)
    return report

def main():
    log_progress("Starting cleanup and reporting.")

    # Remove empty files
    removed_pages = remove_empty_files(FETCH_PAGES_DIR)
    removed_docs = remove_empty_files(FETCH_DOCS_DIR)
    log_progress(f"Removed {removed_pages} empty files from {FETCH_PAGES_DIR}.")
    log_progress(f"Removed {removed_docs} empty files from {FETCH_DOCS_DIR}.")

    # Generate and print summary report
    generate_summary_report()

    log_progress("Cleanup and reporting complete.")

if __name__ == "__main__":
    main()
