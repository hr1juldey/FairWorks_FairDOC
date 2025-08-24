import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
import glob

# Configuration
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

def log_error(filepath, message):
    """Logs errors to errors.log and progress.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"[{timestamp}] [ERROR] Failed to convert {filepath}: {message}"
    with open(ERRORS_LOG_FILE, "a") as f:
        f.write(error_message + "\n")
    log_progress(f"Failed to convert {filepath}: {message}", level="error")

def sanitize_filename(url):
    """
    Converts a URL to a safe filesystem filename.
    This function is used to reverse the process in extract_url.
    """
    filename = url.replace("https://", "").replace("http://", "")
    filename = re.sub(r"[/?&:]", "_", filename)
    filename = filename.strip("_")
    if len(filename) > 200:
        filename = filename[:200]
    return filename + ".txt"

def extract_url(filename):
    """
    Extracts the original URL from a sanitized filename.
    """
    # Remove the .txt extension
    base_filename = os.path.basename(filename).replace(".txt", "")
    
    # Reverse the sanitization: _ back to / or :
    # This is a best effort and might not perfectly reconstruct complex URLs
    # We assume the original URL started with https://dspy.ai/
    
    # Replace the first underscore with :// for the domain part
    # Then replace subsequent underscores with /
    parts = base_filename.split('_')
    if len(parts) > 1:
        # Reconstruct the domain and path
        domain_part = parts[0]
        path_parts = parts[1:]
        
        # Attempt to reconstruct the original URL structure
        # This is a heuristic and might need refinement for more complex URL patterns
        reconstructed_url = f"https://{domain_part.replace('dspy.ai', 'dspy.ai')}"  # Ensure domain is correct
        
        for part in path_parts:
            if part:  # Avoid adding empty segments
                reconstructed_url += f"/{part}"
        
        # Clean up any double slashes that might occur from reconstruction
        reconstructed_url = reconstructed_url.replace('//', '/')
        reconstructed_url = reconstructed_url.replace('https:/', 'https://')  # Fix for https://dspy.ai/api/
        
        # Special handling for the root URL
        if reconstructed_url == "https://dspy.ai":
            return "https://dspy.ai/"
        
        # Add trailing slash if it was originally present (heuristic)
        if base_filename.endswith('_') and not reconstructed_url.endswith('/'):
            reconstructed_url += '/'
            
        return reconstructed_url
    
    return f"https://{base_filename.replace('_', '/')}"  # Fallback for simpler cases


def extract_title(url):
    """
    Extracts a meaningful title from a URL.
    Uses the last path segment, converts hyphens to spaces, and capitalizes.
    """
    path = url.rstrip('/').split('/')
    if path and path[-1]:
        title = path[-1].replace('-', ' ').replace('_', ' ')
        return title.replace("api", "API").title()
    return "Untitled Document"

def convert_html_to_markdown(html_content):
    """
    Converts HTML content to markdown, cleaning artifacts and preserving structure.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Convert to markdown
    # This is a basic conversion. For more robust conversion, a dedicated library
    # like `html2text` would be ideal, but we are limited to standard Python libs.
    # We'll try to extract main content and then convert.
    
    # Find the main content area. This is highly dependent on the website's structure.
    # Common patterns: <main>, <article>, <div> with specific IDs/classes.
    # For dspy.ai, it seems content is often within <div class="prose"> or similar.
    main_content = soup.find('div', class_='prose') or \
                   soup.find('article') or \
                   soup.find('main') or \
                   soup.body  # Fallback to body if specific content area not found

    if not main_content:
        main_content = soup  # Use the whole soup if no main content found

    # Convert headings
    for h_tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        h_tag.string = f"{'#' * int(h_tag.name[1:])} {h_tag.get_text()}\n\n"

    # Convert paragraphs
    for p_tag in main_content.find_all('p'):
        p_tag.string = f"{p_tag.get_text()}\n\n"

    # Convert links
    for a_tag in main_content.find_all('a'):
        if a_tag.get('href') and a_tag.get_text():
            a_tag.string = f"[{a_tag.get_text()}]({a_tag['href']})"

    # Convert lists (basic)
    for ul_tag in main_content.find_all('ul'):
        list_items = []
        for li_tag in ul_tag.find_all('li'):
            list_items.append(f"- {li_tag.get_text()}")
        ul_tag.string = "\n".join(list_items) + "\n\n"
    
    for ol_tag in main_content.find_all('ol'):
        list_items = []
        for i, li_tag in enumerate(ol_tag.find_all('li')):
            list_items.append(f"{i + 1}. {li_tag.get_text()}")
        ol_tag.string = "\n".join(list_items) + "\n\n"

    # Convert code blocks (basic)
    for pre_tag in main_content.find_all('pre'):
        code_tag = pre_tag.find('code')
        if code_tag:
            pre_tag.string = f"```\n{code_tag.get_text()}\n```\n\n"
        else:
            pre_tag.string = f"```\n{pre_tag.get_text()}\n```\n\n"

    # Get text and normalize whitespace
    markdown_content = main_content.get_text(separator="\n", strip=True)
    markdown_content = re.sub(r'\n\s*\n', '\n\n', markdown_content)  # Remove excessive blank lines
    
    return markdown_content.strip()

def main():
    if not os.path.exists(FETCH_PAGES_DIR):
        log_error("", f"Fetched pages directory not found: {FETCH_PAGES_DIR}")
        return

    os.makedirs(FETCH_DOCS_DIR, exist_ok=True)

    txt_files = glob.glob(os.path.join(FETCH_PAGES_DIR, "*.txt"))
    total_files = len(txt_files)
    converted_count = 0
    conversion_errors = 0

    log_progress(f"Starting conversion of {total_files} raw content files to Markdown.")

    for i, txt_file in enumerate(txt_files):
        log_progress(f"Converting {i + 1}/{total_files}: {os.path.basename(txt_file)}")
        
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            original_url = extract_url(txt_file)
            md_filename = os.path.basename(txt_file).replace(".txt", ".md")
            md_filepath = os.path.join(FETCH_DOCS_DIR, md_filename)

            # Clean and format content
            markdown_body = convert_html_to_markdown(content)

            # Add metadata header
            title = extract_title(original_url)
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Determine fetch status (heuristic, as we don't have direct status from fetch_docs.py)
            fetch_status = "Success" if os.path.getsize(txt_file) > 0 else "Empty Content"

            header = f"""# {title}

**Source:** {original_url}
**Fetched:** {current_timestamp}
**Status:** {fetch_status}

---

"""
            formatted_content = header + markdown_body

            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(formatted_content)
            
            if os.path.exists(md_filepath) and os.path.getsize(md_filepath) > 0:
                log_progress(f"Converted: {original_url} -> {md_filename}")
                converted_count += 1
            else:
                log_error(txt_file, "Converted markdown file is empty or not created.")
                conversion_errors += 1

        except Exception as e:
            log_error(txt_file, str(e))
            conversion_errors += 1
        
        log_progress(f"Conversion Progress: {i + 1}/{total_files} ({converted_count} success, {conversion_errors} errors)")

    log_progress(f"Markdown conversion complete. Total files: {total_files}, Converted: {converted_count}, Errors: {conversion_errors}")

if __name__ == "__main__":
    main()
