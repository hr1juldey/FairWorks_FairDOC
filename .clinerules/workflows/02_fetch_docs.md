# Documentation Fetching Workflow

## Description, Functions, Inputs, and Outputs

This workflow automates the process of extracting documentation from a website, either by processing its sitemap or by fetching content from a list of provided URLs. It converts the fetched content into organized Markdown files, handles errors gracefully, and provides a summary report.

**Functions:**
- `sanitize_filename(url)`: Converts a URL into a filesystem-safe filename.
- `extract_url(filename)`: Reconstructs the original URL from a sanitized filename.
- `extract_title(url)`: Generates a human-readable title from a URL.

**Inputs:**
- `sitemap_url` (string, optional): The URL of the sitemap XML file (e.g., `https://dspy.ai/sitemap.xml`).
- `site_links` (list of strings, optional): A list of individual URLs to fetch documentation from.
- `max_retries` (number, default: 3): Maximum retry attempts per URL (handled internally by scripts).
- `timeout` (number, default: 30): Request timeout in seconds (handled internally by scripts).

**Outputs:**
- `./fetch/urls.txt`: A file containing the list of URLs processed.
- `./fetch/pages/`: Directory containing raw fetched content (HTML/text files).
- `./fetch/docs/`: Directory containing converted Markdown documentation files.
- `./fetch/errors.log`: Log file detailing any errors encountered during fetching or conversion.
- `./fetch/progress.log`: Log file tracking the progress of the workflow.
- A summary report printed to the console and `progress.log` upon completion.

## YAML formatted instructions for LLMs

```yaml
workflow_id: 02_fetch_docs
name: Documentation Fetching Workflow
description: |
  Automates fetching and converting website documentation to Markdown.
  Supports sitemap URLs or a list of individual site links.
  Designed for robust execution by various LLMs, including non-tool-calling SLMs.
inputs:
  - name: sitemap_url
    type: string
    description: "The URL of the sitemap XML (e.g., https://dspy.ai/sitemap.xml)."
    optional: true
  - name: site_links
    type: array
    items:
      type: string
    description: "A list of individual URLs to fetch (e.g., ['https://example.com/page1', 'https://example.com/page2'])."
    optional: true
  - name: max_retries
    type: number
    default: 3
    description: "Maximum retry attempts per URL (internal to scripts)."
    optional: true
  - name: timeout
    type: number
    default: 30
    description: "Request timeout in seconds (internal to scripts)."
    optional: true
trigger:
  - type: user_request
    patterns:
      - "fetch documentation from {sitemap_url}"
      - "get docs from {site_links}"
      - "process sitemap {sitemap_url}"
      - "download documentation"
  - type: auto_trigger
    condition: "task_requires_documentation_fetching"
    context_keys: ["sitemap_url", "site_links"]
execution_steps:
  - step_id: run_workflow_script
    tool: execute_command
    command: |
      # Check if 'fetch' directory exists, if not, it will be created by the script.
      # The Python script handles argument parsing and sequential execution of sub-scripts.
      # Ensure 'uv' is available in the environment or virtual environment.
      # Example usage:
      # For sitemap: fairdoc_ai_triage/.venv/bin/python doc_fetch_solution/run_doc_fetch_workflow.py --sitemap_url "https://dspy.ai/sitemap.xml"
      # For site links: fairdoc_ai_triage/.venv/bin/python doc_fetch_solution/run_doc_fetch_workflow.py --site_links "https://dspy.ai/cheatsheet/" "https://dspy.ai/faqs/"
      # The script will automatically install beautifulsoup4 if not present.
      fairdoc_ai_triage/.venv/bin/python doc_fetch_solution/run_doc_fetch_workflow.py \
        {% if sitemap_url %}--sitemap_url "{{ sitemap_url }}"{% endif %} \
        {% if site_links %}--site_links {% for link in site_links %}"{{ link }}" {% endfor %}{% endif %}
    requires_approval: false
    output_parser: "text"
    on_success: "Workflow completed successfully. Check ./fetch/docs for markdown files."
    on_failure: "Workflow failed. Check ./fetch/errors.log and ./fetch/progress.log for details."
