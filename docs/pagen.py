import os

# Define the name of the folder to create inside /docs
project_root = "fairdoc-github-pages-site"

# Define the project structure
structure = {
    "_config.yml": "title: My GitHub Pages Site\n",
    "index.md": "# Homepage\n",
    "vision-and-goals.md": "# Vision and Goals\n",
    "target-users.md": "# Target Users\n",
    "features": {
        "index.md": "# Core Features Overview\n",
        "v0-prototype-user-story.md": "# V0 Prototype User Story\n"
    },
    "architecture": {
        "index.md": "# Technical Architecture Overview\n",
        "frontend-layer.md": "# Frontend Layer\n",
        "api-gateway-layer.md": "# API Gateway Layer\n",
        "ai-orchestration-layer.md": "# AI Orchestration Layer\n",
        "specialized-ml-services.md": "# Specialized ML Services\n",
        "message-queue-cache.md": "# Message Queue & Cache\n",
        "data-layer.md": "# Data Layer\n",
        "pdf-report-generation.md": "# PDF Report Generation\n",
        "logging-monitoring.md": "# Logging & Monitoring\n"
    },
    "infrastructure-deployment.md": "# Infrastructure & Deployment\n",
    "developer-guide": {
        "index.md": "# Developer Guide Overview\n",
        "setup-prerequisites.md": "# Setup Prerequisites\n",
        "core-api-endpoints.md": "# Core API Endpoints\n",
        "sdks-libraries-connectors.md": "# SDKs, Libraries & Connectors\n"
    },
    "roadmap-timeline": {
        "index.md": "# Roadmap Overview\n",
        "v0-prototype-sprint-todo.md": "# V0 Prototype Sprint TODO\n"
    },
    "financial-sustainability.md": "# Financial Sustainability\n",
    "legal-compliance.md": "# Legal Compliance\n",
    "contributing.md": "# Contributing\n",
    "acknowledgements.md": "# Acknowledgements\n",
    "references.md": "# References\n"
}

def create_structure(base_path, tree):
    for name, content in tree.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    # Get path to the /docs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, project_root)

    os.makedirs(base_dir, exist_ok=True)
    create_structure(base_dir, structure)
    print(f"📁 GitHub Pages project structure created at: {base_dir}")
