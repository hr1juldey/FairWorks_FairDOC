# System Patterns

This document outlines the system architecture, key technical decisions, design patterns in use, component relationships, and critical implementation paths for the Fairdoc AI Triage System.

## System Architecture:
- The system appears to be a Python-based application, likely using a web framework (FastAPI is suggested by some file names like `src/app/main.py` and `src/app2/main_v2.py`).
- It integrates with an LLM (Large Language Model) via Ollama, as indicated by `ollama_client.py` and log entries.
- There's a `memory-bank` directory for documentation and context.
- Database interactions are present, suggested by `database.py` and `init.sql`.
- There are two main application versions, `app` and `app2`, with `app2` seemingly being a newer iteration (`main_v2.py`, `router_v2.py`, `config_v2.py`, etc.).

## Key Technical Decisions:
- **LLM Integration:** Ollama is used for local LLM inference.
- **DSPy:** The project heavily utilizes DSPy for optimizing LLM prompts and modules, as seen in `dspy_config_v2.py` and various test files.
- **Database:** PostgreSQL is likely used, given `docker/postgres/init.sql`.
- **Asynchronous Operations:** The `asyncio.run()` errors in `log.txt` suggest extensive use of asynchronous programming.

## Design Patterns in Use:
- **Service Layer:** `src/app2/services/` contains various services (chat, context, database, dspy), indicating a service-oriented architecture.
- **API Versioning:** The presence of `v1` and `v2` in the API paths (`src/app/api/v1`, `src/app2/api/v2`) suggests API versioning.
- **Configuration Management:** `config.py` and `config_v2.py` handle application configuration.
- **Schema Definition:** `models/schemas/` defines data schemas, likely for API request/response validation and database models.

## Component Relationships:
- **Main Application (`app2/main_v2.py`)** orchestrates the overall flow.
- **API Routers (`router_v2.py`)** expose endpoints for interaction.
- **Services (`services/`)** encapsulate business logic and interactions with external components (LLMs, database).
- **Models (`models/`)** define data structures for the database and API.
- **Core (`core/`)** handles fundamental aspects like configuration, database connection, and dependencies.
- **Tests (`tests/`)** validate the functionality and performance of various components.

## Critical Implementation Paths:
- **Conversation Flow:** From user input to AI processing, triage, and response generation. This involves `chat_orchestrator.py`, `medical_agent.py`, `question_generator.py`, and `nice_lookup.py`.
- **DSPy Optimization:** The process of evaluating and optimizing DSPy modules using gold standards, handled by `evaluation_optimizer.py`.
- **Database Persistence:** Saving and retrieving conversation states and gold standards.
- **Emergency Handling:** The `emergency_handler.py` is critical for identifying and escalating urgent medical situations.
