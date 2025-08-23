# Tech Context

This document details the technologies used, development setup, technical constraints, dependencies, and tool usage patterns for the Fairdoc AI Triage System.

## Technologies Used:
- **Python:** Primary programming language.
- **FastAPI:** Likely web framework for building APIs.
- **Ollama:** Used for running Large Language Models locally.
- **DSPy:** Framework for programming with Large Language Models, used for optimizing prompts and modules.
- **PostgreSQL:** Relational database for data persistence.
- **SQLAlchemy:** ORM (Object-Relational Mapper) for interacting with the database.
- **asyncpg:** Asynchronous PostgreSQL driver.
- **Redis:** Potentially used for caching or message queuing, as suggested by `redis_queue.py`.
- **Pytest:** Testing framework.
- **uv:** Python package manager (indicated by `uv.lock`).
- **Sentence Transformers:** Used for embedding, as seen in `test_nice_protocol_optimizer.py` logs.

## Development Setup:
- **Virtual Environment:** Python virtual environments are used (e.g., `.venv/`).
- **Docker:** Docker is used for containerization, especially for the database (`docker/postgres/init.sql`, `docker-compose.yml`).
- **Configuration:** Environment variables and configuration files (`.env.example`, `config.py`, `config_v2.py`) manage settings.

## Technical Constraints:
- **Local LLM Inference:** Reliance on Ollama implies local execution of LLMs, which might have performance implications depending on hardware.
- **Asynchronous Programming:** The `asyncio.run()` errors in the logs indicate challenges with managing asynchronous contexts, which needs careful handling to avoid deadlocks or unexpected behavior.
- **DSPy Optimization:** The optimization process itself can be resource-intensive and time-consuming, as seen in the test logs.
- **Database Connection Management:** Proper handling of asynchronous database connections is crucial to prevent resource leaks or errors.

## Dependencies:
- Python packages managed by `uv` (from `pyproject.toml` and `uv.lock`).
- Ollama for LLM models.
- PostgreSQL database.

## Tool Usage Patterns:
- **`uv run pytest`:** Used for running tests, often with verbose logging and specific log levels.
- **`httpx`:** Used for making HTTP requests, likely for interacting with Ollama or other internal/external APIs.
- **`sqlalchemy.engine.Engine`:** Logs indicate database operations.
- **`dspy.settings.configure` / `dspy.context`:** Used for configuring DSPy, with noted issues when called from different async tasks.
- **`sentence_transformers.SentenceTransformer`:** Used for loading and using embedding models.
