"""
Main Fairdoc Backend Application
Serves both v1 and v2 APIs with unified access
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import version-specific apps
from v1.app_v1 import app as app_v1
from v2.app_v2 import app as app_v2

# Create main application
app = FastAPI(
    title="Fairdoc AI - Healthcare Triage Platform",
    description="Unified API serving both v1 (legacy) and v2 (modern) endpoints",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount version-specific applications
app.mount("/api/v1", app_v1)
app.mount("/api/v2", app_v2)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fairdoc AI Healthcare Triage Platform",
        "versions": {
            "v1": {
                "docs": "/api/v1/docs",
                "description": "Legacy API with original structure"
            },
            "v2": {
                "docs": "/api/v2/docs",
                "description": "Modern API with PostgreSQL/ChromaDB architecture"
            }
        },
        "unified_docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "versions": ["v1", "v2"],
        "services": ["api", "database", "ai_models"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
