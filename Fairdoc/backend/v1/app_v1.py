"""
Fairdoc v1 Backend Application
Original backend structure preserved
Can run independently or as part of unified app
"""
from fastapi import FastAPI

# Create v1 application
app = FastAPI(
    title="Fairdoc AI v1",
    description="Legacy API with original backend structure",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Import v1-specific routes
# from api.auth.routes import router as auth_router
# from api.medical.routes import router as medical_router

# Include routers
# app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# app.include_router(medical_router, prefix="/medical", tags=["Medical"])

@app.get("/")
async def v1_root():
    return {
        "message": "Fairdoc AI v1 - Legacy API",
        "version": "1.0.0",
        "structure": "original",
        "features": [
            "Basic medical triage",
            "PostgreSQL storage",
            "Original data models"
        ]
    }

@app.get("/health")
async def v1_health():
    return {
        "status": "healthy",
        "version": "v1",
        "database": "postgresql",
        "ai_models": "basic"
    }

# For independent development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
