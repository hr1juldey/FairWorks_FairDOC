"""
Fairdoc v2 Backend Application
Modern architecture with PostgreSQL/ChromaDB separation
Can run independently or as part of unified app
"""
from fastapi import FastAPI

# Create v2 application
app = FastAPI(
    title="Fairdoc AI v2",
    description="Modern API with PostgreSQL/ChromaDB separated architecture",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Import v2-specific routes
# from api.auth.routes import router as auth_router
# from api.medical.routes import router as medical_router
# from api.rag.routes import router as rag_router
# from api.nhs.routes import router as nhs_router

# Include routers
# app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# app.include_router(medical_router, prefix="/medical", tags=["Medical"])
# app.include_router(rag_router, prefix="/rag", tags=["RAG Search"])
# app.include_router(nhs_router, prefix="/nhs", tags=["NHS Integration"])

@app.get("/")
async def v2_root():
    return {
        "message": "Fairdoc AI v2 - Modern Architecture",
        "version": "2.0.0",
        "structure": "postgresql_chromadb_separated",
        "features": [
            "Advanced AI triage",
            "PostgreSQL + ChromaDB",
            "RAG document processing",
            "NHS EHR integration",
            "Real-time bias monitoring",
            "Doctor network services"
        ]
    }

@app.get("/health")
async def v2_health():
    return {
        "status": "healthy",
        "version": "v2",
        "databases": ["postgresql", "chromadb", "redis"],
        "ai_models": ["specialized_classifiers", "ollama_llm", "rag_embeddings"],
        "integrations": ["nhs_ehr", "doctor_network"]
    }

# For independent development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
