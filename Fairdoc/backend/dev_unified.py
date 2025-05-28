#!/usr/bin/env python3
"""
Development script for running unified app (v1 + v2)
"""
import uvicorn

if __name__ == "__main__":
    print("Starting Fairdoc Unified API (v1 + v2)")
    print("Unified Documentation: http://localhost:8000/docs")
    print("v1 Documentation: http://localhost:8000/api/v1/docs")
    print("v2 Documentation: http://localhost:8000/api/v2/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
