#!/usr/bin/env python3
"""
Development script for running v1 independently
"""
import os
import sys
import uvicorn

# Add v1 to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'v1'))

if __name__ == "__main__":
    print("Starting Fairdoc v1 (Independent Development)")
    print("API Documentation: http://localhost:8001/docs")
    uvicorn.run("v1.app_v1:app", host="0.0.0.0", port=8001, reload=True)
