#!/usr/bin/env python3
"""
Development script for running v2 independently
"""
import os
import sys
import uvicorn

# Add v2 to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'v2'))

if __name__ == "__main__":
    print("Starting Fairdoc v2 (Independent Development)")
    print("API Documentation: http://localhost:8002/docs")
    uvicorn.run("v2.app_v2:app", host="0.0.0.0", port=8002, reload=True)
