#!/bin/bash
# mcp_startup.sh - Complete MCP server startup with cleanup

set -e

echo "🎯 MCP Server Complete Startup"
echo "================================"

# Configuration
VENV_PATH="/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv"
MEMORY_FILE="/home/riju279/Documents/Cline/MCP/memory-server/memory.json"
MEMORY_FILE_SMALL="/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/tmp/mcp_memory_small.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory
mkdir -p mcp_logs_v2

echo "📋 Step 1: Checking dependencies..."

# Check if required tools are available
check_dependency() {
    if command -v "$1" &> /dev/null; then
        echo "  ✅ $1 available"
        return 0
    else
        echo "  ❌ $1 not found"
        return 1
    fi
}

MISSING_DEPS=0

check_dependency "python3" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_dependency "node" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_dependency "npx" || MISSING_DEPS=$((MISSING_DEPS + 1))

if [ $MISSING_DEPS -gt 0 ]; then
    echo "❌ Missing $MISSING_DEPS dependencies. Please install them first."
    exit 1
fi

# Check Python packages
echo "🐍 Checking Python packages..."
python3 -c "import psutil" 2>/dev/null || {
    echo "  📦 Installing psutil..."
    pip3 install psutil
}

echo "🧹 Step 2: Cleaning memory server data..."

# Clean up memory to prevent context saturation
if [ -f "$MEMORY_FILE" ]; then
    echo "  🔍 Found existing memory file: $MEMORY_FILE"
    echo "  📊 Current size: $(wc -l < "$MEMORY_FILE") entries"
    
    # Run cleanup
    python3 - <<EOF
import json
import os
from pathlib import Path

# Create optimized memory
optimized_memories = [
    {
        "type": "entity",
        "name": "FairDoc_AI_System", 
        "entityType": "project",
        "observations": [
            "Medical AI triage system",
            "FastAPI + Python backend",
            "DSPy optimization framework", 
            "Located: /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC"
        ]
    },
    {
        "type": "entity",
        "name": "Riju279",
        "entityType": "developer", 
        "observations": [
            "Primary developer",
            "Asia/Kolkata timezone",
            "Python/FastAPI expertise"
        ]
    },
    {
        "type": "relation",
        "from": "Riju279",
        "to": "FairDoc_AI_System", 
        "relationType": "develops"
    }
]

# Save small memory file
with open("$MEMORY_FILE_SMALL", "w") as f:
    for memory in optimized_memories:
        f.write(json.dumps(memory) + "\\n")

print(f"  ✅ Created optimized memory: $MEMORY_FILE_SMALL")
print(f"  📊 Size: {len(optimized_memories)} entries")
EOF

else
    echo "  ⚠️ No existing memory file found"
    # Create small initial memory
    python3 - <<EOF
import json

memories = [
    {
        "type": "entity",
        "name": "FairDoc_System",
        "entityType": "project",
        "observations": ["Medical AI triage system", "Python FastAPI"]
    }
]

with open("$MEMORY_FILE_SMALL", "w") as f:
    for memory in memories:
        f.write(json.dumps(memory) + "\\n")
        
print("  ✅ Created initial memory file")
EOF
fi

echo "🚀 Step 3: Starting MCP servers..."

# Kill any existing MCP processes
echo "  🛑 Cleaning up any existing MCP processes..."
pkill -f "mcp-server" || true
pkill -f "@modelcontextprotocol" || true
sleep 2

# Start the robust MCP runner
echo "  🎯 Starting robust MCP runner..."

# Create the runner script if it doesn't exist
if [ ! -f "$SCRIPT_DIR/robust_mcp_runner.py" ]; then
    echo "  ❌ robust_mcp_runner.py not found in $SCRIPT_DIR"
    echo "  💡 Please save the improved MCP runner script as 'robust_mcp_runner.py'"
    exit 1
fi

# Set environment for better resource management
export NODE_OPTIONS="--max-old-space-size=128"
export PYTHONUNBUFFERED=1

# Start with virtual environment if available
if [ -d "$VENV_PATH" ]; then
    echo "  🔄 Using virtual environment: $VENV_PATH"
    PYTHON_CMD="$VENV_PATH/bin/python3"
else
    echo "  ⚠️ No virtual environment found, using system Python"
    PYTHON_CMD="python3"
fi

# Run the MCP manager
echo "  ▶️ Starting MCP servers..."
echo "  📁 Logs will be in: mcp_logs_v2/"
echo "  ⌨️ Press Ctrl+C to stop all servers"
echo ""

# Execute the runner
exec "$PYTHON_CMD" "$SCRIPT_DIR/robust_mcp_runner.py" --venv "$VENV_PATH"