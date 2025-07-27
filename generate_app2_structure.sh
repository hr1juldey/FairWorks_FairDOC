#!/bin/bash

# Script: generate_app2_structure.sh
# Description: Generates the src/app2/ directory structure with placeholder comments.

BASE_DIR="./container/app2"

# Create directory structure
mkdir -p $BASE_DIR/{core,models/{database,schemas},services/{dspy,context,chat},api/v2/endpoints}

# Define files
FILES=(
  "core/config_v2.py"
  "core/database_v2.py"
  "core/dependencies_v2.py"
  "models/database/nice_protocols.py"
  "models/database/conversation_state.py"
  "models/database/gold_standards.py"
  "models/schemas/multiturn_chat.py"
  "models/schemas/medical_triage.py"
  "services/dspy/medical_agent.py"
  "services/dspy/question_generator.py"
  "services/dspy/evaluation_optimizer.py"
  "services/context/redis_queue.py"
  "services/context/nice_lookup.py"
  "services/chat/stakeholder_router.py"
  "services/chat/raven_bridge.py"
  "api/v2/endpoints/multiturn_chat.py"
  "api/v2/endpoints/admin_dashboard.py"
  "api/v2/endpoints/evaluation_metrics.py"
  "api/v2/router_v2.py"
  "main_v2.py"
)

# Generate Python files with filename comments
for file in "${FILES[@]}"; do
  full_path="$BASE_DIR/$file"
  echo "# $(basename "$file")" > "$full_path"
  echo "Created: $full_path"
done

# Add __init__.py to all directories
find "$BASE_DIR" -type d -exec sh -c 'touch "$1/__init__.py"' _ {} \;

echo "✅ Project structure with __init__.py files generated at $BASE_DIR"
