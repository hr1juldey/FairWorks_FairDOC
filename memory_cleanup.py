#!/usr/bin/env python3
"""
memory_cleanup.py - Clean up and optimize MCP memory server data

Fixes context saturation issues by:
- Removing duplicate entries
- Consolidating related observations  
- Limiting memory file size
- Creating focused, relevant memories
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

def load_memory_file(path):
    """Load and parse memory file"""
    memories = []
    if not Path(path).exists():
        print(f"Memory file {path} not found")
        return memories
    
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    memories.append(json.loads(line))
        print(f"Loaded {len(memories)} memory entries")
        return memories
    except Exception as e:
        print(f"Error loading memory file: {e}")
        return []

def analyze_memory_content(memories):
    """Analyze memory content for cleanup opportunities"""
    stats = {
        'entities': 0,
        'relations': 0,
        'observations': 0,
        'duplicates': 0,
        'entity_types': defaultdict(int),
        'relation_types': defaultdict(int)
    }
    
    seen_entities = set()
    seen_relations = set()
    
    for memory in memories:
        if memory.get('type') == 'entity':
            stats['entities'] += 1
            stats['entity_types'][memory.get('entityType', 'unknown')] += 1
            
            entity_key = (memory.get('name'), memory.get('entityType'))
            if entity_key in seen_entities:
                stats['duplicates'] += 1
            else:
                seen_entities.add(entity_key)
                
            stats['observations'] += len(memory.get('observations', []))
            
        elif memory.get('type') == 'relation':
            stats['relations'] += 1
            stats['relation_types'][memory.get('relationType', 'unknown')] += 1
            
            relation_key = (memory.get('from'), memory.get('to'), memory.get('relationType'))
            if relation_key in seen_relations:
                stats['duplicates'] += 1
            else:
                seen_relations.add(relation_key)
    
    return stats

def cleanup_memory_data(memories, max_entities=20, max_observations_per_entity=10):
    """Clean up memory data to prevent context saturation"""
    
    # Group entities by name and type
    entities_map = {}
    relations = []
    
    for memory in memories:
        if memory.get('type') == 'entity':
            key = (memory.get('name'), memory.get('entityType'))
            if key not in entities_map:
                entities_map[key] = {
                    'name': memory.get('name'),
                    'entityType': memory.get('entityType'),
                    'observations': []
                }
            # Merge observations
            entities_map[key]['observations'].extend(memory.get('observations', []))
            
        elif memory.get('type') == 'relation':
            relations.append(memory)
    
    # Deduplicate and limit observations
    for key, entity in entities_map.items():
        # Remove duplicate observations
        unique_observations = list(set(key['observations'], entity['observations']))
        
        # Sort by relevance (longer, more specific observations first)
        unique_observations.sort(key=len, reverse=True)
        
        # Limit to max observations
        entity['observations'] = unique_observations[:max_observations_per_entity]
    
    # Keep most important entities (prioritize project and person entities)
    priority_types = ['project', 'person', 'system', 'file', 'code']
    sorted_entities = []
    
    for priority_type in priority_types:
        type_entities = [(k, v) for k, v in entities_map.items() if v['entityType'] == priority_type]
        # Sort by number of observations (more detailed entities first)
        type_entities.sort(key=lambda x: len(x[1]['observations']), reverse=True)
        sorted_entities.extend(type_entities)
    
    # Add remaining entities
    remaining = [(k, v) for k, v in entities_map.items() if v['entityType'] not in priority_types]
    remaining.sort(key=lambda x: len(x[1]['observations']), reverse=True)
    sorted_entities.extend(remaining)
    
    # Keep only top entities
    kept_entities = sorted_entities[:max_entities]
    kept_entity_names = {entity[1]['name'] for entity in kept_entities}
    
    # Filter relations to only include those between kept entities
    filtered_relations = []
    for relation in relations:
        from_name = relation.get('from')
        to_name = relation.get('to')
        if from_name in kept_entity_names and to_name in kept_entity_names:
            filtered_relations.append(relation)
    
    # Remove duplicate relations
    seen_relations = set()
    unique_relations = []
    for relation in filtered_relations:
        key = (relation.get('from'), relation.get('to'), relation.get('relationType'))
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relations.append(relation)
    
    return [entity[1] for entity in kept_entities], unique_relations

def create_optimized_memory(entities, relations):
    """Create optimized memory entries"""
    optimized = []
    
    # Add entities
    for entity in entities:
        optimized.append({
            'type': 'entity',
            'name': entity['name'],
            'entityType': entity['entityType'],
            'observations': entity['observations']
        })
    
    # Add relations
    for relation in relations:
        optimized.append({
            'type': 'relation',
            'from': relation['from'],
            'to': relation['to'],
            'relationType': relation['relationType']
        })
    
    return optimized

def save_memory_file(memories, path, backup=True):
    """Save cleaned memory data"""
    if backup and Path(path).exists():
        backup_path = f"{path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(path, backup_path)
        print(f"Backup saved to: {backup_path}")
    
    with open(path, 'w') as f:
        for memory in memories:
            f.write(json.dumps(memory) + '\n')
    
    print(f"Saved {len(memories)} optimized memory entries to {path}")

def create_fresh_memory_for_project():
    """Create a fresh, focused memory for the FairDoc project"""
    fresh_memories = [
        {
            "type": "entity",
            "name": "FairDoc_AI_Triage",
            "entityType": "project",
            "observations": [
                "Medical AI triage system using FastAPI",
                "Located at /home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC",
                "Uses DSPy for AI optimization",
                "Implements medical decision support",
                "Has MCP server integration issues"
            ]
        },
        {
            "type": "entity", 
            "name": "Riju279",
            "entityType": "developer",
            "observations": [
                "Primary developer of FairDoc system",
                "Located in Asia/Kolkata timezone", 
                "Works with Python, FastAPI, AI frameworks",
                "Experiencing MCP server context saturation issues"
            ]
        },
        {
            "type": "entity",
            "name": "MCP_Servers",
            "entityType": "system",
            "observations": [
                "Memory server causing context overflow",
                "Time server has validation errors",
                "Need memory cleanup and proper protocol handling",
                "Running multiple servers concurrently"
            ]
        },
        {
            "type": "relation",
            "from": "Riju279", 
            "to": "FairDoc_AI_Triage",
            "relationType": "develops"
        },
        {
            "type": "relation",
            "from": "FairDoc_AI_Triage",
            "to": "MCP_Servers", 
            "relationType": "uses"
        }
    ]
    
    return fresh_memories

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up MCP memory server data")
    parser.add_argument("--memory-file", type=str, 
                       default="/home/riju279/Documents/Cline/MCP/memory-server/memory.json",
                       help="Path to memory file")
    parser.add_argument("--fresh-start", action="store_true",
                       help="Create fresh memory instead of cleaning existing")
    parser.add_argument("--max-entities", type=int, default=15,
                       help="Maximum entities to keep")
    parser.add_argument("--max-observations", type=int, default=8,
                       help="Maximum observations per entity")
    
    args = parser.parse_args()
    
    print("🧹 MCP Memory Cleanup Tool")
    print(f"📂 Memory file: {args.memory_file}")
    print("=" * 50)
    
    if args.fresh_start:
        print("🆕 Creating fresh memory...")
        fresh_memories = create_fresh_memory_for_project()
        save_memory_file(fresh_memories, args.memory_file)
        
        print("\n📊 Fresh Memory Summary:")
        stats = analyze_memory_content(fresh_memories)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {dict(value)}")
            else:
                print(f"  {key}: {value}")
    else:
        # Load existing memory
        memories = load_memory_file(args.memory_file)
        
        if not memories:
            print("No existing memory found, creating fresh memory...")
            fresh_memories = create_fresh_memory_for_project()
            save_memory_file(fresh_memories, args.memory_file)
            return
        
        # Analyze current state
        print("\n📊 Current Memory Analysis:")
        stats = analyze_memory_content(memories)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {dict(value)}")
            else:
                print(f"  {key}: {value}")
        
        # Clean up
        print("\n🔧 Cleaning up memory data...")
        entities, relations = cleanup_memory_data(
            memories, 
            max_entities=args.max_entities,
            max_observations_per_entity=args.max_observations
        )
        
        # Create optimized memory
        optimized_memories = create_optimized_memory(entities, relations)
        
        # Save cleaned data
        save_memory_file(optimized_memories, args.memory_file)
        
        # Show cleanup results
        print("\n📊 Cleanup Results:")
        new_stats = analyze_memory_content(optimized_memories)
        for key, value in new_stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {dict(value)}")
            else:
                print(f"  {key}: {value}")
        
        reduction = len(memories) - len(optimized_memories)
        print(f"\n✅ Reduced memory entries by {reduction} ({reduction / len(memories) * 100:.1f}%)")
    
    print("\n🎯 Memory cleanup complete!")
    print(f"📁 Clean memory file: {args.memory_file}")
    print("💡 Restart your MCP servers to use the cleaned memory")

if __name__ == "__main__":
    main()
