#!/usr/bin/env python3
"""
mcp_diagnostic.py - Diagnose MCP server issues

Quick diagnostic tool to identify problems with your current setup
"""

import os
import sys
import json
import subprocess
import psutil
from pathlib import Path

def check_processes():
    """Check for running MCP processes"""
    print("🔍 Checking running MCP processes...")
    
    mcp_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            if any(keyword in cmdline.lower() for keyword in ['mcp-server', 'modelcontextprotocol', 'uvx', '@modelcontextprotocol']):
                memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                uptime = psutil.boot_time() - proc.info['create_time']
                mcp_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline[:80] + '...' if len(cmdline) > 80 else cmdline,
                    'memory_mb': memory_mb,
                    'uptime': uptime
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if mcp_processes:
        print(f"  📊 Found {len(mcp_processes)} MCP processes:")
        for proc in mcp_processes:
            print(f"    PID {proc['pid']:6} | {proc['memory_mb']:6.1f}MB | {proc['name']}")
            print(f"        CMD: {proc['cmdline']}")
    else:
        print("  ✅ No MCP processes currently running")
    
    return mcp_processes

def check_memory_file():
    """Check memory server file issues"""
    print("\n🧠 Checking memory server files...")
    
    memory_paths = [
        "/home/riju279/Documents/Cline/MCP/memory-server/memory.json",
        "/tmp/mcp_memory_small.json"
    ]
    
    for path in memory_paths:
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    
                file_size = Path(path).stat().st_size
                print(f"  📂 {path}")
                print(f"      Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
                print(f"      Lines: {len(lines)}")
                
                # Analyze content
                entities = sum(1 for line in lines if '"type":"entity"' in line)
                relations = sum(1 for line in lines if '"type":"relation"' in line)
                
                print(f"      Entities: {entities}, Relations: {relations}")
                
                if file_size > 50000:  # 50KB threshold
                    print("      ⚠️ File is large, may cause context issues")
                
                # Check for problematic content
                total_observations = 0
                for line in lines:
                    if '"observations":[' in line:
                        try:
                            obj = json.loads(line.strip())
                            total_observations += len(obj.get('observations', []))
                        except Exception as e:
                            print(f"  ❌ Error reading lines {line}: {e}")
                            pass
                
                if total_observations > 50:
                    print(f"      ⚠️ {total_observations} total observations may cause memory bloat")
                
            except Exception as e:
                print(f"  ❌ Error reading {path}: {e}")
        else:
            print(f"  📂 {path} - Not found")

def check_node_memory():
    """Check Node.js memory settings"""
    print("\n🟢 Checking Node.js memory configuration...")
    
    node_options = os.environ.get('NODE_OPTIONS', '')
    if node_options:
        print(f"  📋 NODE_OPTIONS: {node_options}")
    else:
        print("  ⚠️ NODE_OPTIONS not set (may cause memory issues)")
        print("     💡 Recommended: export NODE_OPTIONS='--max-old-space-size=128'")

def check_system_resources():
    """Check system resource usage"""
    print("\n💻 System resource check...")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"  🧠 Memory: {memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB ({memory.percent:.1f}% used)")
    
    if memory.percent > 80:
        print("     ⚠️ High memory usage may affect MCP servers")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"  🔥 CPU: {cpu_percent:.1f}% usage")
    
    # Check available disk space
    disk = psutil.disk_usage('/')
    disk_free_gb = disk.free / 1024**3
    print(f"  💾 Disk: {disk_free_gb:.1f}GB free ({disk.percent:.1f}% used)")

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Dependency check...")
    
    dependencies = [
        ('python3', '--version'),
        ('node', '--version'),
        ('npx', '--version'),
        ('uvx', '--version')
    ]
    
    for cmd, arg in dependencies:
        try:
            result = subprocess.run([cmd, arg], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                print(f"  ✅ {cmd}: {version}")
            else:
                print(f"  ❌ {cmd}: Not working properly")
        except FileNotFoundError:
            print(f"  ❌ {cmd}: Not found")
        except subprocess.TimeoutExpired:
            print(f"  ⏱️ {cmd}: Timeout")
        except Exception as e:
            print(f"  ❌ {cmd}: Error - {e}")

def check_log_files():
    """Check existing log files"""
    print("\n📄 Log file check...")
    
    log_dirs = ['logs', 'mcp_logs', 'mcp_logs_v2']
    
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if log_path.exists():
            log_files = list(log_path.glob('*.log'))
            print(f"  📁 {log_dir}: {len(log_files)} log files")
            
            # Check recent logs
            recent_logs = []
            for log_file in log_files:
                if log_file.stat().st_size > 0:
                    recent_logs.append(log_file)
            
            if recent_logs:
                print(f"      📝 Recent logs with content: {len(recent_logs)}")
                # Show largest log files
                recent_logs.sort(key=lambda x: x.stat().st_size, reverse=True)
                for log_file in recent_logs[:3]:
                    size_kb = log_file.stat().st_size / 1024
                    print(f"        {log_file.name}: {size_kb:.1f}KB")
        else:
            print(f"  📁 {log_dir}: Not found")

def recommend_fixes():
    """Provide fix recommendations"""
    print("\n🔧 Recommendations:")
    
    # Check for high-memory processes
    mcp_procs = check_processes()
    high_memory_procs = [p for p in mcp_procs if p['memory_mb'] > 100]
    
    if high_memory_procs:
        print("  1. 🧠 High memory usage detected:")
        for proc in high_memory_procs:
            print(f"     - Kill PID {proc['pid']} ({proc['memory_mb']:.1f}MB)")
        print(f"     💡 Run: kill {' '.join(str(p['pid']) for p in high_memory_procs)}")
    
    # Check memory file
    memory_file = Path("/home/riju279/Documents/Cline/MCP/memory-server/memory.json")
    if memory_file.exists() and memory_file.stat().st_size > 10000:
        print("  2. 🧹 Large memory file detected:")
        print("     💡 Run memory cleanup script to reduce context saturation")
    
    # Check for missing NODE_OPTIONS
    if not os.environ.get('NODE_OPTIONS'):
        print("  3. ⚙️ Set Node.js memory limit:")
        print("     💡 export NODE_OPTIONS='--max-old-space-size=128'")
    
    print("  4. 🚀 Use the new robust MCP runner:")
    print("     💡 python3 robust_mcp_runner.py --venv /path/to/venv")
    
    print("  5. 🎯 For immediate relief:")
    print("     💡 pkill -f 'mcp-server' && python3 memory_cleanup.py --fresh-start")

def main():
    print("🔍 MCP Server Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    check_processes()
    check_memory_file()
    check_node_memory()
    check_system_resources()
    check_dependencies()
    check_log_files()
    
    print("\n" + "=" * 50)
    recommend_fixes()
    
    print("\n✅ Diagnostic complete!")
    print("💡 Use the robust MCP runner to fix most issues automatically")

if __name__ == "__main__":
    main()
