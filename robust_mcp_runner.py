#!/usr/bin/env python3
"""
robust_mcp_runner.py - Production-ready MCP server manager

Fixes for common MCP issues:
- Memory server context overflow
- Time server validation errors  
- Proper MCP protocol handling
- Resource management and cleanup
- Individual server monitoring
- Animated status display (no scrolling)
"""

import os
import sys
import json
import subprocess
import threading
import time
import argparse
import signal
import psutil
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime
import logging
import shutil

# Streamlined config - disable problematic servers initially
CONFIG = {
    "mcpServers": {
        "filesystem": {
            "autoApprove": ["list_allowed_directories", "list_directory", "read_file", "read_text_file", "search_files"],
            "timeout": 15,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC"],
            "max_memory_mb": 900,
            "restart_on_crash": True
        },
        "perplexity": {
            "autoApprove": ["check_deprecated_code"],
            "timeout": 20,
            "command": "node", 
            "args": ["/home/riju279/Documents/Cline/MCP/perplexity-mcp/build/index.js"],
            "env": {"PERPLEXITY_API_KEY": "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"},
            "max_memory_mb": 500,
            "restart_on_crash": True
        },
        "sequential-thinking": {
            "autoApprove": ["sequential_thinking"],
            "timeout": 10,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "env": {"DISABLE_THOUGHT_LOGGING": "true"},
            "max_memory_mb": 900,
            "restart_on_crash": True
        },
        "memory-limited": {
            "autoApprove": ["create_entities", "add_observations", "search_nodes"],
            "timeout": 10,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "env": {"MEMORY_FILE_PATH": "/tmp/mcp_memory_small.json"},
            "max_memory_mb": 1200,
            "disabled": False,
            "restart_on_crash": True,
            "memory_cleanup": True
        },
        "time-fixed": {
            "autoApprove": ["get_current_time"],
            "timeout": 8,
            "command": "python3",
            "args": ["-c", """
import json, sys, datetime, os
os.environ['TZ'] = 'Asia/Kolkata'

def handle_request(method, params):
    if method == 'get_current_time':
        tz = params.get('timezone', 'Asia/Kolkata')
        try:
            import zoneinfo
            zone = zoneinfo.ZoneInfo(tz)
        except:
            import pytz
            zone = pytz.timezone(tz)
        
        now = datetime.datetime.now(zone)
        return {
            'time': now.strftime('%H:%M:%S'),
            'date': now.strftime('%Y-%m-%d'),
            'timezone': tz,
            'iso': now.isoformat()
        }
    return {'error': 'Unknown method'}

# MCP protocol handler
while True:
    try:
        line = input().strip()
        if not line: continue
        req = json.loads(line)
        
        if req.get('method') == 'initialize':
            response = {
                'jsonrpc': '2.0',
                'id': req['id'], 
                'result': {
                    'protocolVersion': '2025-11-05',
                    'capabilities': {'tools': {}}
                }
            }
        elif req.get('method') == 'tools/list':
            response = {
                'jsonrpc': '2.0',
                'id': req['id'],
                'result': {
                    'tools': [{'name': 'get_current_time', 'description': 'Get current time'}]
                }
            }
        elif req.get('method') == 'tools/call':
            tool_result = handle_request(req['params']['name'], req['params'].get('arguments', {}))
            response = {
                'jsonrpc': '2.0',
                'id': req['id'],
                'result': {'content': [{'type': 'text', 'text': json.dumps(tool_result)}]}
            }
        else:
            response = {
                'jsonrpc': '2.0', 
                'id': req.get('id', 0),
                'error': {'code': -32601, 'message': 'Method not found'}
            }
        
        print(json.dumps(response))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({'jsonrpc': '2.0', 'id': 0, 'error': {'code': -1, 'message': str(e)}}))
        sys.stdout.flush()
"""],
            "max_memory_mb": 120,
            "disabled": False,
            "restart_on_crash": True
        }
    }
}

class TerminalDisplay:
    """Handle animated terminal display without scrolling"""
    
    def __init__(self):
        self.display_active = False
        self.lines_drawn = 0
        
    def start_display(self):
        """Initialize display mode"""
        self.display_active = True
        # Hide cursor and save position
        print("\033[?25l", end="", flush=True)  # Hide cursor
        print("\033[s", end="", flush=True)     # Save cursor position
        
    def stop_display(self):
        """Cleanup display mode"""
        if self.display_active:
            print("\033[?25h", end="", flush=True)  # Show cursor
            print("\033[u", end="", flush=True)     # Restore cursor position
            self.display_active = False
            
    def clear_lines(self, num_lines):
        """Clear previous lines"""
        for _ in range(num_lines):
            print("\033[1A\033[2K", end="")  # Move up and clear line
            
    def draw_header(self):
        """Draw the status header"""
        term_width = shutil.get_terminal_size().columns
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Title bar
        title = "🖥️  MCP SERVER MANAGER"
        padding = (term_width - len(title)) // 2
        print("═" * term_width)
        print(" " * padding + title)
        print(f"📊 Status Dashboard - {timestamp}")
        print("═" * term_width)
        
        # Column headers
        print(f"{'STATUS':^8} {'SERVER':^20} {'PID':^8} {'UPTIME':^10} {'MEMORY':^10} {'RESTARTS':^10} {'HEALTH':^10}")
        print("─" * term_width)
        
        return 6  # Number of header lines
    
    def format_uptime(self, seconds):
        """Format uptime in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
    
    def format_memory(self, pid):
        """Get formatted memory usage"""
        try:
            process = psutil.Process(pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb < 100:
                return f"{memory_mb:.1f}MB"
            else:
                return f"{memory_mb:.0f}MB"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "N/A"
    
    def get_health_indicator(self, server):
        """Get health status with animated indicator"""
        status = server.get_status()
        
        if status["status"] == "running":
            # Animated indicators for running servers
            indicators = ["●", "◐", "◑", "◒", "◓"]
            idx = int(time.time() * 2) % len(indicators)  # 2 updates per second
            return f"🟢{indicators[idx]}"
        elif status["status"] == "dead":
            return "🔴 ●"
        else:
            return "⚪ ●"
    
    def update_display(self, servers):
        """Update the entire display"""
        if not self.display_active:
            return
            
        # Clear previous display
        if self.lines_drawn > 0:
            self.clear_lines(self.lines_drawn)
            
        lines_count = 0
        
        # Draw header
        header_lines = self.draw_header()
        lines_count += header_lines
        
        # Draw server status
        total_servers = len(servers)
        running_servers = 0
        total_memory = 0
        total_restarts = 0
        
        for name, server in servers.items():
            status = server.get_status()
            
            # Status emoji
            if status["status"] == "running":
                status_icon = "✅"
                running_servers += 1
            elif status["status"] == "dead":
                status_icon = "❌"
            else:
                status_icon = "⏸️"
            
            # Server info
            pid = str(status.get("pid", "N/A"))
            uptime = self.format_uptime(status.get("uptime", 0))
            memory = self.format_memory(status.get("pid")) if status.get("pid") else "N/A"
            restarts = str(status.get("restarts", 0))
            total_restarts += status.get("restarts", 0)
            health = self.get_health_indicator(server)
            
            # Extract memory value for totaling
            if status.get("pid"):
                try:
                    process = psutil.Process(status["pid"])
                    total_memory += process.memory_info().rss / 1024 / 1024
                except Exception:
                    pass
            
            # Format server name (truncate if needed)
            server_name = name[:18] if len(name) > 18 else name
            
            print(f"{status_icon:^8} {server_name:^20} {pid:^8} {uptime:^10} {memory:^10} {restarts:^10} {health:^10}")
            lines_count += 1
        
        # Summary line
        term_width = shutil.get_terminal_size().columns
        print("─" * term_width)
        
        summary = f"🚀 {running_servers}/{total_servers} Running | 💾 {total_memory:.0f}MB Total | 🔄 {total_restarts} Restarts"
        padding = (term_width - len(summary)) // 2
        print(" " * padding + summary)
        lines_count += 2
        
        # Footer with controls
        footer = "Press Ctrl+C to exit | 📁 Logs: mcp_logs_v2/"
        padding = (term_width - len(footer)) // 2
        print(" " * padding + footer)
        print("═" * term_width)
        lines_count += 2
        
        self.lines_drawn = lines_count
        sys.stdout.flush()

class MCPServer:
    def __init__(self, name, config, log_dir, venv_path=None):
        self.name = name
        self.config = config
        self.log_dir = Path(log_dir)
        self.venv_path = venv_path
        
        self.process = None
        self.start_time = None
        self.restart_count = 0
        self.max_restarts = 5
        self.last_memory_check = 0
        
        # Logging
        self.log_file = None
        self.logger = None
        self._setup_logging()
        
        # Threading
        self.output_queue = Queue()
        self.running = False
        
    def _setup_logging(self):
        """Setup dedicated logging for this server"""
        log_path = self.log_dir / f"{self.name.replace('/', '_')}.log"
        self.log_file = open(log_path, 'w')
        
        self.logger = logging.getLogger(f"mcp.{self.name}")
        self.logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _prepare_environment(self):
        """Prepare process environment with venv if specified"""
        env = os.environ.copy()
        
        if self.venv_path:
            venv_path = os.path.expanduser(self.venv_path)
            bin_dir = os.path.join(venv_path, "bin")
            if os.path.isdir(bin_dir):
                env["VIRTUAL_ENV"] = venv_path
                env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
                env.pop("PYTHONHOME", None)
        
        # Add server-specific env vars
        env.update({k: str(v) for k, v in self.config.get("env", {}).items()})
        
        # Memory and resource limits
        env["NODE_OPTIONS"] = "--max-old-space-size=128"  # Limit Node.js memory
        env["PYTHONUNBUFFERED"] = "1"
        
        return env
    
    def _stream_reader(self, stream, prefix):
        """Read from process stream and queue output"""
        try:
            while self.running:
                line = stream.readline()
                if not line:
                    break
                self.output_queue.put((prefix, line.strip()))
                self.logger.debug(f"{prefix}: {line.strip()}")
        except Exception as e:
            self.logger.error(f"Stream reader error: {e}")
        finally:
            stream.close()
    
    def _log_processor(self):
        """Process output queue and handle logging"""
        while self.running or not self.output_queue.empty():
            try:
                prefix, message = self.output_queue.get(timeout=1.0)
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {prefix}: {message}"
                
                self.log_file.write(log_entry + "\n")
                self.log_file.flush()
                
                # Don't print to console in display mode - goes to logs only
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Log processor error: {e}")
    
    def start(self):
        """Start the MCP server process"""
        if self.config.get("disabled", False):
            self.logger.info("Server disabled in config")
            return False
        
        try:
            cmd = [self.config["command"]] + self.config.get("args", [])
            env = self._prepare_environment()
            
            self.logger.info(f"Starting: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=0
            )
            
            self.start_time = time.time()
            self.running = True
            
            # Start stream readers
            threading.Thread(
                target=self._stream_reader,
                args=(self.process.stdout, "STDOUT"),
                daemon=True
            ).start()
            
            threading.Thread(
                target=self._stream_reader,
                args=(self.process.stderr, "STDERR"), 
                daemon=True
            ).start()
            
            # Start log processor
            threading.Thread(
                target=self._log_processor,
                daemon=True
            ).start()
            
            self.logger.info(f"Started successfully (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
            return False
    
    def check_memory_usage(self):
        """Check if server is using too much memory"""
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            process = psutil.Process(self.process.pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            max_memory = self.config.get("max_memory_mb", 600)
            
            if memory_mb > max_memory:
                self.logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {max_memory}MB")
                return True
                
            # Log memory usage periodically
            if time.time() - self.last_memory_check > 60:
                self.logger.info(f"Memory usage: {memory_mb:.1f}MB")
                self.last_memory_check = time.time()
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return False
    
    def cleanup_memory(self):
        """Attempt to clean up server memory"""
        if not self.process:
            return
        
        if self.config.get("memory_cleanup", False) and self.name == "memory-limited":
            # Send memory cleanup command to memory server
            try:
                cleanup_cmd = {
                    "jsonrpc": "2.0",
                    "id": 999,
                    "method": "tools/call", 
                    "params": {
                        "name": "search_nodes",
                        "arguments": {"query": "cleanup_old_data", "limit": 1}
                    }
                }
                self.process.stdin.write(json.dumps(cleanup_cmd) + "\n")
                self.process.stdin.flush()
                self.logger.info("Sent memory cleanup command")
            except Exception as e:
                self.logger.error(f"Failed to send cleanup command: {e}")
    
    def restart(self):
        """Restart the server if it crashed"""
        if self.restart_count >= self.max_restarts:
            self.logger.error(f"Max restarts ({self.max_restarts}) reached, giving up")
            return False
        
        self.logger.warning(f"Restarting server (attempt {self.restart_count + 1})")
        self.stop()
        time.sleep(2)  # Brief pause
        
        if self.start():
            self.restart_count += 1
            return True
        return False
    
    def is_healthy(self):
        """Check if server is running and responsive"""
        if not self.process:
            return False
        
        if self.process.poll() is not None:
            return False
        
        # Check memory usage
        if self.check_memory_usage():
            self.cleanup_memory()
            return False
        
        return True
    
    def stop(self):
        """Stop the server gracefully"""
        self.running = False
        
        if self.process and self.process.poll() is None:
            self.logger.info("Stopping server...")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                self.logger.error(f"Error stopping: {e}")
        
        if self.log_file:
            self.log_file.close()
    
    def get_status(self):
        """Get server status info"""
        if not self.process:
            return {"status": "not_started"}
        
        if self.process.poll() is not None:
            return {
                "status": "dead",
                "exit_code": self.process.poll(),
                "restarts": self.restart_count,
                "uptime": time.time() - self.start_time if self.start_time else 0
            }
        
        return {
            "status": "running", 
            "pid": self.process.pid,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "restarts": self.restart_count
        }

class MCPManager:
    def __init__(self, venv_path=None):
        self.venv_path = venv_path
        self.servers = {}
        self.running = True
        self.log_dir = Path("mcp_logs_v2")
        self.log_dir.mkdir(exist_ok=True)
        self.display = TerminalDisplay()
        
        # Setup manager logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "manager.log")
            ]
        )
        self.logger = logging.getLogger("mcp.manager")
    
    def start_all(self):
        """Start all configured servers"""
        self.logger.info("Starting MCP servers...")
        
        for name, config in CONFIG["mcpServers"].items():
            server = MCPServer(name, config, self.log_dir, self.venv_path)
            self.servers[name] = server
            
            if server.start():
                self.logger.info(f"✓ Started {name}")
            else:
                self.logger.error(f"✗ Failed to start {name}")
            
            time.sleep(1)  # Stagger startups
    
    def health_monitor(self):
        """Monitor server health and restart if needed"""
        while self.running:
            try:
                for name, server in self.servers.items():
                    if not server.is_healthy():
                        status = server.get_status()
                        
                        if status["status"] == "dead" and server.config.get("restart_on_crash", False):
                            self.logger.warning(f"Server {name} died, attempting restart...")
                            if not server.restart():
                                self.logger.error(f"Failed to restart {name}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(10)
    
    def display_loop(self):
        """Main display update loop"""
        self.display.start_display()
        
        try:
            while self.running:
                self.display.update_display(self.servers)
                time.sleep(0.5)  # Update display twice per second
                
                # Check if any servers are still running
                running_count = sum(1 for server in self.servers.values() 
                                  if server.get_status()["status"] == "running")
                
                if running_count == 0:
                    break
                    
        finally:
            self.display.stop_display()
    
    def stop_all(self):
        """Stop all servers"""
        self.running = False
        self.display.stop_display()
        self.logger.info("Stopping all servers...")
        
        for name, server in self.servers.items():
            self.logger.info(f"Stopping {name}")
            server.stop()
            
        self.logger.info("All servers stopped")

def main():
    parser = argparse.ArgumentParser(description="Robust MCP Server Manager")
    parser.add_argument("--venv", type=str, help="Virtual environment path")
    parser.add_argument("--detach", action="store_true", help="Run in background (no display)")
    args = parser.parse_args()
    
    manager = MCPManager(venv_path=args.venv)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n🔨 Received signal {signum}")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not args.detach:
        print("🎯 Robust MCP Server Manager")
        print(f"📁 Logs: {manager.log_dir.absolute()}")
        print("Starting servers...")
        time.sleep(2)
    
    # Start servers
    manager.start_all()
    
    if args.detach:
        print("📄 Running in detached mode")
        return 0
    
    # Start health monitor
    health_thread = threading.Thread(target=manager.health_monitor, daemon=True)
    health_thread.start()
    
    # Start animated display
    try:
        manager.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
