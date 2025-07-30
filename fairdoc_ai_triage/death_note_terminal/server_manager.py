"""
Death Note Terminal - Server Process Manager

Handles starting/stopping V1/V2 servers on different port configurations.
Single responsibility: Process lifecycle management.

File: server_manager.py
"""

import subprocess
import threading
import psutil
import time as time_module
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

class ServerProcess:
    """Represents a running server process"""
    
    def __init__(self, name: str, config_data: Dict, process: subprocess.Popen):
        self.name = name
        self.config = config_data
        self.process = process
        self.start_time = time_module.time()
        self.custom_port = config_data.get("custom", False)
    
    @property
    def pid(self) -> Optional[int]:
        """Get process ID"""
        return self.process.pid if self.process else None
    
    @property
    def is_running(self) -> bool:
        """Check if process is still running"""
        if not self.process:
            return False
        return self.process.poll() is None
    
    @property
    def uptime(self) -> float:
        """Get uptime in seconds"""
        return time_module.time() - self.start_time
    
    @property
    def memory_usage(self) -> float:
        """Get memory usage in MB"""
        try:
            if self.pid and psutil.pid_exists(self.pid):
                process = psutil.Process(self.pid)
                return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return 0.0
    
    def get_status(self) -> Dict:
        """Get comprehensive process status"""
        return {
            "name": self.name,
            "pid": self.pid,
            "port": self.config.get("port"),
            "module": self.config.get("module"),
            "description": self.config.get("description"),
            "running": self.is_running,
            "uptime": self.uptime,
            "memory_mb": self.memory_usage,
            "custom_port": self.custom_port,
            "endpoints": self.config.get("endpoints", [])
        }

class ServerManager:
    """Death Note Terminal Server Process Manager"""
    
    def __init__(self):
        self.processes: Dict[str, ServerProcess] = {}
        self.root_dir = config.ROOT_DIR
        
    def start_server(self, server_type: str, custom_port: Optional[int] = None) -> Tuple[bool, str]:
        """Start a server with specified configuration"""
        
        # Get server configuration
        if custom_port:
            server_config = config.get_custom_port_config(server_type, custom_port)
        else:
            server_config = config.get_server_config(server_type)
        
        if not server_config:
            return False, f"Unknown server type: {server_type}"
        
        # Check if already running
        if server_type in self.processes and self.processes[server_type].is_running:
            return False, f"Server {server_type} is already running"
        
        # Check port availability
        port = server_config["port"]
        if self.is_port_in_use(port):
            return False, f"Port {port} is already in use"
        
        try:
            # Build uvicorn command
            cmd = [
                "python", "-m", "uvicorn",
                server_config["module"],
                "--host", config.HOST,
                "--port", str(port),
                "--reload",
                "--log-level", "info"
            ]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=4,
                universal_newlines=True
            )
            
            # Wait for server startup or process failure
            for i in range(40):  # 40 seconds total
                time_module.sleep(2)
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    error_msg = stderr or stdout or "Unknown startup error"
                    return False, f"Server failed to start: {error_msg}"
                if self.is_port_in_use(port):
                    logger.info(f"Server {server_type} started successfully in {i * 2} seconds")
                    break
            else:
                return False, f"Server did not start within {40 * 2} seconds"

            # Process is still running after timeout - assume success
            logger.info(f"Server {server_type} startup completed after {i * 2} seconds")

            
            
            # Store process
            server_process = ServerProcess(server_type, server_config, process)
            self.processes[server_type] = server_process
            # Start log streaming in background

            def stream_logs():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line.strip():
                            # Store log line for WebSocket streaming
                            if not hasattr(server_process, 'log_buffer'):
                                server_process.log_buffer = []
                            server_process.log_buffer.append({
                                "timestamp": time_module.time(),
                                "line": line.strip(),
                                "server": server_type
                            })
                            # Keep only last 100 lines
                            if len(server_process.log_buffer) > 100:
                                server_process.log_buffer.pop(0)
                except Exception as e:
                    logger.error(f"Log streaming error for {server_type}: {e}")

            log_thread = threading.Thread(target=stream_logs, daemon=True)
            log_thread.start()

            # Verify server is actually responding
            time_module.sleep(3)  # Brief startup delay
            if server_process.is_running:
                logger.info(f"✅ Started {server_type} server on port {port} (PID: {process.pid})")
                return True, f"Server {server_type} started successfully on port {port}"
            else:
                logger.error(f"❌ Server {server_type} died immediately after startup")
                return False, f"Server {server_type} failed to start - process died"

            
        except Exception as e:
            logger.error(f"Failed to start {server_type}: {e}")
            return False, f"Failed to start server: {str(e)}"


    def stop_server(self, server_type: str) -> Tuple[bool, str]:
        """Stop a specific server"""
        
        if server_type not in self.processes:
            return False, f"Server {server_type} is not running"
        
        server_process = self.processes[server_type]
        
        try:
            # Graceful termination
            server_process.process.terminate()
            
            # Wait for graceful shutdown
            try:
                server_process.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                server_process.process.kill()
                server_process.process.wait()
            
            # Remove from tracking
            del self.processes[server_type]
            
            logger.info(f"Stopped {server_type} server")
            return True, f"Server {server_type} stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping {server_type}: {e}")
            return False, f"Failed to stop server: {str(e)}"
    
    def stop_all_servers(self) -> List[str]:
        """Stop all running servers"""
        stopped = []
        
        for server_type in list(self.processes.keys()):
            success, _ = self.stop_server(server_type)
            if success:
                stopped.append(server_type)
        
        return stopped
    
    def get_server_status(self, server_type: str) -> Optional[Dict]:
        """Get status of specific server"""
        if server_type in self.processes:
            return self.processes[server_type].get_status()
        return None
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all servers"""
        status = {}
        
        # Include configured servers
        for server_type in config.SERVER_CONFIGS.keys():
            if server_type in self.processes:
                status[server_type] = self.processes[server_type].get_status()
            else:
                status[server_type] = {
                    "name": server_type,
                    "running": False,
                    "port": config.SERVER_CONFIGS[server_type]["port"],
                    "description": config.SERVER_CONFIGS[server_type]["description"]
                }
        
        return status
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return True
        except (psutil.AccessDenied, AttributeError):
            pass
        return False
    
    def get_available_ports(self) -> List[int]:
        """Get list of available ports"""
        available = []
        for port in config.AVAILABLE_PORTS:
            if not self.is_port_in_use(port):
                available.append(port)
        return available
    
    def cleanup_dead_processes(self):
        """Remove dead processes from tracking"""
        dead_processes = []
        
        for server_type, server_process in self.processes.items():
            if not server_process.is_running:
                dead_processes.append(server_type)
        
        for server_type in dead_processes:
            del self.processes[server_type]
            logger.info(f"Cleaned up dead process: {server_type}")
    
    def restart_server(self, server_type: str) -> Tuple[bool, str]:
        """Restart a server"""
        # Get current config if running
        custom_port = None
        if server_type in self.processes:
            custom_port = self.processes[server_type].config.get("port")
            if not self.processes[server_type].custom_port:
                custom_port = None
        
        # Stop if running
        if server_type in self.processes:
            stop_success, stop_msg = self.stop_server(server_type)
            if not stop_success:
                return False, f"Failed to stop for restart: {stop_msg}"
        
        # Start again
        time_module.sleep(1)  # Brief pause
        return self.start_server(server_type, custom_port)

# Global server manager instance
server_manager = ServerManager()
