"""
Terminal Handler - WebSocket per-terminal I/O

Manages individual terminal sessions with WebSocket communication.
Handles real-time bidirectional communication between web terminals
and running processes (servers/tests).

Single responsibility: WebSocket terminal I/O management
File: terminal_handler.py
"""

import asyncio
import json
import uuid
from typing import Dict, Optional, Set
from datetime import datetime
import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)


class TerminalSession:
    """Individual terminal session with process management"""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.process: Optional[asyncio.subprocess.Process] = None
        self.created_at = datetime.now()
        self.is_active = True
        self.command_history = []
        
    async def send_output(self, data: str, output_type: str = "stdout"):
        """Send output to WebSocket client"""
        if not self.is_active:
            return
            
        try:
            message = {
                "type": "output",
                "output_type": output_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error("Failed to send output", session_id=self.session_id, error=str(e))
            self.is_active = False
    
    async def execute_command(self, command: str, cwd: str = None):
        """Execute command and stream output"""
        self.command_history.append(command)
        logger.info("Executing command", session_id=self.session_id, command=command)
        
        try:
            # Start process
            self.process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            # Stream stdout and stderr concurrently
            await asyncio.gather(
                self._stream_output(self.process.stdout, "stdout"),
                self._stream_output(self.process.stderr, "stderr"),
                return_exceptions=True
            )
            
            # Wait for process completion
            return_code = await self.process.wait()
            await self.send_output(f"\n✅ Process completed with exit code: {return_code}\n", "system")
            
        except Exception as e:
            logger.error("Command execution failed", session_id=self.session_id, error=str(e))
            await self.send_output(f"\n❌ Error: {str(e)}\n", "stderr")
    
    async def _stream_output(self, stream, output_type: str):
        """Stream output from subprocess to WebSocket"""
        while True:
            try:
                line = await stream.readline()
                if not line:
                    break
                    
                text = line.decode('utf-8', errors='replace')
                await self.send_output(text, output_type)
                
            except Exception as e:
                logger.error("Stream error", session_id=self.session_id, error=str(e))
                break
    
    async def send_input(self, data: str):
        """Send input to running process"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(data.encode('utf-8'))
                await self.process.stdin.drain()
            except Exception as e:
                logger.error("Failed to send input", session_id=self.session_id, error=str(e))
    
    async def terminate(self):
        """Terminate the terminal session and cleanup"""
        self.is_active = False
        
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
            except Exception as e:
                logger.error("Process termination error", session_id=self.session_id, error=str(e))


class TerminalManager:
    """Manages multiple terminal sessions and WebSocket connections"""
    
    def __init__(self):
        self.sessions: Dict[str, TerminalSession] = {}
        self.websockets: Set[WebSocket] = set()
        
    def create_session(self, websocket: WebSocket) -> str:
        """Create new terminal session"""
        session_id = str(uuid.uuid4())[:8]
        session = TerminalSession(session_id, websocket)
        self.sessions[session_id] = session
        self.websockets.add(websocket)
        
        logger.info("Terminal session created", session_id=session_id)
        return session_id
    
    async def handle_websocket(self, websocket: WebSocket, session_id: str = None):
        """Handle WebSocket connection for terminal"""
        await websocket.accept()
        
        if not session_id:
            session_id = self.create_session(websocket)
        
        session = self.sessions.get(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return
        
        # Send welcome message
        await session.send_output("🖤 Death Note Terminal Connected\n", "system")
        await session.send_output(f"📝 Session ID: {session_id}\n", "system")
        await session.send_output("💀 Type your commands...\n\n", "system")
        
        try:
            while session.is_active:
                # Receive message from WebSocket
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await self._process_message(session, message)
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", session_id=session_id)
        except Exception as e:
            logger.error("WebSocket error", session_id=session_id, error=str(e))
        finally:
            await self.cleanup_session(session_id)
    
    async def _process_message(self, session: TerminalSession, message: dict):
        """Process incoming WebSocket message"""
        msg_type = message.get("type")
        
        if msg_type == "command":
            command = message.get("data", "").strip()
            cwd = message.get("cwd")
            
            if command:
                await session.execute_command(command, cwd)
                
        elif msg_type == "input":
            data = message.get("data", "")
            await session.send_input(data)
            
        elif msg_type == "interrupt":
            if session.process:
                session.process.terminate()
                await session.send_output("\n🛑 Process interrupted\n", "system")
        
        elif msg_type == "ping":
            await session.send_output("", "pong")
    
    async def cleanup_session(self, session_id: str):
        """Clean up terminal session"""
        session = self.sessions.get(session_id)
        if session:
            await session.terminate()
            self.websockets.discard(session.websocket)
            del self.sessions[session_id]
            logger.info("Terminal session cleaned up", session_id=session_id)
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get session information"""
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active,
            "command_count": len(session.command_history),
            "last_commands": session.command_history[-5:] if session.command_history else []
        }
    
    def list_sessions(self) -> list:
        """List all active sessions"""
        return [
            self.get_session_info(session_id) 
            for session_id in self.sessions.keys()
        ]


# Global terminal manager instance
terminal_manager = TerminalManager()