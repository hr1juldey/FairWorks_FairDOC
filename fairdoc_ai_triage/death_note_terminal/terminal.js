/**
 * Death Note Terminal - xterm.js Integration
 * 
 * Handles terminal UI rendering with xterm.js library,
 * WebSocket communication, and terminal-specific interactions.
 * 
 * Single responsibility: Terminal UI and xterm.js management
 * File: terminal.js
 */

class DeathNoteTerminal {
    constructor(terminalId, containerId) {
        this.terminalId = terminalId;
        this.containerId = containerId;
        this.terminal = null;
        this.websocket = null;
        this.sessionId = null;
        this.isConnected = false;
        
        this.init();
    }

    init() {
        this.setupTerminal();
        this.connectWebSocket();
    }

    setupTerminal() {
        // xterm.js terminal configuration
        this.terminal = new Terminal({
            theme: this.getTheme(),
            fontFamily: '"Courier New", "DejaVu Sans Mono", monospace',
            fontSize: 14,
            lineHeight: 1.2,
            cursorBlink: true,
            cursorStyle: 'block',
            scrollback: 1000,
            tabStopWidth: 4,
            bellStyle: 'sound',
            allowTransparency: true,
            // Death Note aesthetic
            rendererType: 'canvas'
        });

        // Fit addon for responsive terminal
        const fitAddon = new FitAddon.FitAddon();
        this.terminal.loadAddon(fitAddon);

        // Web links addon
        const webLinksAddon = new WebLinksAddon.WebLinksAddon();
        this.terminal.loadAddon(webLinksAddon);

        // Search addon
        const searchAddon = new SearchAddon.SearchAddon();
        this.terminal.loadAddon(searchAddon);

        // Open terminal in container
        const container = document.getElementById(this.containerId);
        if (container) {
            this.terminal.open(container);
            fitAddon.fit();
        }

        // Handle terminal input
        this.terminal.onData(data => {
            if (this.isConnected && this.websocket) {
                this.sendInput(data);
            }
        });

        // Handle terminal resize
        this.terminal.onResize(size => {
            if (this.isConnected && this.websocket) {
                this.sendMessage({
                    type: 'resize',
                    data: { cols: size.cols, rows: size.rows }
                });
            }
        });

        // Resize on window resize
        window.addEventListener('resize', () => {
            fitAddon.fit();
        });

        // Theme switching support
        document.addEventListener('theme-changed', () => {
            this.terminal.setOption('theme', this.getTheme());
        });
    }

    getTheme() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        
        if (isDark) {
            return {
                background: '#0a0a0a',    // Near black
                foreground: '#e0e0e0',    // Light gray
                cursor: '#ff0000',        // Red cursor (Death Note theme)
                cursorAccent: '#ffffff',
                selection: '#333333',
                black: '#000000',
                red: '#ff4444',
                green: '#44ff44',
                yellow: '#ffff44',
                blue: '#4444ff',
                magenta: '#ff44ff',
                cyan: '#44ffff',
                white: '#ffffff',
                brightBlack: '#444444',
                brightRed: '#ff6666',
                brightGreen: '#66ff66',
                brightYellow: '#ffff66',
                brightBlue: '#6666ff',
                brightMagenta: '#ff66ff',
                brightCyan: '#66ffff',
                brightWhite: '#ffffff'
            };
        } else {
            return {
                background: '#f5f5f0',    // Paper white
                foreground: '#2c2c2c',    // Dark text
                cursor: '#cc0000',        // Dark red cursor
                cursorAccent: '#000000',
                selection: '#d4d4aa',
                black: '#000000',
                red: '#cc0000',
                green: '#006600',
                yellow: '#cc6600',
                blue: '#0066cc',
                magenta: '#cc00cc',
                cyan: '#0066cc',
                white: '#2c2c2c',
                brightBlack: '#666666',
                brightRed: '#ff3333',
                brightGreen: '#33cc33',
                brightYellow: '#ff9933',
                brightBlue: '#3399ff',
                brightMagenta: '#ff33ff',
                brightCyan: '#33ccff',
                brightWhite: '#333333'
            };
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/terminal`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            this.isConnected = true;
            this.terminal.writeln('🖤 Connected to Death Note Terminal Server');
            this.terminal.writeln('💀 Welcome to the underworld of development...\r\n');
            this.sendMessage({ type: 'init', terminal_id: this.terminalId });
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = () => {
            this.isConnected = false;
            this.terminal.writeln('\r\n❌ Connection closed');
            this.terminal.writeln('🔄 Attempting to reconect...');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.terminal.writeln('\r\n❌ Connection error occurred');
        };
    }

    handleMessage(message) {
        switch (message.type) {
            case 'output':
                this.handleOutput(message);
                break;
            case 'session_created':
                this.sessionId = message.session_id;
                this.terminal.writeln(`📝 Session ID: ${this.sessionId}`);
                break;
            case 'command_started':
                this.terminal.writeln(`🚀 Executing: ${message.command}`);
                break;
            case 'command_completed':
                this.terminal.writeln(`✅ Command completed (exit code: ${message.exit_code})`);
                break;
            case 'error':
                this.terminal.writeln(`❌ Error: ${message.message}`);
                break;
            case 'pong':
                // Handle ping/pong for connection monitoring
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handleOutput(message) {
        const { output_type, data } = message;
        
        // Style output based on type
        if (output_type === 'stderr') {
            // Red text for errors
            this.terminal.write('\x1b[31m' + data + '\x1b[0m');
        } else if (output_type === 'system') {
            // Cyan text for system messages
            this.terminal.write('\x1b[36m' + data + '\x1b[0m');
        } else {
            // Normal output
            this.terminal.write(data);
        }
    }

    sendMessage(message) {
        if (this.isConnected && this.websocket) {
            this.websocket.send(JSON.stringify(message));
        }
    }

    sendInput(data) {
        this.sendMessage({
            type: 'input',
            data: data,
            session_id: this.sessionId
        });
    }

    executeCommand(command) {
        this.sendMessage({
            type: 'command',
            data: command,
            session_id: this.sessionId
        });
    }

    interrupt() {
        this.sendMessage({
            type: 'interrupt',
            session_id: this.sessionId
        });
        this.terminal.writeln('\r\n🛑 Interrupt signal sent');
    }

    clear() {
        this.terminal.clear();
    }

    focus() {
        this.terminal.focus();
    }

    resize() {
        if (this.terminal && this.terminal.fitAddon) {
            this.terminal.fitAddon.fit();
        }
    }

    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        if (this.terminal) {
            this.terminal.dispose();
        }
    }

    // Utility methods for common commands
    runFairdocV1() {
        this.executeCommand('uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload');
    }

    runFairdocV2() {
        this.executeCommand('uv run uvicorn src.app2.main_v2:app --host 0.0.0.0 --port 8000 --reload');
    }

    runTests(testType = 'unit') {
        this.executeCommand(`uv run pytest src/tests/${testType} -v`);
    }

    checkSystemHealth() {
        this.executeCommand('curl -s http://localhost:8000/api/v2/health | jq');
    }
}

// Terminal manager for handling multiple terminals
class TerminalManager {
    constructor() {
        this.terminals = new Map();
        this.activeTerminalId = null;
    }

    createTerminal(containerId) {
        const terminalId = `terminal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const terminal = new DeathNoteTerminal(terminalId, containerId);
        
        this.terminals.set(terminalId, terminal);
        this.activeTerminalId = terminalId;
        
        return terminalId;
    }

    getTerminal(terminalId) {
        return this.terminals.get(terminalId);
    }

    getActiveTerminal() {
        return this.terminals.get(this.activeTerminalId);
    }

    setActiveTerminal(terminalId) {
        if (this.terminals.has(terminalId)) {
            this.activeTerminalId = terminalId;
            this.terminals.get(terminalId).focus();
        }
    }

    destroyTerminal(terminalId) {
        const terminal = this.terminals.get(terminalId);
        if (terminal) {
            terminal.destroy();
            this.terminals.delete(terminalId);
            
            if (this.activeTerminalId === terminalId) {
                this.activeTerminalId = this.terminals.keys().next().value || null;
            }
        }
    }

    destroyAllTerminals() {
        this.terminals.forEach(terminal => terminal.destroy());
        this.terminals.clear();
        this.activeTerminalId = null;
    }

    resizeAllTerminals() {
        this.terminals.forEach(terminal => terminal.resize());
    }
}

// Global function to initialize terminal (called from app.js)
window.initializeTerminal = function(containerId) {
    if (!window.terminalManager) {
        window.terminalManager = new TerminalManager();
    }
    
    return window.terminalManager.createTerminal(containerId);
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Load xterm.js and addons from CDN
    const xtermCss = document.createElement('link');
    xtermCss.rel = 'stylesheet';
    xtermCss.href = 'https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css';
    document.head.appendChild(xtermCss);
    
    // Initialize default terminal if container exists
    const defaultContainer = document.getElementById('default-terminal');
    if (defaultContainer) {
        window.initializeTerminal('default-terminal');
    }
});