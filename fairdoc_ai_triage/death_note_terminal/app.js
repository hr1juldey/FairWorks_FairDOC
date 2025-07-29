/**
 * Death Note Control Panel - Client-Side Application Logic
 * 
 * Handles UI interactions, theme switching, server management,
 * test execution, and WebSocket terminal connections.
 * 
 * Single responsibility: Frontend application logic and theme management
 * File: app.js
 */

class DeathNoteApp {
    constructor() {
        this.servers = {};
        this.terminals = {};
        this.activeTests = [];
        this.theme = localStorage.getItem('death-note-theme') || 'dark';
        this.apiBaseUrl = `${window.location.protocol}//${window.location.host}`;
        
        this.init();
    }

    async init() {
        this.setupTheme();
        this.setupEventListeners();
        await this.loadServerStatus();
        await this.loadTests();
        this.setupWebSockets();
        
        // Show welcome message
        this.showNotification('🖤 Death Note Control Panel Loaded', 'success');
    }

    // Theme Management
    setupTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.textContent = this.theme === 'dark' ? '☀️' : '🌙';
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('death-note-theme', this.theme);
        document.documentElement.setAttribute('data-theme', this.theme);
        
        // Update toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.textContent = this.theme === 'dark' ? '☀️' : '🌙';
        }
        
        // Animate theme transition
        document.body.style.transition = 'all 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
        
        this.showNotification(`Switched to ${this.theme} mode`, 'info');
    }

    // Event Listeners Setup
    setupEventListeners() {
        // Theme toggle
        document.getElementById('theme-toggle')?.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Server management
        document.querySelectorAll('.server-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const server = e.target.dataset.server;
                const port = document.getElementById(`${server}-port`)?.value || 8000;
                this.manageServer(server, port);
            });
        });

        // Test execution
        document.getElementById('run-selected-tests')?.addEventListener('click', () => {
            this.runSelectedTests();
        });

        // Terminal controls
        document.getElementById('new-terminal')?.addEventListener('click', () => {
            this.createNewTerminal();
        });

        // Ollama integration
        document.getElementById('analyze-logs')?.addEventListener('click', () => {
            this.analyzeWithOllama();
        });

        // Refresh buttons
        document.getElementById('refresh-servers')?.addEventListener('click', () => {
            this.loadServerStatus();
        });

        document.getElementById('refresh-tests')?.addEventListener('click', () => {
            this.loadTests();
        });
    }

    // Server Management
    async loadServerStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/servers/status`);
            const data = await response.json();
            this.servers = data.servers;
            this.updateServerUI();
        } catch (error) {
            console.error('Failed to load server status:', error);
            this.showNotification('Failed to load server status', 'error');
        }
    }

    async manageServer(serverType, port = 8000) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/servers/manage`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    server_type: serverType,
                    port: parseInt(port)
                })
            });

            const result = await response.json();
            if (result.success) {
                this.showNotification(result.message, 'success');
                await this.loadServerStatus();
            } else {
                this.showNotification(result.message, 'error');
            }
        } catch (error) {
            console.error('Server management error:', error);
            this.showNotification('Server operation failed', 'error');
        }
    }

    updateServerUI() {
        Object.entries(this.servers).forEach(([serverName, info]) => {
            const statusElement = document.getElementById(`${serverName}-status`);
            if (statusElement) {
                statusElement.textContent = info.running ? '🟢 Running' : '🔴 Stopped';
                statusElement.className = `server-status ${info.running ? 'running' : 'stopped'}`;
            }

            const portElement = document.getElementById(`${serverName}-port`);
            if (portElement && info.running) {
                portElement.value = info.port;
            }
        });
    }

    // Test Management
    async loadTests() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tests/discover`);
            const data = await response.json();
            this.updateTestsUI(data.tests);
        } catch (error) {
            console.error('Failed to load tests:', error);
            this.showNotification('Failed to load tests', 'error');
        }
    }

    updateTestsUI(tests) {
        const container = document.getElementById('tests-container');
        if (!container) return;

        container.innerHTML = '';

        ['unit', 'integration', 'e2e'].forEach(category => {
            if (tests[category] && tests[category].length > 0) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'test-category';
                categoryDiv.innerHTML = `
                    <h3 class="category-title">${category.toUpperCase()} Tests (${tests[category].length})</h3>
                    <div class="test-list">
                        ${tests[category].map(test => `
                            <label class="test-item">
                                <input type="checkbox" value="${test}" class="test-checkbox">
                                <span class="test-name">${test.split('/').pop()}</span>
                                <small class="test-path">${test}</small>
                            </label>
                        `).join('')}
                    </div>
                `;
                container.appendChild(categoryDiv);
            }
        });
    }

    async runSelectedTests() {
        const selectedTests = Array.from(document.querySelectorAll('.test-checkbox:checked'))
            .map(cb => cb.value);

        if (selectedTests.length === 0) {
            this.showNotification('Please select tests to run', 'warning');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tests/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    test_files: selectedTests,
                    live_mode: document.getElementById('live-mode')?.checked || false
                })
            });

            const result = await response.json();
            if (result.success) {
                this.showNotification(`Started ${selectedTests.length} tests`, 'success');
                this.monitorTestExecution(result.data.run_id);
            } else {
                this.showNotification('Failed to start tests', 'error');
            }
        } catch (error) {
            console.error('Test execution error:', error);
            this.showNotification('Test execution failed', 'error');
        }
    }

    async monitorTestExecution(runId) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/tests/output/${runId}`);
                const data = await response.json();
                
                this.updateTestOutput(data.output);
                
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(interval);
                    this.showNotification(`Tests ${data.status}`, 
                        data.status === 'completed' ? 'success' : 'error');
                }
            } catch (error) {
                console.error('Failed to monitor tests:', error);
                clearInterval(interval);
            }
        }, 1000);
    }

    updateTestOutput(output) {
        const outputElement = document.getElementById('test-output');
        if (outputElement) {
            outputElement.textContent = output;
            outputElement.scrollTop = outputElement.scrollHeight;
        }
    }

    // WebSocket and Terminal Management
    setupWebSockets() {
        // Will be extended by terminal.js
    }

    createNewTerminal() {
        // Create new terminal instance
        const terminalId = `terminal-${Date.now()}`;
        const terminalContainer = document.getElementById('terminals-container');
        
        const terminalDiv = document.createElement('div');
        terminalDiv.id = terminalId;
        terminalDiv.className = 'terminal-instance';
        terminalDiv.innerHTML = `
            <div class="terminal-header">
                <span class="terminal-title">💀 Terminal ${terminalId.split('-')[1]}</span>
                <button class="terminal-close" onclick="app.closeTerminal('${terminalId}')">×</button>
            </div>
            <div class="terminal-content" id="${terminalId}-content"></div>
        `;
        
        terminalContainer.appendChild(terminalDiv);
        
        // Initialize xterm.js terminal (handled by terminal.js)
        if (window.initializeTerminal) {
            window.initializeTerminal(terminalId);
        }
    }

    closeTerminal(terminalId) {
        const terminalElement = document.getElementById(terminalId);
        if (terminalElement) {
            terminalElement.remove();
            delete this.terminals[terminalId];
        }
    }

    // Ollama Integration
    async analyzeWithOllama() {
        const output = document.getElementById('test-output')?.textContent || '';
        if (!output.trim()) {
            this.showNotification('No output to analyze', 'warning');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/ollama/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: output,
                    analysis_type: 'test_results'
                })
            });

            const result = await response.json();
            if (result.success) {
                this.showAnalysisResults(result.data.analysis);
            } else {
                this.showNotification('Analysis failed', 'error');
            }
        } catch (error) {
            console.error('Ollama analysis error:', error);
            this.showNotification('Analysis service unavailable', 'error');
        }
    }

    showAnalysisResults(analysis) {
        const modal = document.createElement('div');
        modal.className = 'analysis-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>🧠 DeepSeek Analysis</h2>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <pre class="analysis-text">${analysis}</pre>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal functionality
        modal.querySelector('.modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }

    // Utility Functions
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeathNoteApp();
});