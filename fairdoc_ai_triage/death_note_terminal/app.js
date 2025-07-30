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
        this.websocket = null;
        this.isInitialized = false;
        
        // Bind methods to preserve 'this' context
        this.toggleTheme = this.toggleTheme.bind(this);
        this.handleServerAction = this.handleServerAction.bind(this);
        this.stopAllServers = this.stopAllServers.bind(this);
        this.refreshServers = this.refreshServers.bind(this);
        this.runSelectedTests = this.runSelectedTests.bind(this);
        this.createNewTerminal = this.createNewTerminal.bind(this);
        this.analyzeWithOllama = this.analyzeWithOllama.bind(this);
    }

    async init() {
        console.log('🎭 Initializing Death Note Terminal...');
        
        try {
            this.setupTheme();
            await this.loadServerStatus();
            await this.loadTests();
            this.setupEventListeners();
            this.setupWebSocket();
            this.isInitialized = true;
            
            this.showNotification('🖤 Death Note Control Panel Loaded', 'success');
            console.log('✅ Death Note Terminal initialized successfully');
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showNotification('Failed to initialize Death Note Terminal', 'error');
        }
    }

    // Theme Management
    setupTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.textContent = this.theme === 'dark' ? '☀️' : '🌙';
            themeToggle.title = `Switch to ${this.theme === 'dark' ? 'light' : 'dark'} mode`;
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('death-note-theme', this.theme);
        document.documentElement.setAttribute('data-theme', this.theme);
        
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.textContent = this.theme === 'dark' ? '☀️' : '🌙';
            themeToggle.title = `Switch to ${this.theme === 'dark' ? 'light' : 'dark'} mode`;
        }
        
        this.showNotification(`Switched to ${this.theme} mode`, 'info');
        
        // Dispatch custom event for terminal theme updates
        document.dispatchEvent(new CustomEvent('theme-changed', { detail: this.theme }));
    }

    // Event Listeners Setup - FIXED with proper element detection
    setupEventListeners() {
        console.log('🔗 Setting up event listeners...');

        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', this.toggleTheme);
            console.log('✅ Theme toggle listener added');
        } else {
            console.warn('⚠️ Theme toggle button not found');
        }

        // Server management buttons - Use event delegation
        document.addEventListener('click', (e) => {
            // Handle server action buttons
            if (e.target.classList.contains('server-btn') || e.target.dataset.action) {
                e.preventDefault();
                const server = e.target.dataset.server;
                const action = e.target.dataset.action;
                
                if (server && action) {
                    console.log(`🎮 Server action: ${action} ${server}`);
                    this.handleServerAction(server, action);
                }
            }
            
            // Handle other buttons
            if (e.target.id === 'stop-all-servers') {
                e.preventDefault();
                console.log('🛑 Stop all servers clicked');
                this.stopAllServers();
            }
            
            if (e.target.id === 'refresh-servers') {
                e.preventDefault();
                console.log('🔄 Refresh servers clicked');
                this.refreshServers();
            }
            
            if (e.target.id === 'run-selected-tests') {
                e.preventDefault();
                console.log('🧪 Run tests clicked');
                this.runSelectedTests();
            }
            
            if (e.target.id === 'new-terminal') {
                e.preventDefault();
                console.log('💻 New terminal clicked');
                this.createNewTerminal();
            }
            
            if (e.target.id === 'analyze-logs') {
                e.preventDefault();
                console.log('🤖 Analyze logs clicked');
                this.analyzeWithOllama();
            }
        });

        console.log('✅ Event listeners set up complete');
    }

    // Server Management - FIXED with proper error handling
    async loadServerStatus() {
        console.log('📊 Loading server status...');
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/servers/status`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.servers = data;
            this.updateServerUI();
            console.log('✅ Server status loaded:', data);
        } catch (error) {
            console.error('❌ Failed to load server status:', error);
            this.showNotification('Failed to load server status', 'error');
        }
    }

    async handleServerAction(serverType, action) {
        console.log(`🎮 Handling server action: ${action} ${serverType}`);
        
        try {
            let response;
            let requestBody = { server_type: serverType };
            
            // Get custom port if specified
            const portInput = document.getElementById(`${serverType}-port`);
            if (portInput && portInput.value) {
                requestBody.port = parseInt(portInput.value);
            }
            
            if (action === 'start') {
                response = await fetch(`${this.apiBaseUrl}/api/servers/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
            } else if (action === 'stop') {
                response = await fetch(`${this.apiBaseUrl}/api/servers/stop/${serverType}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Server ${action} failed`);
            }

            const result = await response.json();
            console.log(`✅ Server ${action} result:`, result);
            
            this.showNotification(`Server ${serverType} ${action}ed successfully`, 'success');
            await this.loadServerStatus(); // Refresh status
            
        } catch (error) {
            console.error(`❌ Server ${action} error:`, error);
            this.showNotification(`Failed to ${action} server: ${error.message}`, 'error');
        }
    }

    async stopAllServers() {
        console.log('🛑 Stopping all servers...');
        const runningServers = Object.entries(this.servers).filter(([_, info]) => info.running);
        
        for (const [serverName, _] of runningServers) {
            await this.handleServerAction(serverName, 'stop');
        }
    }

    async refreshServers() {
        console.log('🔄 Refreshing server status...');
        await this.loadServerStatus();
    }

    updateServerUI() {
        console.log('🖥️ Updating server UI with data:', this.servers);
        
        Object.entries(this.servers).forEach(([serverName, info]) => {
            console.log(`Updating ${serverName}:`, info);
            
            // Update status indicator with more robust selection
            const statusElement = document.getElementById(`${serverName}-status`);
            if (statusElement) {
                const isRunning = info.running === true;
                statusElement.textContent = isRunning ? '🟢 Running' : '🔴 Stopped';
                statusElement.className = `server-status ${isRunning ? 'running' : 'stopped'}`;
                
                // Add port info if available
                if (isRunning && info.port) {
                    statusElement.textContent += ` (Port: ${info.port})`;
                }
            }

            // Update server buttons with explicit state management
            const startBtn = document.querySelector(`[data-server="${serverName}"][data-action="start"]`);
            const stopBtn = document.querySelector(`[data-server="${serverName}"][data-action="stop"]`);
            
            if (startBtn) {
                const isRunning = info.running === true;
                startBtn.disabled = isRunning;
                startBtn.textContent = isRunning ? 'Running...' : 'Start';
                startBtn.className = `btn ${isRunning ? 'btn-secondary' : 'btn-primary'}`;
                console.log(`Start button for ${serverName}: disabled=${isRunning}`);
            }
            
            if (stopBtn) {
                const isRunning = info.running === true;
                stopBtn.disabled = !isRunning;
                stopBtn.textContent = isRunning ? 'Stop' : 'Stopped';
                stopBtn.className = `btn ${isRunning ? 'btn-danger' : 'btn-secondary'}`;
                console.log(`Stop button for ${serverName}: disabled=${!isRunning}`);
            }
        });
    }


    // Test Management - FIXED
    async loadTests() {
        console.log('🧪 Loading tests...');
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tests/discover`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.updateTestsUI(data);
            console.log('✅ Tests loaded:', data);
        } catch (error) {
            console.error('❌ Failed to load tests:', error);
            this.showNotification('Failed to load tests', 'error');
        }
    }

    updateTestsUI(tests) {
        console.log('🖥️ Updating tests UI...');
        const categories = ['unit', 'integration', 'e2e'];
        
        categories.forEach(category => {
            const container = document.getElementById(`${category}-tests`);
            if (!container) {
                console.warn(`⚠️ Test container not found: ${category}-tests`);
                return;
            }

            container.innerHTML = '';
            
            if (tests[category] && tests[category].length > 0) {
                tests[category].forEach(test => {
                    const testItem = document.createElement('div');
                    testItem.className = 'test-item';
                    testItem.innerHTML = `
                        <input type="checkbox" class="test-checkbox" value="${test}" id="${category}-${test}">
                        <label for="${category}-${test}">${test}</label>
                    `;
                    container.appendChild(testItem);
                });
            } else {
                container.innerHTML = '<p class="no-tests">No tests found</p>';
            }
        });
    }

    async runSelectedTests() {
        console.log('🧪 Running selected tests...');
        
        const selectedTests = [];
        const testTypes = [];
        
        // Collect selected tests
        document.querySelectorAll('.test-checkbox:checked').forEach(checkbox => {
            const [category, ...testParts] = checkbox.id.split('-');
            const testName = testParts.join('-');
            
            if (!testTypes.includes(category)) {
                testTypes.push(category);
            }
            selectedTests.push(testName);
        });

        if (testTypes.length === 0) {
            this.showNotification('Please select at least one test', 'warning');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tests/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    test_types: testTypes,
                    specific_tests: selectedTests.length > 0 ? selectedTests : null,
                    pytest_args: document.getElementById('pytest-args')?.value || null
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Test execution failed');
            }

            const result = await response.json();
            console.log('✅ Test execution started:', result);
            
            this.showNotification(`Test execution started: ${result.session_id}`, 'success');
            this.pollTestResults(result.session_id);
        } catch (error) {
            console.error('❌ Test execution error:', error);
            this.showNotification(`Test execution failed: ${error.message}`, 'error');
        }
    }

    async pollTestResults(sessionId) {
        console.log('📊 Polling test results for session:', sessionId);
        
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/tests/sessions/${sessionId}/output`);
                const data = await response.json();
                
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollInterval);
                    this.showTestResults(data);
                }
                
                this.updateTestOutput(data);
            } catch (error) {
                console.error('❌ Error polling test results:', error);
                clearInterval(pollInterval);
            }
        }, 1000);
    }

    updateTestOutput(data) {
        const outputElement = document.getElementById('test-output');
        if (outputElement && data.output) {
            const output = data.output.map(item => item.line).join('\n');
            outputElement.textContent = output;
            outputElement.scrollTop = outputElement.scrollHeight;
        }
        
        // Update test results if completed
        if (data.results) {
            const resultsDiv = document.getElementById('test-results') || this.createTestResultsDiv();
            resultsDiv.innerHTML = `
                <div class="test-summary">
                    <h4>Test Results</h4>
                    <p>Total: ${data.results.total || 0}</p>
                    <p>Passed: ${data.results.passed || 0}</p>
                    <p>Failed: ${data.results.failed || 0}</p>
                    <p>Duration: ${data.results.duration || 0}s</p>
                </div>
            `;
        }
    }

    createTestResultsDiv() {
        const resultsDiv = document.createElement('div');
        resultsDiv.id = 'test-results';
        resultsDiv.className = 'test-results';
        const outputElement = document.getElementById('test-output');
        if (outputElement && outputElement.parentNode) {
            outputElement.parentNode.appendChild(resultsDiv);
        }
        return resultsDiv;
    }


    showTestResults(data) {
        const results = data.results || {};
        const message = `Tests completed: ${results.passed || 0} passed, ${results.failed || 0} failed`;
        const type = results.failed > 0 ? 'warning' : 'success';
        this.showNotification(message, type);
    }

    // WebSocket Setup - FIXED
    setupWebSocket() {
        console.log('🔌 Setting up WebSocket connection...');
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        try {
            // Connect to status WebSocket for real-time updates
            this.websocket = new WebSocket(`${protocol}//${window.location.host}/ws/status/${clientId}`);
            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.showNotification('Terminal connection established', 'success');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ WebSocket message error:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.showNotification('Terminal connection lost', 'warning');
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.showNotification('Terminal connection error', 'error');
            };
        } catch (error) {
            console.error('❌ Failed to setup WebSocket:', error);
        }
    }

    handleWebSocketMessage(data) {
        console.log('📨 WebSocket message:', data);
        
        if (data.type === 'status_update') {
            // Update server status in real-time
            this.servers = data.servers;
            this.updateServerUI();
            this.updateStatusBar(data);
        } else if (data.type === 'server_logs') {
            this.displayServerLogs(data);
        } else if (data.type === 'test_completed') {
            this.showTestResults(data.result);
            // Send to Ollama for analysis
            this.sendTestResultsToOllama(data.result);
        } else if (data.type === 'ollama_analysis') {
            this.showFormattedOllamaAnalysis(data.analysis);
        }
    }

    displayServerLogs(data) {
        const logContainer = document.getElementById('server-logs') || this.createServerLogsContainer();
        const serverSection = logContainer.querySelector(`[data-server="${data.server}"]`) || this.createServerLogSection(data.server);
        
        data.logs.forEach(log => {
            const logLine = document.createElement('div');
            logLine.className = 'log-line';
            logLine.innerHTML = `<span class="timestamp">${new Date(log.timestamp * 1000).toLocaleTimeString()}</span> ${log.line}`;
            serverSection.appendChild(logLine);
        });
        
        // Auto-scroll to bottom
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    updateStatusBar(data) {
        const statusBar = document.getElementById('status-bar') || this.createStatusBar();
        const runningServers = Object.values(data.servers).filter(s => s.running).length;
        const totalServers = Object.keys(data.servers).length;
        
        statusBar.innerHTML = `
            <div class="status-item">🖥️ Servers: ${runningServers}/${totalServers} running</div>
            <div class="status-item">🕒 ${new Date().toLocaleTimeString()}</div>
            <div class="status-item">🔄 Live Updates Active</div>
        `;
    }

    showFormattedOllamaAnalysis(analysis) {
        const analysisContainer = document.getElementById('ollama-analysis') || this.createOllamaAnalysisDiv();
        
        // Format analysis with sections
        const formattedAnalysis = this.formatOllamaAnalysis(analysis);
        analysisContainer.innerHTML = `
            <div class="ollama-result">
                <h4>🤖 AI Analysis Results</h4>
                ${formattedAnalysis}
            </div>
        `;
    }

    formatOllamaAnalysis(analysis) {
        // Split analysis into logical sections
        const sections = analysis.split(/\n\s*\n/);
        let formatted = '';
        
        sections.forEach((section, index) => {
            if (section.trim()) {
                const isHeading = section.includes(':') && section.length < 100;
                if (isHeading) {
                    formatted += `<h5 class="analysis-heading">${section.trim()}</h5>`;
                } else {
                    formatted += `<div class="analysis-content">${section.trim()}</div>`;
                }
            }
        });
        
        return formatted || `<div class="analysis-content">${analysis}</div>`;
    }


    

    
    // showOllamaAnalysis(analysis) {
    //     const analysisDiv = document.getElementById('ollama-analysis') || this.createOllamaAnalysisDiv();
    //     analysisDiv.innerHTML = `
    //         <div class="ollama-result">
    //             <h4>🤖 AI Analysis</h4>
    //             <pre>${analysis}</pre>
    //         </div>
    //     `;
    // }

    createOllamaAnalysisDiv() {
        const analysisDiv = document.createElement('div');
        analysisDiv.id = 'ollama-analysis';
        analysisDiv.className = 'ollama-analysis';
        const testOutput = document.getElementById('test-output');
        if (testOutput && testOutput.parentNode) {
            testOutput.parentNode.appendChild(analysisDiv);
        }
        return analysisDiv;
    }


    // Terminal Management - SIMPLIFIED
    createNewTerminal() {
        console.log('💻 Creating new terminal...');
        
        const terminalContainer = document.getElementById('terminal-container');
        if (!terminalContainer) {
            this.showNotification('Terminal container not found', 'error');
            return;
        }

        // Create simple terminal UI
        const terminalDiv = document.createElement('div');
        terminalDiv.className = 'terminal-session';
        terminalDiv.innerHTML = `
            <div class="terminal-header">
                <span>💀 Death Note Terminal</span>
                <button class="close-terminal" type="button">×</button>
            </div>
            <div class="terminal-output" style="height: 200px; overflow-y: auto; background: #000; color: #0f0; padding: 10px; font-family: monospace;"></div>
            <div class="terminal-input-area" style="display: flex; background: #000; padding: 5px;">
                <span class="prompt" style="color: #f00;">💀 > </span>
                <input type="text" class="terminal-input" placeholder="Enter command..." style="flex: 1; background: transparent; border: none; color: #0f0; font-family: monospace;">
            </div>
        `;

        terminalContainer.appendChild(terminalDiv);
        
        // Add event listeners
        const closeBtn = terminalDiv.querySelector('.close-terminal');
        const input = terminalDiv.querySelector('.terminal-input');
        const output = terminalDiv.querySelector('.terminal-output');
        
        closeBtn.addEventListener('click', () => {
            terminalDiv.remove();
            this.showNotification('Terminal closed', 'info');
        });

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const command = input.value.trim();
                if (command) {
                    this.executeTerminalCommand(command, output);
                    input.value = '';
                }
            }
        });

        // Focus the input
        input.focus();
        this.showNotification('New terminal created', 'success');
    }

    executeTerminalCommand(command, outputElement) {
        console.log('⚡ Executing terminal command:', command);
        
        // Add command to output
        outputElement.innerHTML += `<div style="color: #f00;">💀 > ${command}</div>`;
        
        // Simple command responses
        let response = '';
        if (command.toLowerCase().includes('help')) {
            response = 'Available commands: help, clear, status, servers, tests, exit';
        } else if (command.toLowerCase() === 'clear') {
            outputElement.innerHTML = '';
            return;
        } else if (command.toLowerCase() === 'status') {
            const runningCount = Object.values(this.servers).filter(s => s.running).length;
            response = `Death Note Terminal Status: ${runningCount} servers running`;
        } else if (command.toLowerCase() === 'servers') {
            response = Object.entries(this.servers)
                .map(([name, info]) => `${name}: ${info.running ? '🟢' : '🔴'}`)
                .join('\n');
        } else if (command.toLowerCase() === 'exit') {
            response = 'Use the × button to close the terminal';
        } else {
            response = `Command executed: ${command}`;
        }
        
        if (response) {
            outputElement.innerHTML += `<div style="color: #0f0;">${response}</div>`;
        }
        
        // Scroll to bottom
        outputElement.scrollTop = outputElement.scrollHeight;
    }

    // Ollama Integration - FIXED
    async analyzeWithOllama() {
        console.log('🤖 Analyzing with Ollama...');
        
        const contentElement = document.getElementById('analysis-content');
        const resultsElement = document.getElementById('analysis-results');
        
        if (!contentElement) {
            this.showNotification('Analysis content field not found', 'error');
            return;
        }
        
        const content = contentElement.value.trim();
        if (!content) {
            this.showNotification('Please enter content to analyze', 'warning');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: content,
                    analysis_type: document.getElementById('analysis-type')?.value || 'general'
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const result = await response.json();
            console.log('✅ Analysis completed:', result);
            
            if (resultsElement) {
                resultsElement.textContent = result.analysis;
            }
            
            this.showNotification('Analysis completed', 'success');
        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        }
    }

    // Notification System - ENHANCED
    showNotification(message, type = 'info') {
        console.log(`📢 Notification [${type}]: ${message}`);
        
        // Remove existing notifications
        document.querySelectorAll('.death-note-notification').forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `death-note-notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            z-index: 10000;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
        `;
        
        // Set background color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545', 
            warning: '#ffc107',
            info: '#17a2b8'
        };
        
        notification.style.backgroundColor = colors[type] || colors.info;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 4000);
    }
}

// CSS for notifications
const notificationCSS = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;

// Add CSS to document
const style = document.createElement('style');
style.textContent = notificationCSS;
document.head.appendChild(style);

// Initialize the application - FIXED with proper error handling
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🎭 DOM Content Loaded - Starting Death Note Terminal App');
    
    try {
        window.deathNoteApp = new DeathNoteApp();
        await window.deathNoteApp.init();
        console.log('✅ Death Note Terminal App initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize Death Note Terminal App:', error);
        
        // Show error notification even if app failed to initialize
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed; top: 20px; right: 20px; 
            background: #dc3545; color: white; padding: 15px; 
            border-radius: 8px; z-index: 10000;
        `;
        errorDiv.textContent = 'Failed to initialize Death Note Terminal';
        document.body.appendChild(errorDiv);
    }
});

// Export for debugging
window.DeathNoteApp = DeathNoteApp;
