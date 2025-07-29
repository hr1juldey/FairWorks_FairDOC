# Death Note Terminal - Development Architecture

## 🏗️ System Architecture Overview

Building a **Death Note themed terminal webapp** with FastAPI backend (port 8999) for managing Fairdoc AI V1/V2 servers with advanced capabilities.

### 📁 Flat File Structure (12 files max)

```
death_note_terminal/
├── main.py                    # FastAPI server & API endpoints
├── server_manager.py          # Process & port management logic
├── test_runner.py             # Test discovery & pytest execution  
├── terminal_handler.py        # WebSocket terminal management
├── ollama_client.py           # DeepSeek-R1 integration
├── index.html                 # Main Death Note themed interface
├── styles.css                 # Gothic dark/light theme styles
├── terminal.js                # xterm.js terminal implementation
├── app.js                     # Frontend application logic
├── theme.js                   # Dark/light mode switching
├── config.py                  # Configuration management
└── requirements.txt           # Python dependencies
```

## 🎯 Core Capabilities

### 1. **Server Management**
- **Port Options**: 8000 (combined), 8001 (V1), 8002 (V2)
- **Process Control**: Start/stop/restart servers
- **Health Monitoring**: Real-time server status

### 2. **Web Terminal Interface**
- **Multiple Sessions**: Separate terminals for each running app
- **xterm.js Integration**: Full terminal emulation
- **WebSocket Communication**: Real-time terminal I/O

### 3. **Test Management**
- **Folder Scanning**: Auto-discover unit/integration/e2e tests
- **Pytest Parameters**: Configurable test execution options
- **Live Output**: Stream test results to web terminal

### 4. **Ollama Integration**
- **DeepSeek-R1 8B**: Local model on default port (11434)
- **Log Analysis**: AI-powered problem identification
- **Summarization**: Intelligent terminal output interpretation

### 5. **Death Note Aesthetic**
- **Gothic Theme**: Dark paper textures, red accents
- **Material UI**: Responsive desktop-focused design
- **Theme Toggle**: Seamless dark/light mode switching
- **Personal Diary**: Notebook-style interface design

## 🔧 Technical Stack

### Backend (Python)
- **FastAPI**: Modern async web framework
- **WebSockets**: Real-time terminal communication
- **subprocess**: Process management for servers
- **pytest**: Test discovery and execution
- **requests**: Ollama API integration

### Frontend (HTML5/CSS3/JS)
- **xterm.js**: Terminal emulation library
- **Material Design**: Component-based UI
- **CSS Variables**: Dynamic theme switching
- **WebSocket API**: Real-time communication

## 🎨 Death Note Theme Design

### Color Palette
```css
/* Dark Mode (Primary) */
--death-primary: #1a1a1a    /* Deep black */
--death-secondary: #8B0000  /* Dark red */
--death-accent: #FFD700     /* Gold highlights */
--death-paper: #2C2416      /* Aged paper */

/* Light Mode (Inverse) */
--light-primary: #F5F5DC    /* Beige paper */
--light-secondary: #8B0000  /* Consistent red */
--light-accent: #DAA520     /* Darker gold */
--light-paper: #FFFACD     /* Light cream */
```

### Typography
- **Headers**: Gothic serif fonts (Crimson Text)
- **Body**: Monospace for terminals (Fira Code)
- **Accents**: Dramatic serif (Playfair Display)

### UI Elements
- **Paper Texture**: CSS background patterns
- **Torn Edges**: Border styling effects
- **Shadow Effects**: Depth and drama
- **Gothic Icons**: Death Note inspired symbols

## 🚀 Implementation Plan

### Phase 1: Backend Core (Files 1-5)
1. **main.py**: FastAPI server with all API endpoints
2. **server_manager.py**: Port management and process control
3. **test_runner.py**: Test discovery and execution logic
4. **terminal_handler.py**: WebSocket terminal sessions
5. **ollama_client.py**: AI integration for log analysis

### Phase 2: Frontend Interface (Files 6-10)
6. **index.html**: Death Note themed HTML structure
7. **styles.css**: Complete gothic styling system
8. **terminal.js**: xterm.js terminal implementation
9. **app.js**: Main application logic and API calls
10. **theme.js**: Dark/light mode management

### Phase 3: Configuration (Files 11-12)
11. **config.py**: Centralized configuration management
12. **requirements.txt**: Python dependency specification

## 📊 API Endpoints Design

```
GET  /                    # Serve main HTML interface
GET  /api/servers/status  # Get all server status
POST /api/servers/start   # Start server (port config)
POST /api/servers/stop    # Stop specific server
GET  /api/tests/discover  # Scan and list available tests
POST /api/tests/run       # Execute selected tests
WS   /ws/terminal/{id}    # WebSocket terminal sessions
POST /api/ollama/analyze  # Send logs to DeepSeek-R1
GET  /api/health         # System health check
```

## 🎭 User Experience Flow

1. **Landing**: Death Note themed dashboard
2. **Server Control**: Choose port configuration and start servers
3. **Terminal Access**: Multiple web terminals for each app
4. **Test Execution**: Select and run tests with live output
5. **AI Analysis**: Send logs to Ollama for intelligent insights
6. **Theme Toggle**: Switch between dark/light Death Note modes

Each file will be under 200 lines, focusing on single responsibility principles for maintainable, production-grade code.