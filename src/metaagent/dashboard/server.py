"""WebSocket server for real-time dashboard updates.

This module provides a FastAPI server with WebSocket support for
streaming refinement progress to connected dashboard clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .events import Event, EventEmitter, get_emitter

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Meta-Agent Dashboard")

# Store connected WebSocket clients
connected_clients: list[WebSocket] = []


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total clients: {len(self.active_connections)}")

        # Send current state to new client
        emitter = get_emitter()
        await websocket.send_json({
            "type": "state_sync",
            "data": emitter.state.to_dict(),
            "timestamp": "",
        })

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, event: Event) -> None:
        """Broadcast an event to all connected clients."""
        if not self.active_connections:
            return

        message = event.to_json()
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


async def event_handler(event: Event) -> None:
    """Handle events from the emitter and broadcast to clients."""
    await manager.broadcast(event)


@app.on_event("startup")
async def startup_event() -> None:
    """Set up event handling on server startup."""
    emitter = get_emitter()
    emitter.subscribe_async(event_handler)
    # Start async dispatch in background
    asyncio.create_task(emitter.run_async_dispatch())
    logger.info("Dashboard server started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up on server shutdown."""
    emitter = get_emitter()
    emitter.stop_async_dispatch()
    emitter.unsubscribe(event_handler)
    logger.info("Dashboard server stopped")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """Serve the dashboard HTML page."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/state")
async def get_state() -> dict:
    """Get current session state."""
    emitter = get_emitter()
    return emitter.state.to_dict()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await websocket.receive_text()
            # Could handle commands from client here if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Inline dashboard HTML for simplicity (no build step needed)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Agent Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        background: '#0a0a0a',
                        foreground: '#fafafa',
                        card: '#171717',
                        border: '#262626',
                        accent: '#8b5cf6',
                    }
                }
            }
        }
    </script>
    <style>
        body { background: #0a0a0a; color: #fafafa; font-family: 'Inter', system-ui, sans-serif; }
        .log-container { height: 300px; overflow-y: auto; }
        .log-container::-webkit-scrollbar { width: 6px; }
        .log-container::-webkit-scrollbar-track { background: #171717; }
        .log-container::-webkit-scrollbar-thumb { background: #404040; border-radius: 3px; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        .layer-bar { transition: all 0.3s ease; }
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-6xl mx-auto space-y-6">
        <!-- Header -->
        <div class="flex items-center justify-between">
            <div>
                <h1 class="text-2xl font-bold">Meta-Agent Dashboard</h1>
                <p id="feature-request" class="text-gray-400 text-sm mt-1">Waiting for session...</p>
            </div>
            <div class="flex items-center gap-3">
                <div id="connection-status" class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-red-500"></span>
                    <span class="text-sm text-gray-400">Disconnected</span>
                </div>
            </div>
        </div>

        <!-- Progress Overview -->
        <div class="grid grid-cols-4 gap-4">
            <div class="bg-card border border-border rounded-lg p-4">
                <div class="text-sm text-gray-400">Phase</div>
                <div class="text-2xl font-bold mt-1">
                    <span id="phase-current">-</span>
                    <span class="text-gray-500">/</span>
                    <span id="phase-total" class="text-gray-500">5</span>
                </div>
                <div id="phase-name" class="text-sm text-accent mt-1">-</div>
            </div>
            <div class="bg-card border border-border rounded-lg p-4">
                <div class="text-sm text-gray-400">Iteration</div>
                <div class="text-2xl font-bold mt-1">
                    <span id="iteration-current">0</span>
                    <span class="text-gray-500">/</span>
                    <span id="iteration-max" class="text-gray-500">10</span>
                </div>
                <div id="layer-name" class="text-sm text-accent mt-1">scaffold</div>
            </div>
            <div class="bg-card border border-border rounded-lg p-4">
                <div class="text-sm text-gray-400">Tasks</div>
                <div class="text-2xl font-bold mt-1">
                    <span id="tasks-completed">0</span>
                    <span class="text-gray-500">/</span>
                    <span id="tasks-total" class="text-gray-500">0</span>
                </div>
                <div id="current-task" class="text-sm text-gray-400 mt-1 truncate">-</div>
            </div>
            <div class="bg-card border border-border rounded-lg p-4">
                <div class="text-sm text-gray-400">Files Modified</div>
                <div class="text-2xl font-bold mt-1" id="files-count">0</div>
                <div class="text-sm text-gray-400 mt-1">
                    Errors: <span id="errors-count" class="text-red-400">0</span>
                </div>
            </div>
        </div>

        <!-- Layer Progress -->
        <div class="bg-card border border-border rounded-lg p-4">
            <div class="text-sm text-gray-400 mb-3">Layer Progress</div>
            <div class="flex gap-2">
                <div id="layer-scaffold" class="layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-gray-800 text-gray-400">
                    Scaffold
                </div>
                <div id="layer-core" class="layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-gray-800 text-gray-400">
                    Core
                </div>
                <div id="layer-integration" class="layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-gray-800 text-gray-400">
                    Integration
                </div>
                <div id="layer-polish" class="layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-gray-800 text-gray-400">
                    Polish
                </div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-2 gap-6">
            <!-- Live Log -->
            <div class="bg-card border border-border rounded-lg p-4">
                <div class="flex items-center justify-between mb-3">
                    <div class="text-sm text-gray-400">Live Log</div>
                    <button onclick="clearLogs()" class="text-xs text-gray-500 hover:text-gray-300">Clear</button>
                </div>
                <div id="log-container" class="log-container font-mono text-xs space-y-1">
                    <div class="text-gray-500">Waiting for events...</div>
                </div>
            </div>

            <!-- Task List -->
            <div class="bg-card border border-border rounded-lg p-4">
                <div class="text-sm text-gray-400 mb-3">Tasks</div>
                <div id="task-list" class="space-y-2 max-h-[300px] overflow-y-auto">
                    <div class="text-gray-500 text-sm">No tasks yet</div>
                </div>
            </div>
        </div>

        <!-- Files Modified -->
        <div class="bg-card border border-border rounded-lg p-4">
            <div class="text-sm text-gray-400 mb-3">Files Modified</div>
            <div id="files-list" class="flex flex-wrap gap-2">
                <div class="text-gray-500 text-sm">No files modified yet</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let state = {};
        let logs = [];

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                updateConnectionStatus(true);
                addLog('Connected to server', 'system');
            };

            ws.onclose = () => {
                updateConnectionStatus(false);
                addLog('Disconnected from server', 'error');
                // Reconnect after 3 seconds
                setTimeout(connect, 3000);
            };

            ws.onerror = (error) => {
                addLog('WebSocket error', 'error');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };
        }

        function updateConnectionStatus(connected) {
            const el = document.getElementById('connection-status');
            if (connected) {
                el.innerHTML = '<span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span><span class="text-sm text-gray-400">Connected</span>';
            } else {
                el.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span><span class="text-sm text-gray-400">Disconnected</span>';
            }
        }

        function handleEvent(event) {
            if (event.type === 'state_sync') {
                state = event.data;
                updateUI();
                return;
            }

            // Update state based on event
            switch (event.type) {
                case 'session_start':
                    state.feature_request = event.data.feature_request || '';
                    state.max_iterations = event.data.max_iterations || 10;
                    state.started_at = event.timestamp;
                    addLog(`Session started: ${state.feature_request}`, 'system');
                    break;
                case 'phase_start':
                    state.phase = event.data.phase || '';
                    state.phase_number = event.data.phase_number || 0;
                    state.total_phases = event.data.total_phases || 5;
                    addLog(`Phase ${state.phase_number}: ${state.phase}`, 'info');
                    break;
                case 'iteration_start':
                    state.iteration = event.data.iteration || 0;
                    addLog(`Iteration ${state.iteration} started`, 'info');
                    break;
                case 'layer_update':
                    state.current_layer = event.data.current_layer || 1;
                    state.layer_name = event.data.layer_name || 'scaffold';
                    state.layers_complete = event.data.layers_complete || {};
                    addLog(`Layer: ${state.layer_name}`, 'info');
                    break;
                case 'task_list':
                    state.tasks = event.data.tasks || [];
                    state.total_tasks = state.tasks.length;
                    state.tasks_completed = 0;
                    addLog(`Generated ${state.total_tasks} tasks`, 'info');
                    break;
                case 'task_start':
                    state.current_task = event.data.title || '';
                    addLog(`Starting: ${state.current_task}`, 'info');
                    break;
                case 'task_complete':
                    state.tasks_completed = (state.tasks_completed || 0) + 1;
                    addLog(`Completed: ${event.data.title || 'task'}`, 'success');
                    break;
                case 'file_created':
                case 'file_modified':
                    if (!state.files_modified) state.files_modified = [];
                    const file = event.data.file || '';
                    if (file && !state.files_modified.includes(file)) {
                        state.files_modified.push(file);
                    }
                    addLog(`${event.type === 'file_created' ? 'Created' : 'Modified'}: ${file}`, 'file');
                    break;
                case 'log':
                    addLog(event.data.message || '', event.data.level || 'info');
                    break;
                case 'error':
                    if (!state.errors) state.errors = [];
                    state.errors.push({ message: event.data.message, timestamp: event.timestamp });
                    addLog(`Error: ${event.data.message}`, 'error');
                    break;
                case 'ollama_start':
                    addLog('Ollama selecting relevant files...', 'info');
                    break;
                case 'ollama_complete':
                    addLog(`Ollama selected ${event.data.file_count || 0} files`, 'success');
                    break;
                case 'perplexity_start':
                    addLog('Perplexity analyzing...', 'info');
                    break;
                case 'perplexity_complete':
                    addLog('Perplexity analysis complete', 'success');
                    break;
                case 'session_end':
                    addLog('Session completed', 'system');
                    break;
            }

            updateUI();
        }

        function addLog(message, level = 'info') {
            const colors = {
                info: 'text-gray-300',
                success: 'text-green-400',
                error: 'text-red-400',
                system: 'text-accent',
                file: 'text-blue-400',
            };
            const time = new Date().toLocaleTimeString();
            logs.push({ time, message, level });
            if (logs.length > 100) logs.shift();

            const container = document.getElementById('log-container');
            container.innerHTML = logs.map(l =>
                `<div class="${colors[l.level] || 'text-gray-300'}"><span class="text-gray-500">${l.time}</span> ${escapeHtml(l.message)}</div>`
            ).join('');
            container.scrollTop = container.scrollHeight;
        }

        function clearLogs() {
            logs = [];
            document.getElementById('log-container').innerHTML = '<div class="text-gray-500">Logs cleared</div>';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function updateUI() {
            // Feature request
            document.getElementById('feature-request').textContent = state.feature_request || 'Waiting for session...';

            // Phase
            document.getElementById('phase-current').textContent = state.phase_number || '-';
            document.getElementById('phase-total').textContent = state.total_phases || 5;
            document.getElementById('phase-name').textContent = state.phase || '-';

            // Iteration
            document.getElementById('iteration-current').textContent = state.iteration || 0;
            document.getElementById('iteration-max').textContent = state.max_iterations || 10;
            document.getElementById('layer-name').textContent = state.layer_name || 'scaffold';

            // Tasks
            document.getElementById('tasks-completed').textContent = state.tasks_completed || 0;
            document.getElementById('tasks-total').textContent = state.total_tasks || 0;
            document.getElementById('current-task').textContent = state.current_task || '-';

            // Files & Errors
            document.getElementById('files-count').textContent = (state.files_modified || []).length;
            document.getElementById('errors-count').textContent = (state.errors || []).length;

            // Layer progress
            const layers = ['scaffold', 'core', 'integration', 'polish'];
            const layerComplete = state.layers_complete || {};
            const currentLayer = state.layer_name || 'scaffold';

            layers.forEach(layer => {
                const el = document.getElementById(`layer-${layer}`);
                if (layerComplete[layer]) {
                    el.className = 'layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-green-600 text-white';
                } else if (layer === currentLayer) {
                    el.className = 'layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-accent text-white animate-pulse';
                } else {
                    el.className = 'layer-bar flex-1 h-8 rounded flex items-center justify-center text-sm font-medium bg-gray-800 text-gray-400';
                }
            });

            // Task list
            const taskList = document.getElementById('task-list');
            if (state.tasks && state.tasks.length > 0) {
                taskList.innerHTML = state.tasks.map((task, i) => {
                    const isComplete = i < (state.tasks_completed || 0);
                    const isCurrent = i === (state.tasks_completed || 0);
                    const statusClass = isComplete ? 'text-green-400' : (isCurrent ? 'text-accent' : 'text-gray-400');
                    const icon = isComplete ? '✓' : (isCurrent ? '▶' : '○');
                    return `<div class="flex items-start gap-2 text-sm ${statusClass}">
                        <span class="mt-0.5">${icon}</span>
                        <span class="flex-1">${escapeHtml(task.title || task)}</span>
                    </div>`;
                }).join('');
            } else {
                taskList.innerHTML = '<div class="text-gray-500 text-sm">No tasks yet</div>';
            }

            // Files list
            const filesList = document.getElementById('files-list');
            if (state.files_modified && state.files_modified.length > 0) {
                filesList.innerHTML = state.files_modified.map(f =>
                    `<span class="px-2 py-1 bg-gray-800 rounded text-xs text-blue-400">${escapeHtml(f.split('/').pop())}</span>`
                ).join('');
            } else {
                filesList.innerHTML = '<div class="text-gray-500 text-sm">No files modified yet</div>';
            }
        }

        // Start connection
        connect();
    </script>
</body>
</html>
"""


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
