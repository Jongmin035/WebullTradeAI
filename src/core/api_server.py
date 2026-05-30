"""
Dashboard API server — FastAPI + uvicorn with WebSocket push.

Runs alongside the trader on EC2. Dashboard buttons POST to REST endpoints;
the WebSocket endpoint pushes live stats.json updates to connected clients
so the dashboard never needs to poll.

Usage:
    python api_server.py            # default port 5000
    python api_server.py 8080       # custom port

REST endpoints (POST):
    /api/add-funds          {"amount": 500}
    /api/remove-funds       {"amount": 300}
    /api/take-control       {"symbol": "AAPL", "sell": false}
    /api/release-control    {"symbol": "AAPL"}
    /api/emergency-stop     {}
    /api/clear-emergency    {}

WebSocket:
    ws://localhost:5000/ws  — pushes stats.json content on every file change
"""

import contextlib
import io
import os
import sys
import asyncio
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _d in (_src, os.path.join(_src, "core"), os.path.join(_src, "pipeline")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import controls
from controls import ControlError
from dashboard_logger import refresh_controls, STATS_FILE

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

_connections: List[WebSocket] = []


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_broadcast_loop())


async def _broadcast_loop():
    """Push stats.json to all connected WebSocket clients whenever the file changes."""
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(STATS_FILE)
            if mtime != last_mtime:
                last_mtime = mtime
                with open(STATS_FILE) as f:
                    content = f.read()
                dead = []
                for ws in _connections:
                    try:
                        await ws.send_text(content)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    _connections.remove(ws)
        except Exception:
            pass
        await asyncio.sleep(1.0)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connections.append(websocket)
    # Send current snapshot immediately on connect
    try:
        with open(STATS_FILE) as f:
            await websocket.send_text(f.read())
    except Exception:
        pass
    try:
        while True:
            await websocket.receive_text()   # keep connection alive
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _connections:
            _connections.remove(websocket)


# --- Request models ---

class AmountBody(BaseModel):
    amount: float

class SymbolBody(BaseModel):
    symbol: str
    sell: bool = False


# --- Shared dispatch helper ---

def _run(fn):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
        try:
            refresh_controls()
        except Exception:
            pass
        return {"ok": True, "message": buf.getvalue().strip()}
    except ControlError as e:
        return JSONResponse(status_code=400, content={"ok": False, "message": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})


# --- Endpoints ---

@app.post("/api/add-funds")
def add_funds(body: AmountBody):
    return _run(lambda: controls.add_funds(body.amount))

@app.post("/api/remove-funds")
def remove_funds(body: AmountBody):
    return _run(lambda: controls.remove_funds(body.amount))

@app.post("/api/take-control")
def take_control(body: SymbolBody):
    return _run(lambda: controls.take_control(body.symbol, sell=body.sell))

@app.post("/api/release-control")
def release_control(body: SymbolBody):
    return _run(lambda: controls.release_control(body.symbol))

@app.post("/api/emergency-stop")
def emergency_stop():
    return _run(controls.emergency_stop)

@app.post("/api/clear-emergency")
def clear_emergency():
    return _run(controls.clear_emergency)


def run(port=5000):
    """Start the API server (blocking). Intended to be called in a daemon thread."""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"API server running on port {port}")
    run(port)
