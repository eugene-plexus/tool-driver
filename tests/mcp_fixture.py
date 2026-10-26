"""A remote MCP server for the tests: Streamable HTTP, scripted per test.

Answers `initialize` with a session id, `notifications/initialized` with
202, `tools/list` one page at a time, `tools/call` by echoing, and records
every request's headers so a test can see what reached the server. A test
changes `answer_as` (`json` or `sse`), `tools`, `page_size`, and the hooks
that make it misbehave: an error that echoes the caller's token, a 401, a
redirect, an oversized or slow answer.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


def tool(name: str, description: str = "A tool.") -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    }


class FixtureServer:
    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = [tool("echo", "Says the text back."), tool("shout")]
        self.page_size = 1
        self.answer_as = "json"
        self.seen: list[tuple[str, str, dict[str, str], Any]] = []
        self.ended: list[str] = []
        #: Called with the JSON-RPC message; a Response it returns is sent instead.
        self.hook: Callable[[Request, dict[str, Any]], Response | None] | None = None
        self.slow = 0.0
        self.app = Starlette(
            routes=[
                Route("/mcp", self.handle, methods=["POST", "DELETE"]),
                Route("/moved", self.handle, methods=["POST", "DELETE"]),
            ]
        )

    def methods(self) -> list[str]:
        return [m for (_, m, _, _) in self.seen]

    async def handle(self, request: Request) -> Response:
        if request.method == "DELETE":
            self.ended.append(request.headers.get("mcp-session-id", ""))
            return Response(status_code=200)
        message = json.loads(await request.body())
        self.seen.append(
            (request.url.path, message.get("method", ""), dict(request.headers), message)
        )
        if self.hook is not None:
            replaced = self.hook(request, message)
            if replaced is not None:
                return replaced
        if self.slow:
            await asyncio.sleep(self.slow)
        method = message.get("method")
        if "id" not in message:
            return Response(status_code=202)
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
            return self._answer(message, result, {"Mcp-Session-Id": "session-1"})
        if request.headers.get("mcp-session-id") != "session-1":
            return JSONResponse({"error": "no session"}, status_code=400)
        if method == "tools/list":
            start = int((message.get("params") or {}).get("cursor") or 0)
            page = self.tools[start : start + self.page_size]
            result = {"tools": page}
            if start + self.page_size < len(self.tools):
                result["nextCursor"] = str(start + self.page_size)
            return self._answer(message, result)
        if method == "tools/call":
            params = message.get("params") or {}
            text = str((params.get("arguments") or {}).get("text", ""))
            if params.get("name") == "shout":
                text = text.upper()
            if params.get("name") == "fail":
                return self._answer(
                    message, {"content": [{"type": "text", "text": "it broke"}], "isError": True}
                )
            return self._answer(
                message,
                {
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                    ]
                },
            )
        return self._answer(message, None, error={"code": -32601, "message": "no such method"})

    def _answer(
        self,
        message: dict[str, Any],
        result: dict[str, Any] | None,
        headers: dict[str, str] | None = None,
        *,
        error: dict[str, Any] | None = None,
    ) -> Response:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if error is not None:
            body["error"] = error
        else:
            body["result"] = result
        if self.answer_as == "sse":
            note = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"x": 1}}

            async def events() -> Any:
                yield f"event: message\ndata: {json.dumps(note)}\n\n".encode()
                yield b": a comment\n\n"
                yield f"event: message\ndata: {json.dumps(body)}\n\n".encode()

            return StreamingResponse(events(), media_type="text/event-stream", headers=headers)
        return JSONResponse(body, headers=headers)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        port: int = s.getsockname()[1]
        return port


@contextmanager
def serving(server: FixtureServer, host: str = "127.0.0.1") -> Iterator[int]:
    """The fixture on a real socket, on an OS-chosen port, for as long as the block runs."""
    port = free_port(host)
    config = uvicorn.Config(server.app, host=host, port=port, log_level="warning", lifespan="off")
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    for _ in range(200):
        if running.started:
            break
        threading.Event().wait(0.02)
    try:
        yield port
    finally:
        running.should_exit = True
        thread.join(timeout=5)
