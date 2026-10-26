"""POST /v1/tools/mcp/list_tools and /v1/tools/mcp/call -- remote MCP (row 59b)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .._generated.models import (
    McpCallRequest,
    McpCallResponse,
    McpListToolsRequest,
    McpListToolsResponse,
)
from .._generated.models import McpTool as McpToolModel
from ..mcp_remote import McpFailure, RemoteMcp
from .search import problem

router = APIRouter(tags=["tools"])


def _remote(request: Request) -> RemoteMcp | None:
    found: RemoteMcp | None = getattr(request.app.state, "mcp", None)
    return found


def _safe_mode() -> JSONResponse:
    return problem(McpFailure(503, "not_configured", "This tool driver is in safe mode."))


@router.post("/v1/tools/mcp/list_tools", response_model=McpListToolsResponse)
async def list_tools(
    request: Request, body: McpListToolsRequest
) -> McpListToolsResponse | JSONResponse:
    remote = _remote(request)
    if remote is None:
        return _safe_mode()
    started = time.perf_counter()
    try:
        tools = await remote.list_tools(
            body.serverUrl,
            reach=body.reach.value,
            headers=body.headers,
            allowed=[str(a.root) for a in body.allowedTools]
            if body.allowedTools is not None
            else None,
        )
    except McpFailure as failure:
        return problem(failure)
    return McpListToolsResponse(
        tools=[
            McpToolModel(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
                annotations=t.annotations,
            )
            for t in tools
        ],
        latencyMs=round((time.perf_counter() - started) * 1000, 1),
    )


@router.post("/v1/tools/mcp/call", response_model=McpCallResponse)
async def call_tool(request: Request, body: McpCallRequest) -> McpCallResponse | JSONResponse:
    remote = _remote(request)
    if remote is None:
        return _safe_mode()
    started = time.perf_counter()
    try:
        called = await remote.call(
            body.serverUrl,
            reach=body.reach.value,
            headers=body.headers,
            name=body.name,
            arguments=dict(body.arguments or {}),
        )
    except McpFailure as failure:
        return problem(failure)
    return McpCallResponse(
        output=called.output,
        error=called.error,
        latencyMs=round((time.perf_counter() - started) * 1000, 1),
    )
