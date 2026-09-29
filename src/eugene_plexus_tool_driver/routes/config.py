"""Config protocol routes: GET, PATCH, schema, test."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from .._generated.models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    WebSearchRequest,
)
from ..config import ConfigStore, as_schema
from ..search import SearchFailure
from ..service import PROBE_QUERY, SearchService

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema() -> ConfigSchema:
    return as_schema()


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(request: Request, body: ConfigTestRequest | None = None) -> ConfigTestResult:
    """One real search with the saved config and the overrides, nothing saved.

    The only way an operator learns *before a model asks* that a SearXNG
    instance has JSON output off, or that a Brave key was refused.
    """
    started = time.perf_counter()
    service: SearchService | None = getattr(request.app.state, "search", None)
    if service is None:
        return ConfigTestResult(
            ok=False, component="tool-driver", latencyMs=0, error="running in safe mode"
        )
    overrides: dict[str, Any] = {}
    if body and body.overrides:
        overrides = {
            k: v
            for k, v in body.overrides.model_dump(exclude_none=True).items()
            if v != "<redacted>"
        }
    try:
        answer = await service.web_search(
            WebSearchRequest(query=PROBE_QUERY, maxResults=3), overrides=overrides
        )
    except SearchFailure as failure:
        return ConfigTestResult(
            ok=False,
            component="tool-driver",
            latencyMs=int((time.perf_counter() - started) * 1000),
            error=failure.detail,
        )
    first = answer.results[0] if answer.results else None
    return ConfigTestResult(
        ok=True,
        component="tool-driver",
        latencyMs=int((time.perf_counter() - started) * 1000),
        summary=(
            f"{answer.provider} answered {len(answer.results)} result"
            f"{'' if len(answer.results) == 1 else 's'} for {PROBE_QUERY!r}."
        ),
        sampleOutput=f"{first.title} — {first.url}" if first is not None else None,
    )
