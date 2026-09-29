"""POST /v1/tools/web_search — one search."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .._generated.models import WebSearchRequest, WebSearchResponse
from ..search import SearchFailure
from ..service import SearchService

router = APIRouter(tags=["tools"])


def problem(failure: SearchFailure) -> JSONResponse:
    """A provider's failure as problem+json, with its machine name as `code`.

    `code` is an RFC 7807 extension member; the gateway reads it to tell
    the model *which kind* of failure it was (a refused key is not a busy
    provider), and `detail` is the sentence a person reads.
    """
    headers = {}
    if failure.retry_after is not None:
        headers["Retry-After"] = str(int(failure.retry_after + 0.999))
    return JSONResponse(
        status_code=failure.status,
        media_type="application/problem+json",
        headers=headers,
        content={
            "type": f"https://github.com/eugene-plexus/tool-driver#{failure.code}",
            "title": failure.code.replace("_", " "),
            "status": failure.status,
            "detail": failure.detail,
            "component": "tool-driver",
            "code": failure.code,
            **(
                {"retryAfterSeconds": failure.retry_after}
                if failure.retry_after is not None
                else {}
            ),
        },
    )


@router.post("/v1/tools/web_search", response_model=WebSearchResponse)
async def web_search(request: Request, body: WebSearchRequest) -> WebSearchResponse | JSONResponse:
    service: SearchService | None = getattr(request.app.state, "search", None)
    if service is None:
        return problem(SearchFailure(503, "not_configured", "This search account is in safe mode."))
    try:
        return await service.web_search(body)
    except SearchFailure as failure:
        return problem(failure)
