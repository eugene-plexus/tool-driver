"""GET /healthz — liveness, and whether the provider answered its last search."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import Health, Status
from ..service import SearchService

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    safe_mode = bool(getattr(request.app.state, "safe_mode", False))
    service: SearchService | None = getattr(request.app.state, "search", None)
    config_error = getattr(request.app.state, "config_error", None)
    if safe_mode or service is None:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="tool-driver",
            safeMode=safe_mode,
            details={"error": config_error or "running in safe mode"},
        )
    if config_error:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="tool-driver",
            details={"error": config_error},
        )
    provider, account = service.resolve()
    missing = provider.missing(account)
    if missing is not None:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="tool-driver",
            details={"error": f"not set up: {missing}", "provider": provider.key},
        )
    last = service.last
    details: dict[str, object] = {"provider": provider.key}
    if last is not None:
        details["lastSearchOk"] = last.ok
        details["lastSearchAt"] = last.at
        if not last.ok:
            details["error"] = last.error
            details["code"] = last.code
    return Health(
        status=Status.ok if last is None or last.ok else Status.degraded,
        version=__version__,
        component="tool-driver",
        details=details,
    )
