"""GET /v1/info — which provider, which tools."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import Egress, ToolBilling, ToolDriverInfo, ToolName
from ..providers import PROVIDERS
from ..service import SearchService

router = APIRouter(tags=["meta"])


@router.get("/v1/info", response_model=ToolDriverInfo)
async def info(request: Request) -> ToolDriverInfo:
    service: SearchService | None = getattr(request.app.state, "search", None)
    key = service.provider_key() if service is not None else "searxng"
    provider = PROVIDERS.get(key)
    configured = service is not None and service.configured()
    return ToolDriverInfo(
        provider=key,
        label=provider.label if provider is not None else key,
        # Nothing is offered until it can run: the gateway reads this list,
        # and an account with no address would turn every search into a
        # failure the model has to be told about.
        tools=[ToolName.web_search] if configured else [],
        egress=Egress.internet,
        configured=configured,
        billing=ToolBilling(provider.billing) if provider is not None else ToolBilling.per_search,
        version=__version__,
    )
