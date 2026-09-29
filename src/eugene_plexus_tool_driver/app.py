"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .auth_state import load_auth_state
from .body_limit import InferenceBodyLimit
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import health as health_routes
from .routes import info as info_routes
from .routes import search as search_routes
from .service import SearchService
from .settings import Settings, load_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Auth first: its master key opens the sealed `apiKey` in the config.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            trust_bundle_file=settings.trust_bundle_file,
            trust_authority=settings.trust_authority,
            auth_recipient=settings.auth_recipient,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )
    store = ConfigStore(settings.config_file, master_key=app.state.auth_state.master_key)
    app.state.config_store = store
    app.state.safe_mode = settings.safe_mode
    app.state.config_error = None
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_TOOL_DRIVER_SAFE_MODE=1); ignoring %s. "
            "Fix config via /v1/config, then restart without the variable.",
            settings.config_file,
        )
    else:
        try:
            store.load()
        except Exception as e:
            # A config file that will not load must not take the process
            # down: its config endpoints are how it gets fixed (degraded
            # mode is required).
            app.state.config_error = f"config file {settings.config_file} could not be read: {e}"
            log.error("%s; running on defaults", app.state.config_error)
    service = SearchService(store, transports=getattr(app.state, "search_transports", None))
    app.state.search = None if settings.safe_mode else service
    if not settings.safe_mode:
        service.start_probe()
    try:
        yield
    finally:
        await service.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(
        title="Eugene Plexus — tool-driver",
        description="Runs the tools the hub runs itself, for one provider account.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default echoes input values, and a query is made from a prompt.
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "title": "Invalid request",
                    "status": 422,
                    "detail": "body: has an invalid, missing or unsupported value.",
                }
            },
        )

    app.include_router(health_routes.router)
    authorized = [Depends(require_authorized)]
    app.include_router(info_routes.router, dependencies=authorized)
    app.include_router(search_routes.router, dependencies=authorized)
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)
    app.add_middleware(InferenceBodyLimit, paths={"/v1/tools/web_search"}, driver=True)
    return app
