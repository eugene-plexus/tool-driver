"""Entrypoint: `python -m eugene_plexus_tool_driver`."""

from __future__ import annotations

import contextlib
import logging
import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

#: The contract's `servers` default. The agent always passes the port it
#: declared in its topology; this is for a standalone launch.
_DEFAULT_PORT = 8190


def _resolve_port() -> int:
    """`EUGENE_PLEXUS_TOOL_DRIVER_BIND_PORT` (the agent sets it from the
    topology's URL), else the default. The topology owns ports: there is
    no port in this component's own config, so there is one place a port
    can be wrong."""
    env_port = os.environ.get("EUGENE_PLEXUS_TOOL_DRIVER_BIND_PORT")
    return int(env_port) if env_port else _DEFAULT_PORT


def main() -> None:
    settings = load_settings()
    bootstrap_store = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        # The app degrades on a bad file (and says so on /healthz); the
        # log level is the only thing read here, so its default will do.
        with contextlib.suppress(Exception):
            bootstrap_store.load()
    log_level = str(bootstrap_store.get("logLevel") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=_resolve_port(),
        log_level=log_level.lower(),
        # Trust no forwarding header from anyone (review §6.1 #1, R1.2):
        # every caller reaches this component over loopback or the LAN,
        # and uvicorn's default would let any of them set `scope["client"]`.
        forwarded_allow_ips=[],
    )


if __name__ == "__main__":
    main()
