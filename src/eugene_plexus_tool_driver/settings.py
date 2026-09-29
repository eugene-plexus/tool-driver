"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable via
`PATCH /v1/config`. These control bootstrap only: where the config file
is, which interface to bind, and the credentials the agent hands a child.

The prefix is `EUGENE_PLEXUS_TOOL_DRIVER_`, not the inference-driver's
`EUGENE_PLEXUS_DRIVER_`: the agent passes a child only the variables that
carry its own prefix, so two kinds sharing one would read each other's.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_TOOL_DRIVER_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where the runtime config is persisted. PATCH /v1/config writes here."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. The agent sets 0.0.0.0 when the node advertises a LAN address."""

    safe_mode: bool = False
    """Boot from built-in defaults, ignoring the config file (the agent's
    safe-mode contract). PATCH /v1/config still writes the file."""

    trust_bundle_file: str | None = None
    """The trust bundle the agent keeps beside `node.yaml`, reloaded when it changes."""

    trust_authority: str | None = None
    """The public key that bundle must be signed by (base64url Ed25519)."""

    auth_recipient: str | None = None
    """This machine as a token's audience names it: `node:<name>`."""

    service_token: str | None = None
    """This process's own token, addressed to this machine alone."""

    agent_url: str = "http://127.0.0.1:8079"
    """This machine's agent. Unused today; threaded by the agent to every child."""

    master_key: str | None = None
    """Base64 32-byte secretbox key that seals `apiKey` at rest."""


def load_settings() -> Settings:
    return Settings()
