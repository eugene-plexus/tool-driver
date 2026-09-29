"""Runtime configuration: the shared config protocol, for one search account.

`GET /v1/config/schema`, `GET /v1/config`, `PATCH /v1/config` -- the trio
every component implements, so the console renders this account's page
with no code of its own. `apiKey` is sealed at rest with the master key
the agent hands this process, exactly as an inference-driver's is.

**Nothing here needs a restart but the log level.** A search reads the
account's settings when it runs, so a corrected SearXNG address or a new
Brave token takes effect on the next search. Restarting a process to
change an address was a cost with no reason behind it.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from . import _private_files, security
from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigFieldShowWhen,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)
from .providers import PROVIDERS

log = logging.getLogger(__name__)

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "account": "Search account",
    "results": "Results",
    "network": "Network",
    "logging": "Logging",
}


def _only_for(*providers: str) -> ConfigFieldShowWhen:
    return ConfigFieldShowWhen(key="provider", equals=list(providers))


def _build_fields() -> list[ConfigField]:
    keys = list(PROVIDERS)
    return [
        ConfigField(
            key="provider",
            label="Search provider",
            description=(
                "Who runs the searches. SearXNG is free and runs on your own "
                "machine or server; you give it the address. Brave Search is a "
                "paid service with its own index; you give it a key. Either way "
                "the words searched for go to the internet."
            ),
            category="account",
            valueType=ConfigValueType.enum,
            default="searxng",
            enumValues=keys,
            enumLabels=[PROVIDERS[k].label for k in keys],
            required=True,
        ),
        ConfigField(
            key="baseUrl",
            label="Address",
            description=(
                "SearXNG: the address of your instance, such as "
                "http://192.168.1.20:8888. Its settings must allow JSON output "
                "(`search.formats` includes `json`); without it every search is "
                "refused. Brave: leave empty to use Brave's own address."
            ),
            category="account",
            valueType=ConfigValueType.url,
        ),
        ConfigField(
            key="apiKey",
            label="API key",
            description=(
                "Your Brave Search API subscription token, from "
                "api-dashboard.search.brave.com. Stored encrypted."
            ),
            category="account",
            valueType=ConfigValueType.secret,
            sensitive=True,
            showWhen=_only_for("brave"),
        ),
        ConfigField(
            key="maxResults",
            label="Results per search",
            description=(
                "How many results a model is given when it does not ask for a "
                "number. More results mean a longer prompt for the model to read."
            ),
            category="results",
            valueType=ConfigValueType.integer,
            default=5,
            minimum=1,
            maximum=20,
        ),
        ConfigField(
            key="safeSearch",
            label="Safe search",
            description="How strictly the provider filters adult content.",
            category="results",
            valueType=ConfigValueType.enum,
            default="moderate",
            enumValues=["off", "moderate", "strict"],
            enumLabels=["Off", "Moderate", "Strict"],
        ),
        ConfigField(
            key="language",
            label="Language",
            description=(
                "A language code such as `en` or `de` to prefer results in. "
                "Leave empty to let the provider decide."
            ),
            category="results",
            valueType=ConfigValueType.string,
            pattern=r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})?$",
        ),
        ConfigField(
            key="timeoutSeconds",
            label="Search timeout",
            description=(
                "How long one search may take before it is reported to the model "
                "as failed. A model waits on this, so it is short."
            ),
            category="network",
            valueType=ConfigValueType.duration,
            default=15,
            minimum=2,
            maximum=120,
        ),
        ConfigField(
            key="probeMinutes",
            label="Check every (minutes)",
            description=(
                "How often a free SearXNG instance is asked one search so its "
                "health is known before a model needs it. 0 turns the check off. "
                "A paid provider is never checked on a timer, because every "
                "check would be a search you pay for."
            ),
            category="network",
            valueType=ConfigValueType.integer,
            default=10,
            minimum=0,
            maximum=1440,
            showWhen=_only_for("searxng"),
        ),
        ConfigField(
            key="logLevel",
            label="Log level",
            description="How chatty this process's log is.",
            category="logging",
            valueType=ConfigValueType.enum,
            default="INFO",
            enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
            requiresRestart=True,
        ),
    ]


FIELDS: list[ConfigField] = _build_fields()
_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema() -> ConfigSchema:
    return ConfigSchema(component="tool-driver", fields=list(FIELDS), categories=CATEGORY_LABELS)


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """None if valid, otherwise why not."""
    if value is None:
        return None
    vt = field.valueType
    if vt in (ConfigValueType.string, ConfigValueType.url):
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if vt == ConfigValueType.url and value and not re.match(r"^https?://[^\s/]+", value):
            return "must be an http:// or https:// address"
        if field.pattern is not None and value and re.search(field.pattern, value) is None:
            return f"value does not match pattern {field.pattern!r}"
        return None
    if vt == ConfigValueType.secret:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value == REDACTED:
            return "refusing to write the literal redacted value back"
        return None
    if vt in (ConfigValueType.integer, ConfigValueType.number, ConfigValueType.duration):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if vt == ConfigValueType.integer and not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None
    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value not in (field.enumValues or []):
            return f"must be one of {field.enumValues}"
        return None
    return f"unsupported valueType: {vt}"


class ConfigStore:
    """File-backed config state, sealing `sensitive` fields when a master key exists.

    The same shape as the inference-driver's store, including its one
    rule about secrets: an envelope on disk that cannot be opened (no key
    yet, a different install's key) reads as unset, so the account says
    "no key" rather than presenting ciphertext to Brave.
    """

    def __init__(self, path: Path, *, master_key: bytes | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        self._pending_restart: set[str] = set()
        self._master_key = master_key

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for key, value in raw.items():
                    if key in _FIELDS_BY_KEY:
                        merged[key] = self._opened(key, value)
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()

    def _opened(self, key: str, value: Any) -> Any:
        if not security.is_envelope(value):
            return value
        if self._master_key is None:
            log.warning(
                "config field %r is encrypted on disk but no master key is available; "
                "treating it as unset until the agent is unlocked",
                key,
            )
            return None
        try:
            return security.open_envelope(security.Envelope.from_dict(value), self._master_key)
        except ValueError as e:
            log.warning("config field %r failed to decrypt (%s); treating it as unset", key, e)
            return None

    def as_document(self) -> ConfigDocument:
        with self._lock:
            out: dict[str, Any] = {}
            for key, value in self._values.items():
                field = _FIELDS_BY_KEY.get(key)
                out[key] = REDACTED if field and field.sensitive and value else value
            return ConfigDocument.model_validate(out)

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        patch: dict[str, Any] = request.model_dump()
        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue
                err = _validate_value(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue
                self._values[key] = (
                    field.default if new_value is None and field.default is not None else new_value
                )
                applied.append(key)
                if field.requiresRestart:
                    self._pending_restart.add(key)
            if applied:
                self._write_locked()
            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=bool(self._pending_restart),
                pendingRestart=sorted(self._pending_restart),
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        on_disk: dict[str, Any] = {}
        for key, value in self._values.items():
            field = _FIELDS_BY_KEY.get(key)
            if (
                field is not None
                and field.sensitive
                and isinstance(value, str)
                and value
                and self._master_key is not None
            ):
                on_disk[key] = security.seal(value, self._master_key).to_dict()
            else:
                on_disk[key] = value
        # 0600 and replaced, never rewritten in place: a key on disk, and a
        # truncate-then-write killed halfway is an account that forgot it.
        _private_files.write_private_text(
            self._path, yaml.safe_dump(on_disk, sort_keys=True, default_flow_style=False)
        )
