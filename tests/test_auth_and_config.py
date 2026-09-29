"""Who may search, who may configure, and a Brave key sealed at rest."""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

import httpx
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_tool_driver.app import create_app
from eugene_plexus_tool_driver.settings import Settings

from .conftest import FakeInstall
from .test_web_search import SEARXNG, Upstream


def _app(tmp_path: Path, install: FakeInstall, master: str | None = None):
    config = tmp_path / "search.yaml"
    config.write_text(
        yaml.safe_dump(
            {"provider": "searxng", "baseUrl": "http://10.1.1.1:8888", "probeMinutes": 0}
        ),
        encoding="utf-8",
    )
    app = create_app(Settings(config_file=config))
    app.state.auth_state = install.auth_state(master_key_b64=master)
    lan = Upstream(lambda request: httpx.Response(200, json=SEARXNG))
    app.state.search_transports = (lan.transport(), lan.transport())
    return app, config


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_the_gateway_may_search_and_a_client_key_may_not(tmp_path, install) -> None:
    app, _ = _app(tmp_path, install)
    with TestClient(app) as client:
        body = {"query": "x"}
        assert client.post("/v1/tools/web_search", json=body).status_code == 401
        assert (
            client.post(
                "/v1/tools/web_search", json=body, headers=bearer(install.service("gateway"))
            ).status_code
            == 200
        )
        # A client key is for the gateway's front door, never a component.
        refused = client.post(
            "/v1/tools/web_search", json=body, headers=bearer(install.client_key())
        )
        assert refused.status_code == 401
        # Another machine's agent is not the gateway.
        foreign = client.post(
            "/v1/tools/web_search", json=body, headers=bearer(install.foreign_service("agent"))
        )
        assert foreign.status_code == 401
        # Health needs no token.
        assert client.get("/healthz").status_code == 200


def test_config_is_the_operators_alone(tmp_path, install) -> None:
    app, _ = _app(tmp_path, install)
    with TestClient(app) as client:
        service = bearer(install.service("gateway"))
        assert client.get("/v1/config", headers=service).status_code == 401
        assert (
            client.patch("/v1/config", json={"maxResults": 3}, headers=service).status_code == 401
        )
        operator = bearer(install.session())
        assert client.get("/v1/config", headers=operator).status_code == 200
        assert (
            client.get("/v1/config/schema", headers=operator).json()["component"] == "tool-driver"
        )


def test_a_brave_key_is_sealed_on_disk_and_redacted_on_read(tmp_path, install) -> None:
    master = base64.b64encode(secrets.token_bytes(32)).decode()
    app, config = _app(tmp_path, install, master)
    with TestClient(app) as client:
        operator = bearer(install.session())
        result = client.patch(
            "/v1/config",
            json={"provider": "brave", "apiKey": "BSA-plaintext-key"},
            headers=operator,
        ).json()
        assert result["applied"] == ["provider", "apiKey"]
        assert client.get("/v1/config", headers=operator).json()["apiKey"] == "<redacted>"
        refused = client.patch("/v1/config", json={"apiKey": "<redacted>"}, headers=operator).json()
        assert refused["rejected"][0]["key"] == "apiKey"
    on_disk = config.read_text(encoding="utf-8")
    assert "BSA-plaintext-key" not in on_disk
    assert "ciphertext" in on_disk or "nonce" in on_disk
    # And a second boot with the same key opens it.
    app2 = create_app(Settings(config_file=config))
    app2.state.auth_state = install.auth_state(master_key_b64=master)
    with TestClient(app2):
        assert app2.state.config_store.get("apiKey") == "BSA-plaintext-key"


def test_a_config_file_that_will_not_load_leaves_the_process_up_and_says_why(tmp_path) -> None:
    config = tmp_path / "search.yaml"
    config.write_text("- not\n- a mapping\n", encoding="utf-8")
    with TestClient(create_app(Settings(config_file=config))) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert "could not be read" in health["details"]["error"]
        assert client.get("/v1/config").status_code == 200


def test_safe_mode_boots_on_defaults(tmp_path) -> None:
    config = tmp_path / "search.yaml"
    config.write_text(yaml.safe_dump({"baseUrl": "http://10.1.1.1:8888"}), encoding="utf-8")
    with TestClient(create_app(Settings(config_file=config, safe_mode=True))) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded" and health["safeMode"] is True
        assert client.post("/v1/tools/web_search", json={"query": "x"}).status_code == 503
        assert client.get("/v1/info").json()["configured"] is False
