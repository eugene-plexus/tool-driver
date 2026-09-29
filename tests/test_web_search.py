"""One search through each provider, and every failure it can name.

The SearXNG answer is the real one captured on 2026-09-29
(`fixtures/searxng-answer.json`); the Brave answer follows Brave's API
documentation, because there is no Brave key on the development box.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_tool_driver.app import create_app
from eugene_plexus_tool_driver.search import Hit, domain_matches, keep, plain, with_site
from eugene_plexus_tool_driver.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"
SEARXNG = json.loads((FIXTURES / "searxng-answer.json").read_text(encoding="utf-8"))

BRAVE = {
    "type": "search",
    "query": {"original": "eugene plexus", "more_results_available": False},
    "web": {
        "type": "search",
        "results": [
            {
                "title": "Eugene Plexus on <strong>GitHub</strong>",
                "url": "https://github.com/eugene-plexus",
                "description": "A <strong>self-hosted</strong> control plane &amp; more.",
                "page_age": "2026-09-11T00:00:00",
                "age": "2 weeks ago",
                "extra_snippets": ["It installs llama.cpp.", "It routes across backends."],
            },
            {
                "title": "Not this one",
                "url": "https://spam.example.net/page",
                "description": "Unwanted.",
            },
        ],
    },
}

Handler = Callable[[httpx.Request], httpx.Response]


class Upstream:
    """A mock transport that records every request and answers with `handler`."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.seen: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def respond(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            return self.handler(request)

        return httpx.MockTransport(respond)


def _client(tmp_path: Path, config: dict[str, Any], lan: Upstream, public: Upstream) -> TestClient:
    config_file = tmp_path / "search.yaml"
    config_file.write_text(yaml.safe_dump({"probeMinutes": 0, **config}), encoding="utf-8")
    app = create_app(Settings(config_file=config_file))
    app.state.search_transports = (lan.transport(), public.transport())
    return TestClient(app)


@pytest.fixture
def searxng() -> Upstream:
    return Upstream(lambda request: httpx.Response(200, json=SEARXNG))


@pytest.fixture
def nowhere() -> Upstream:
    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"this transport must not be used: {request.url}")

    return Upstream(fail)


@pytest.fixture
def searx_client(tmp_path: Path, searxng: Upstream, nowhere: Upstream) -> Iterator[TestClient]:
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"}, searxng, nowhere
    ) as c:
        yield c


def search(client: TestClient, **body: Any) -> httpx.Response:
    return client.post("/v1/tools/web_search", json={"query": "llama.cpp rpc server", **body})


# --- SearXNG, against its real answer --------------------------------------


def test_searxng_results_are_mapped_from_its_real_answer(searx_client, searxng) -> None:
    response = search(searx_client, maxResults=3)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "searxng"
    assert len(body["results"]) == 3
    first = body["results"][0]
    assert first["url"] == SEARXNG["results"][0]["url"]
    assert first["title"] == SEARXNG["results"][0]["title"]
    assert first["snippet"] and first["snippet"].startswith("Finally, when running llama-cli")
    dated = next(r for r in body["results"] if r["publishedAt"])
    assert dated["publishedAt"] == "2025-06-11T15:42:39"
    sent = searxng.seen[0]
    assert sent.url.path == "/search"
    assert sent.url.params["format"] == "json"
    assert sent.url.params["q"] == "llama.cpp rpc server"


def test_the_accounts_own_count_applies_when_the_request_names_none(
    tmp_path, searxng, nowhere
) -> None:
    with _client(
        tmp_path,
        {"provider": "searxng", "baseUrl": "http://10.0.0.5:8888", "maxResults": 2},
        searxng,
        nowhere,
    ) as client:
        assert len(search(client).json()["results"]) == 2
        # And a request's own number wins over it.
        assert len(search(client, maxResults=4).json()["results"]) == 4


def test_a_lan_searxng_is_dialled_without_the_proxy_and_a_public_one_with_it(
    tmp_path, searxng
) -> None:
    unused = Upstream(lambda request: httpx.Response(500))
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://searx:8888"}, searxng, unused
    ) as c:
        assert search(c).status_code == 200
    assert len(searxng.seen) == 1 and not unused.seen
    public = Upstream(lambda request: httpx.Response(200, json=SEARXNG))
    lan = Upstream(lambda request: httpx.Response(500))
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "https://searx.example.org"}, lan, public
    ) as c:
        assert search(c).status_code == 200
    assert len(public.seen) == 1 and not lan.seen


def test_json_output_switched_off_is_named(tmp_path, nowhere) -> None:
    # Measured: SearXNG answers a format it has not enabled with a 403 and
    # an HTML body, and nothing in it says why.
    forbidden = Upstream(
        lambda request: httpx.Response(403, text="<!doctype html><title>403 Forbidden</title>")
    )
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"}, forbidden, nowhere
    ) as client:
        response = search(client)
        assert response.status_code == 502
        problem = response.json()
        assert problem["code"] == "json_disabled"
        assert "search.formats" in problem["detail"] and "json" in problem["detail"]
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert health["details"]["code"] == "json_disabled"


def test_a_limiter_is_a_429_with_its_retry_after(tmp_path, nowhere) -> None:
    busy = Upstream(lambda request: httpx.Response(429, headers={"Retry-After": "7"}, text="slow"))
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"}, busy, nowhere
    ) as client:
        response = search(client)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "7"
    assert response.json()["code"] == "rate_limited"
    assert "limiter" in response.json()["detail"]


def test_a_slow_or_absent_instance_is_a_timeout_or_unreachable(tmp_path, nowhere) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with _client(
        tmp_path,
        {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888", "timeoutSeconds": 3},
        Upstream(slow),
        nowhere,
    ) as client:
        response = search(client)
    assert response.status_code == 504
    assert response.json()["code"] == "timeout" and "3s" in response.json()["detail"]

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(
        tmp_path,
        {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"},
        Upstream(refused),
        nowhere,
    ) as client:
        response = search(client)
    assert response.status_code == 502
    assert response.json()["code"] == "unreachable"
    assert "192.168.1.20:8888" in response.json()["detail"]


def test_one_allowed_domain_is_a_site_hint_and_every_filter_is_applied_after(
    searx_client, searxng
) -> None:
    body = search(searx_client, allowedDomains=["github.com"], maxResults=10).json()
    assert searxng.seen[-1].url.params["q"] == "llama.cpp rpc server site:github.com"
    assert body["results"] and all("github.com" in r["url"] for r in body["results"])
    body = search(searx_client, allowedDomains=["github.com", "reddit.com"], maxResults=10).json()
    assert "site:" not in searxng.seen[-1].url.params["q"]
    hosts = {httpx.URL(r["url"]).host for r in body["results"]}
    assert hosts <= {"github.com", "www.reddit.com"} and "www.reddit.com" in hosts
    body = search(searx_client, blockedDomains=["reddit.com"], maxResults=10).json()
    assert body["results"] and not any("reddit.com" in r["url"] for r in body["results"])


def test_a_location_searxng_cannot_use_is_named_as_ignored(searx_client) -> None:
    body = search(
        searx_client, userLocation={"type": "approximate", "approximate": {"country": "DE"}}
    ).json()
    assert body["ignored"] == ["userLocation"]
    assert (
        "ignored" not in search(searx_client).json() or not search(searx_client).json()["ignored"]
    )


def test_an_instant_answer_is_carried_as_the_providers_answer(tmp_path, nowhere) -> None:
    answered = dict(SEARXNG, answers=[{"answer": "4", "url": None}, "four"])
    lan = Upstream(lambda request: httpx.Response(200, json=answered))
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"}, lan, nowhere
    ) as c:
        assert search(c).json()["answer"] == "4 four"


# --- Brave, from its documentation -----------------------------------------


@pytest.fixture
def brave() -> Upstream:
    return Upstream(lambda request: httpx.Response(200, json=BRAVE))


def brave_client(tmp_path: Path, public: Upstream, lan: Upstream, **config: Any) -> TestClient:
    return _client(
        tmp_path, {"provider": "brave", "apiKey": "brave-test-key", **config}, lan, public
    )


def test_brave_is_asked_in_its_own_words_and_answered_in_plain_text(
    tmp_path, brave, nowhere
) -> None:
    with brave_client(tmp_path, brave, nowhere, language="de-AT", safeSearch="strict") as client:
        body = search(
            client,
            maxResults=2,
            contextSize="high",
            userLocation={"type": "approximate", "approximate": {"country": "at"}},
        ).json()
    sent = brave.seen[0]
    assert str(sent.url).startswith("https://api.search.brave.com/res/v1/web/search")
    assert sent.headers["X-Subscription-Token"] == "brave-test-key"
    assert sent.url.params["count"] == "2"
    assert sent.url.params["country"] == "AT"
    assert sent.url.params["search_lang"] == "de"
    assert sent.url.params["safesearch"] == "strict"
    assert sent.url.params["extra_snippets"] == "true"
    first = body["results"][0]
    assert first["title"] == "Eugene Plexus on GitHub"
    assert "<strong>" not in first["snippet"] and "&amp;" not in first["snippet"]
    assert "It routes across backends." in first["snippet"]
    assert first["publishedAt"] == "2026-09-11T00:00:00"
    assert not body.get("ignored")


def test_a_refused_brave_key_is_our_credential_not_the_callers_request(tmp_path, nowhere) -> None:
    refused = Upstream(lambda request: httpx.Response(401, json={"type": "ErrorResponse"}))
    with brave_client(tmp_path, refused, nowhere) as client:
        response = search(client)
    assert response.status_code == 502
    assert response.json()["code"] == "upstream_auth_error"
    assert "API key" in response.json()["detail"]


def test_brave_over_its_limit_and_a_bad_parameter(tmp_path, nowhere) -> None:
    busy = Upstream(lambda request: httpx.Response(429, headers={"Retry-After": "2"}))
    with brave_client(tmp_path, busy, nowhere) as client:
        response = search(client)
    assert response.status_code == 429 and response.headers["Retry-After"] == "2"
    bad = Upstream(
        lambda request: httpx.Response(
            422, json={"type": "ErrorResponse", "error": {"detail": "count too large"}}
        )
    )
    with brave_client(tmp_path, bad, nowhere) as client:
        response = search(client)
    assert response.status_code == 502 and "count too large" in response.json()["detail"]


def test_brave_blocked_domains_are_removed_after_the_fact(tmp_path, brave, nowhere) -> None:
    with brave_client(tmp_path, brave, nowhere) as client:
        body = search(client, blockedDomains=["example.net"]).json()
    assert [r["url"] for r in body["results"]] == ["https://github.com/eugene-plexus"]
    assert brave.seen[0].url.params["count"] == "10"


# --- not set up ------------------------------------------------------------


def test_an_account_with_no_address_offers_nothing_and_says_why(tmp_path, nowhere) -> None:
    with _client(tmp_path, {"provider": "searxng"}, nowhere, nowhere) as client:
        info = client.get("/v1/info").json()
        assert info["tools"] == [] and info["configured"] is False
        assert info["egress"] == "internet"
        response = search(client)
        assert response.status_code == 503
        assert response.json()["code"] == "not_configured"
        assert "address" in response.json()["detail"]
        health = client.get("/healthz").json()
        assert health["status"] == "degraded" and "address" in health["details"]["error"]


def test_brave_without_a_key_is_not_set_up(tmp_path, nowhere) -> None:
    with _client(tmp_path, {"provider": "brave"}, nowhere, nowhere) as client:
        assert client.get("/v1/info").json()["configured"] is False
        assert client.get("/v1/info").json()["billing"] == "per_search"
        assert "API key" in search(client).json()["detail"]


def test_a_configured_account_offers_web_search_and_reports_healthy(searx_client) -> None:
    info = searx_client.get("/v1/info").json()
    assert info == {
        "provider": "searxng",
        "label": "SearXNG",
        "tools": ["web_search"],
        "egress": "internet",
        "configured": True,
        "billing": "free",
        "version": info["version"],
    }
    assert searx_client.get("/healthz").json()["status"] == "ok"
    search(searx_client)
    assert searx_client.get("/healthz").json()["details"]["lastSearchOk"] is True


def test_an_address_changed_at_runtime_is_used_by_the_next_search(
    tmp_path, searxng, nowhere
) -> None:
    with _client(tmp_path, {"provider": "searxng"}, searxng, nowhere) as client:
        assert search(client).status_code == 503
        patched = client.patch("/v1/config", json={"baseUrl": "http://192.168.1.30:8888"})
        assert patched.json()["requiresRestart"] is False
        assert search(client).status_code == 200
    assert searxng.seen[0].url.host == "192.168.1.30"


# --- the config Test button -------------------------------------------------


def test_the_test_button_runs_one_real_search_with_unsaved_overrides(
    tmp_path, searxng, nowhere
) -> None:
    with _client(tmp_path, {"provider": "searxng"}, searxng, nowhere) as client:
        result = client.post(
            "/v1/config/test", json={"overrides": {"baseUrl": "http://192.168.1.40:8888"}}
        ).json()
        assert result["ok"] is True and "searxng answered 3 results" in result["summary"]
        # Nothing was saved.
        assert client.get("/v1/config").json().get("baseUrl") is None
    assert searxng.seen[0].url.host == "192.168.1.40"


def test_the_test_button_reports_the_providers_own_failure(tmp_path, nowhere) -> None:
    forbidden = Upstream(lambda request: httpx.Response(403, text="<html>"))
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"}, forbidden, nowhere
    ) as client:
        result = client.post("/v1/config/test").json()
    assert result["ok"] is False and "search.formats" in result["error"]


# --- the shared rules, directly --------------------------------------------


def test_a_domain_matches_itself_and_its_subdomains_and_nothing_else() -> None:
    assert domain_matches("docs.example.com", "example.com")
    assert domain_matches("example.com", "example.com")
    assert not domain_matches("notexample.com", "example.com")
    assert domain_matches("docs.example.com", "https://example.com/docs")
    assert domain_matches("a.example.com", "*.example.com")
    assert not domain_matches("", "example.com")


def test_keep_and_site_and_plain() -> None:
    hits = [Hit(url="https://a.example.com/x", title="a"), Hit(url="https://b.org/y", title="b")]
    assert [h.title for h in keep(hits, ["example.com"], None)] == ["a"]
    assert [h.title for h in keep(hits, None, ["example.com"])] == ["b"]
    assert with_site("q", ["https://example.com/docs"]) == "q site:example.com"
    assert with_site("q site:x.org", ["example.com"]) == "q site:x.org"
    assert plain("<b>x</b> &amp;  y") == "x & y"
