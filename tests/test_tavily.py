"""Tavily as a search account (row 59a).

Tavily's answer and its errors follow Tavily's API reference
(docs.tavily.com, `POST /search`): there is no Tavily key on the
development box, so the fixture plays what the reference documents, as the
Brave fixture does.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from eugene_plexus_tool_driver.config import as_schema
from eugene_plexus_tool_driver.providers import PROVIDERS

from .test_web_search import FakeTime, Upstream, _client, search

KEY = "tvly-dev-test-key-do-not-leak-0123456789"

TAVILY: dict[str, Any] = {
    "query": "llama.cpp rpc server",
    "answer": "llama.cpp ships an RPC server that runs a model's layers on another machine.",
    "images": [],
    "results": [
        {
            "title": "llama.cpp/tools/rpc at master",
            "url": "https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc",
            "content": "The rpc-server lets you run a <b>backend</b> on a remote host.",
            "score": 0.91,
            "raw_content": None,
            "favicon": "https://github.com/favicon.ico",
        },
        {
            "title": "Distributed inference with llama.cpp",
            "url": "https://example.org/blog/llama-rpc",
            "content": "How to split a model across two machines.",
            "score": 0.74,
            "raw_content": None,
            "published_date": "2026-03-02",
        },
        {
            "title": "Unwanted",
            "url": "https://spam.example.net/llama",
            "content": "Not this one.",
            "score": 0.2,
        },
    ],
    "response_time": "1.12",
    "request_id": "123e4567-e89b-12d3-a456-426614174111",
}


@pytest.fixture
def nowhere() -> Upstream:
    """The LAN transport: Tavily is on the internet, so it is never used."""

    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"this transport must not be used: {request.url}")

    return Upstream(fail)


def tavily_error(status: int, words: str, **headers: str) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"error": words}}, headers=headers)


def config(**more: Any) -> dict[str, Any]:
    return {"provider": "tavily", "apiKey": KEY, **more}


def test_results_and_answer_map_to_found(tmp_path, nowhere) -> None:
    tavily = Upstream(lambda request: httpx.Response(200, json=TAVILY))
    with _client(tmp_path, config(), nowhere, tavily) as client:
        response = search(client, maxResults=3, blockedDomains=["spam.example.net"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "tavily"
    # Tavily's answer is the provider's own summary, as SearXNG's answers are.
    assert body["answer"] == TAVILY["answer"]
    assert [r["url"] for r in body["results"]] == [
        TAVILY["results"][0]["url"],
        TAVILY["results"][1]["url"],
    ]
    first, second = body["results"]
    assert first["title"] == "llama.cpp/tools/rpc at master"
    assert first["snippet"] == "The rpc-server lets you run a backend on a remote host."
    assert second["publishedAt"] == "2026-03-02"
    sent = tavily.seen[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.tavily.com/search"
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    payload = json.loads(sent.content)
    # The key rides in the header only, never in the body Tavily might echo.
    assert KEY not in sent.content.decode()
    assert payload["query"] == "llama.cpp rpc server"
    assert payload["search_depth"] == "basic"
    assert payload["max_results"] == 3
    assert payload["include_answer"] is True
    assert payload["exclude_domains"] == ["spam.example.net"]
    assert "include_domains" not in payload


def test_allowed_domains_ride_on_the_search_and_are_filtered_after(tmp_path, nowhere) -> None:
    tavily = Upstream(lambda request: httpx.Response(200, json=TAVILY))
    with _client(tmp_path, config(searchDepth="advanced"), nowhere, tavily) as client:
        response = search(client, allowedDomains=["https://github.com/ggml-org", "example.org"])
    assert response.status_code == 200, response.text
    assert [r["url"] for r in response.json()["results"]] == [
        TAVILY["results"][0]["url"],
        TAVILY["results"][1]["url"],
    ]
    payload = json.loads(tavily.seen[0].content)
    assert payload["include_domains"] == ["github.com", "example.org"]
    assert payload["search_depth"] == "advanced"


def test_a_rate_limit_names_the_retry(tmp_path, nowhere) -> None:
    words = (
        "Your request has been blocked due to excessive requests. Please reduce rate of requests."
    )
    tavily = Upstream(lambda request: tavily_error(429, words, **{"retry-after": "40"}))
    clock = FakeTime()
    with _client(tmp_path, config(), nowhere, tavily, time=clock) as client:
        response = search(client)
    assert response.status_code == 429, response.text
    problem = response.json()
    assert problem["code"] == "rate_limited"
    assert problem["retryAfterSeconds"] == 40
    assert response.headers["retry-after"] == "40"
    assert "Tavily" in problem["detail"] and "HTTP 429" in problem["detail"]
    assert "40 seconds" in problem["detail"]
    # Tavily's own words, so the person reads what Tavily said.
    assert "excessive requests" in problem["detail"]
    # A 40 s wait does not fit in a 15 s search: not retried.
    assert len(tavily.seen) == 1


def test_a_short_rate_limit_is_tried_once_more(tmp_path, nowhere) -> None:
    answers = iter([tavily_error(429, "Too many requests.", **{"retry-after": "2"}), None])

    def respond(request: httpx.Request) -> httpx.Response:
        refused = next(answers)
        return refused if refused is not None else httpx.Response(200, json=TAVILY)

    tavily = Upstream(respond)
    clock = FakeTime()
    with _client(tmp_path, config(), nowhere, tavily, time=clock) as client:
        response = search(client)
    assert response.status_code == 200, response.text
    assert len(tavily.seen) == 2 and clock.slept == [2.0]


def test_a_bad_key_is_named(tmp_path, nowhere) -> None:
    tavily = Upstream(
        lambda request: tavily_error(401, "Unauthorized: missing or invalid API key.")
    )
    with _client(tmp_path, config(), nowhere, tavily) as client:
        response = search(client)
    assert response.status_code == 502, response.text
    problem = response.json()
    assert problem["code"] == "upstream_auth_error"
    assert "Tavily refused this account's API key (HTTP 401)" in problem["detail"]
    assert KEY not in response.text


def test_a_quota_used_up_is_named_and_not_retried(tmp_path, nowhere) -> None:
    for status, words in (
        (
            432,
            "This request exceeds your plan's set usage limit. Please upgrade your plan or "
            "contact support@tavily.com",
        ),
        (
            433,
            "This request exceeds the pay-as-you-go limit. You can increase your limit on "
            "the Tavily dashboard.",
        ),
    ):
        tavily = Upstream(lambda request, s=status, w=words: tavily_error(s, w))
        clock = FakeTime()
        with _client(tmp_path, config(), nowhere, tavily, time=clock) as client:
            response = search(client)
        assert response.status_code == 429, response.text
        problem = response.json()
        assert problem["code"] == "quota_exhausted"
        assert f"HTTP {status}" in problem["detail"] and words in problem["detail"]
        assert len(tavily.seen) == 1 and clock.slept == []


def test_a_location_is_named_as_ignored(tmp_path, nowhere) -> None:
    tavily = Upstream(lambda request: httpx.Response(200, json=TAVILY))
    with _client(tmp_path, config(), nowhere, tavily) as client:
        response = search(
            client, userLocation={"type": "approximate", "approximate": {"country": "US"}}
        )
    assert response.status_code == 200, response.text
    assert response.json()["ignored"] == ["userLocation"]


def test_no_key_is_not_set_up(tmp_path, nowhere) -> None:
    tavily = Upstream(lambda request: httpx.Response(200, json=TAVILY))
    with _client(tmp_path, {"provider": "tavily"}, nowhere, tavily) as client:
        response = search(client)
        info = client.get("/v1/info").json()
    assert response.status_code == 503
    assert "add your Tavily key" in response.json()["detail"]
    assert not tavily.seen
    assert info["configured"] is False and info["tools"] == ["mcp"]


def test_tavily_is_paid_and_never_probed() -> None:
    tavily = PROVIDERS["tavily"]
    assert tavily.billing == "per_search" and tavily.probes is False


def test_the_settings_page_offers_tavily_and_its_depth() -> None:
    fields = {f.key: f for f in as_schema(values=config()).fields}
    provider = fields["provider"]
    assert provider.enumValues is not None and "tavily" in provider.enumValues
    assert provider.enumLabels is not None
    assert provider.enumLabels[provider.enumValues.index("tavily")] == "Tavily"
    depth = fields["searchDepth"]
    assert depth.default == "basic" and depth.enumValues == ["basic", "advanced"]
    assert depth.enumLabels == ["Basic: 1 credit a search", "Advanced: 2 credits a search"]
    assert depth.showWhen is not None and depth.showWhen.equals == ["tavily"]
    key = fields["apiKey"]
    assert key.showWhen is not None and "tavily" in key.showWhen.equals
    assert fields["baseUrl"].unsetResolvesTo == "https://api.tavily.com"
    unset = {f.key: f for f in as_schema(values={"provider": "tavily"}).fields}
    assert unset["apiKey"].unsetMeans == "Not set: Tavily refuses a search without a key."
    # Tavily has no language or safe-search setting: neither is shown for it.
    for name in ("language", "safeSearch"):
        shown = fields[name].showWhen
        assert shown is not None and "tavily" not in shown.equals, name
