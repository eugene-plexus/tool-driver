"""One search through each provider, and every failure it can name.

The SearXNG answer is the real one captured on 2026-09-29
(`fixtures/searxng-answer.json`); the Brave answer follows Brave's API
documentation, because there is no Brave key on the development box.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_tool_driver._generated.models import ConfigUpdateRequest, WebSearchRequest
from eugene_plexus_tool_driver.app import create_app
from eugene_plexus_tool_driver.config import ConfigStore
from eugene_plexus_tool_driver.pacing import Clock, Sleep
from eugene_plexus_tool_driver.providers import _brave_limited
from eugene_plexus_tool_driver.search import (
    Hit,
    SearchFailure,
    domain_matches,
    keep,
    plain,
    with_site,
)
from eugene_plexus_tool_driver.service import SearchService
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

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


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


class FakeTime:
    """A clock that moves only when slept on or told to, so a wait is
    asserted in exact seconds and no test spends a real one."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)

    def timing(self) -> tuple[Clock, Sleep]:
        return self.clock, self.sleep


def _client(
    tmp_path: Path,
    config: dict[str, Any],
    lan: Upstream,
    public: Upstream,
    *,
    time: FakeTime | None = None,
) -> TestClient:
    config_file = tmp_path / "search.yaml"
    config_file.write_text(yaml.safe_dump({"probeMinutes": 0, **config}), encoding="utf-8")
    app = create_app(Settings(config_file=config_file))
    app.state.search_transports = (lan.transport(), public.transport())
    if time is not None:
        app.state.search_timing = time.timing()
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
    time = FakeTime()
    with _client(
        tmp_path,
        {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"},
        busy,
        nowhere,
        time=time,
    ) as client:
        response = search(client)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "7"
    assert response.json()["code"] == "rate_limited"
    assert "limiter" in response.json()["detail"]
    # Handed straight back, deliberately: only Brave's refusal says whether
    # a short wait will do, and a 7s wait would fit a 15s search.
    assert len(busy.seen) == 1 and time.slept == []


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


def brave_client(
    tmp_path: Path, public: Upstream, lan: Upstream, *, time: FakeTime | None = None, **config: Any
) -> TestClient:
    return _client(
        tmp_path,
        {"provider": "brave", "apiKey": "brave-test-key", **config},
        lan,
        public,
        time=time,
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
    # tool-driver#3 changed this deliberately: a short refusal is tried
    # once more, and only the second comes back -- still a 429 with its
    # Retry-After, so the gateway moves on to another account if it has one.
    busy = Upstream(lambda request: httpx.Response(429, headers={"Retry-After": "2"}))
    time = FakeTime()
    with brave_client(tmp_path, busy, nowhere, time=time) as client:
        response = search(client)
    assert response.status_code == 429 and response.headers["Retry-After"] == "2"
    assert response.json()["code"] == "rate_limited"
    assert "tried once more after 2s" in response.json()["detail"]
    assert len(busy.seen) == 2 and time.slept == [2.0]
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


# --- Brave's limits: taking turns, and one retry (tool-driver#3) -----------

#: Seconds left in Brave's monthly window, from its documentation's example.
MONTH = "1419704"


def brave_answers(*answers: dict[str, str] | None) -> Upstream:
    """Brave answering each in turn, then the last one again: a dict is a
    429 carrying those headers, None the documented results."""
    left = list(answers)

    def respond(request: httpx.Request) -> httpx.Response:
        answer = left.pop(0) if len(left) > 1 else left[0]
        if answer is None:
            return httpx.Response(200, json=BRAVE)
        return httpx.Response(
            429,
            headers=answer,
            json={"type": "ErrorResponse", "error": {"code": "RATE_LIMITED", "status": 429}},
        )

    return Upstream(respond)


def _service(tmp_path: Path, upstream: Upstream, time: FakeTime, **config: Any) -> SearchService:
    store = ConfigStore(tmp_path / "search.yaml")
    store.load()
    store.apply_patch(ConfigUpdateRequest.model_validate({"probeMinutes": 0, **config}))
    transport = upstream.transport()
    return SearchService(store, transports=(transport, transport), timing=time.timing())


class InFlight:
    """An upstream that counts how many searches it is answering at once."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.now = self.peak = 0
        self.body = body

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.now += 1
        self.peak = max(self.peak, self.now)
        # Real, and long enough for another search to start if it may.
        await asyncio.sleep(0.005)
        self.now -= 1
        return httpx.Response(200, json=self.body)


def test_brave_searches_are_spaced_from_the_last_answer(tmp_path, nowhere) -> None:
    time = FakeTime()
    sent: list[float] = []

    def answers_in_0_4s(request: httpx.Request) -> httpx.Response:
        sent.append(time.now)
        time.now += 0.4
        return httpx.Response(200, json=BRAVE)

    with brave_client(tmp_path, Upstream(answers_in_0_4s), nowhere, time=time) as client:
        assert search(client).status_code == 200
        assert search(client).status_code == 200
        time.now += 5
        assert search(client).status_code == 200
    # A full second after the first ANSWER, so 1.4s after the first request:
    # spacing the requests would let two arrive inside one of Brave's seconds.
    assert time.slept == pytest.approx([1.0])
    assert sent[1] - sent[0] == pytest.approx(1.4)
    # And nothing is waited once the second has passed by itself.
    assert sent[2] - sent[1] == pytest.approx(5.4)


async def test_searches_on_a_paced_account_take_turns(tmp_path) -> None:
    time = FakeTime()
    upstream = InFlight(BRAVE)
    seen = Upstream(upstream)
    service = _service(tmp_path, seen, time, provider="brave", apiKey="k")
    try:
        found = await asyncio.gather(
            *(service.web_search(WebSearchRequest(query=f"chat {n}")) for n in range(3))
        )
    finally:
        await service.aclose()
    assert all(answer.results for answer in found) and len(seen.seen) == 3
    assert upstream.peak == 1
    assert time.slept == pytest.approx([1.0, 1.0])
    # Waiting for a turn is part of the search's 15s, not on top of it.
    assert [r.extensions["timeout"]["read"] for r in seen.seen] == pytest.approx([15, 14, 13])


async def test_an_account_takes_no_turns_unless_it_is_paced(tmp_path) -> None:
    time = FakeTime()
    searxng = InFlight(SEARXNG)
    service = _service(
        tmp_path, Upstream(searxng), time, provider="searxng", baseUrl="http://192.168.1.20:8888"
    )
    try:
        await asyncio.gather(*(service.web_search(WebSearchRequest(query="q")) for _ in range(2)))
        # SearXNG is not paced by default: the two ran side by side.
        assert searxng.peak == 2 and time.slept == []
        service.store.apply_patch(
            ConfigUpdateRequest.model_validate({"searchIntervalSeconds": 0.5})
        )
        searxng.peak = 0
        await asyncio.gather(*(service.web_search(WebSearchRequest(query="q")) for _ in range(2)))
        # Paced now, and the first gap counts from the last answer, paced or not.
        assert searxng.peak == 1 and time.slept == pytest.approx([0.5, 0.5])
    finally:
        await service.aclose()

    # And a Brave account its owner set to 0 (a paid plan) is not paced:
    # 0 is a choice, not "unset".
    brave = InFlight(BRAVE)
    service = _service(
        tmp_path, Upstream(brave), FakeTime(), provider="brave", apiKey="k", searchIntervalSeconds=0
    )
    try:
        await asyncio.gather(*(service.web_search(WebSearchRequest(query="q")) for _ in range(2)))
    finally:
        await service.aclose()
    assert brave.peak == 2


@pytest.mark.parametrize(
    ("reset", "waited"),
    [(f"3, {MONTH}", 3.0), (f"0, {MONTH}", 1.0)],
    ids=["for-the-windows-reset", "never-less-than-the-interval"],
)
def test_a_too_soon_refusal_is_tried_once_more_and_answered(
    tmp_path, nowhere, reset, waited
) -> None:
    time = FakeTime()
    brave = brave_answers({"X-RateLimit-Remaining": "0, 1500", "X-RateLimit-Reset": reset}, None)
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
        assert response.status_code == 200, response.text
        assert client.get("/healthz").json()["details"]["lastSearchOk"] is True
    assert response.json()["results"]
    assert len(brave.seen) == 2 and time.slept == [waited]
    # The retry has what is left of the search's 15s, not 15s more.
    timeouts = [r.extensions["timeout"]["read"] for r in brave.seen]
    assert timeouts == pytest.approx([15, 15 - waited])


def test_a_retry_that_finds_the_month_used_up_says_quota(tmp_path, nowhere) -> None:
    time = FakeTime()
    brave = brave_answers(
        {"X-RateLimit-Remaining": "0, 1", "X-RateLimit-Reset": f"1, {MONTH}"},
        {"X-RateLimit-Remaining": "0, 0", "X-RateLimit-Reset": f"1, {MONTH}"},
    )
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
    assert response.status_code == 429 and response.headers["Retry-After"] == MONTH
    problem = response.json()
    assert problem["code"] == "quota_exhausted"
    assert problem["detail"].endswith("resets in about 16 days. It was tried once more after 1s.")
    assert len(brave.seen) == 2 and time.slept == [1.0]


def test_a_second_refusal_comes_back_as_a_429_with_its_wait(tmp_path, nowhere) -> None:
    time = FakeTime()
    brave = brave_answers({"X-RateLimit-Remaining": "0, 1500", "X-RateLimit-Reset": f"1, {MONTH}"})
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
    assert response.status_code == 429 and response.headers["Retry-After"] == "1"
    problem = response.json()
    assert problem["code"] == "rate_limited"
    assert "faster than its plan allows" in problem["detail"]
    assert "tried once more after 1s" in problem["detail"]
    assert len(brave.seen) == 2 and time.slept == [1.0]


@pytest.mark.parametrize(
    ("headers", "resets"),
    [
        ({"X-RateLimit-Remaining": "1, 0", "X-RateLimit-Reset": f"1, {MONTH}"}, MONTH),
        ({"X-RateLimit-Remaining": "0, 0", "X-RateLimit-Reset": f"1, {MONTH}"}, MONTH),
        ({"X-RateLimit-Remaining": "1, 0"}, None),
        # A legacy free key's own limits (1 a second, 2,000 a month) say
        # the month has an allocation, so 0 left in it is used up.
        (
            {
                "X-RateLimit-Limit": "1, 2000",
                "X-RateLimit-Remaining": "1, 0",
                "X-RateLimit-Reset": f"1, {MONTH}",
            },
            MONTH,
        ),
        # A limit header that cannot be paired with `Remaining`, or does not
        # read, is not believed, and the reading without it stands.
        (
            {
                "X-RateLimit-Limit": "50, 0, 0",
                "X-RateLimit-Remaining": "1, 0",
                "X-RateLimit-Reset": f"1, {MONTH}",
            },
            MONTH,
        ),
        (
            {
                "X-RateLimit-Limit": "50, none",
                "X-RateLimit-Remaining": "1, 0",
                "X-RateLimit-Reset": f"1, {MONTH}",
            },
            MONTH,
        ),
    ],
    ids=[
        "month-used-up",
        "second-and-month-used-up",
        "no-reset-given",
        "legacy-free-key-with-its-limits",
        "limit-windows-disagree",
        "limit-not-a-number",
    ],
)
def test_a_used_up_month_says_quota_and_is_not_retried(tmp_path, nowhere, headers, resets) -> None:
    time = FakeTime()
    brave = brave_answers(headers)
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
        health = client.get("/healthz").json()
    assert response.status_code == 429
    problem = response.json()
    assert problem["code"] == "quota_exhausted"
    assert "monthly quota is used up" in problem["detail"]
    assert ("resets in about 16 days" in problem["detail"]) == (resets is not None)
    assert response.headers.get("Retry-After") == resets
    assert len(brave.seen) == 1 and time.slept == []
    assert health["details"]["code"] == "quota_exhausted"


@pytest.mark.parametrize(
    ("reset", "refusal_takes"),
    [(f"20, {MONTH}", 0.0), (f"10, {MONTH}", 3.0)],
    ids=["longer-than-the-timeout", "no-room-left-for-the-answer"],
)
def test_a_wait_the_search_timeout_cannot_hold_is_not_taken(
    tmp_path, nowhere, reset, refusal_takes
) -> None:
    time = FakeTime()

    def refuse(request: httpx.Request) -> httpx.Response:
        time.now += refusal_takes
        return httpx.Response(
            429, headers={"X-RateLimit-Remaining": "0, 1500", "X-RateLimit-Reset": reset}
        )

    brave = Upstream(refuse)
    with brave_client(tmp_path, brave, nowhere, time=time, timeoutSeconds=15) as client:
        response = search(client)
    assert response.status_code == 429 and response.json()["code"] == "rate_limited"
    assert response.headers["Retry-After"] == reset.split(",")[0]
    assert len(brave.seen) == 1 and time.slept == []


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-RateLimit-Remaining": "lots, 0", "X-RateLimit-Reset": f"1, {MONTH}"},
        {"X-RateLimit-Remaining": "1, 0", "X-RateLimit-Reset": "1, 2, 3"},
        {"X-RateLimit-Remaining": "-1, 0"},
        {"X-RateLimit-Remaining": "", "X-RateLimit-Reset": ","},
        {"X-RateLimit-Reset": "nan, inf"},
        {"X-RateLimit-Reset": "soon"},
        {"Retry-After": "Wed, 01 Oct 2026 12:00:00 GMT"},
    ],
    ids=[
        "none",
        "remaining-not-numbers",
        "windows-disagree",
        "negative",
        "empty",
        "not-finite",
        "reset-not-a-number",
        "retry-after-as-a-date",
    ],
)
def test_limit_headers_that_do_not_read_are_treated_as_absent(tmp_path, nowhere, headers) -> None:
    time = FakeTime()
    brave = brave_answers(headers)
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
    # Still the provider's 429, never a 500 ...
    assert response.status_code == 429
    problem = response.json()
    # ... no quota claimed, since nothing readable says the month is used up ...
    assert problem["code"] == "rate_limited"
    assert "rate or monthly limit" in problem["detail"]
    # ... and with no wait read, the one retry waits one interval.
    assert len(brave.seen) == 2 and time.slept == [1.0]


def test_a_turn_that_comes_after_the_timeout_is_not_taken(tmp_path, brave, nowhere) -> None:
    time = FakeTime()
    with brave_client(
        tmp_path, brave, nowhere, time=time, searchIntervalSeconds=10, timeoutSeconds=5
    ) as client:
        assert search(client).status_code == 200
        response = search(client)
    assert response.status_code == 429 and response.json()["code"] == "rate_limited"
    assert "10s between searches" in response.json()["detail"]
    assert response.headers["Retry-After"] == "10"
    assert len(brave.seen) == 1 and time.slept == []


def test_a_brave_refusal_is_read_by_window() -> None:
    def read(**headers: str) -> SearchFailure:
        return _brave_limited(httpx.Response(429, headers=headers))

    too_soon = read(**{"X-RateLimit-Remaining": "0, 9", "X-RateLimit-Reset": "1, 99"})
    assert too_soon.worth_a_retry and too_soon.retry_after == 1
    # The longer of the two waits Brave gives.
    both = read(**{"Retry-After": "4", "X-RateLimit-Reset": "2, 99"})
    assert both.retry_after == 4 and both.worth_a_retry
    quota = read(**{"X-RateLimit-Remaining": "5, 0", "X-RateLimit-Reset": "1, 7200"})
    assert not quota.worth_a_retry and quota.retry_after == 7200
    assert "resets in about 2 hours" in quota.detail
    # A window Brave allocates nothing to is not one that ran out.
    prepaid = read(**PREPAID)
    assert prepaid.code == "rate_limited" and prepaid.worth_a_retry
    assert prepaid.retry_after == 1


# Brave stopped issuing free keys on 2026-02-12. A prepaid key reports a
# second window with no allocation at all, which always reads 0 left:
# `X-RateLimit-Limit: 50, 0` / `X-RateLimit-Remaining: 49, 0`, its month
# resetting in about 29 days. From a third party's header capture
# (nicobailon/pi-web-access#501, fixed in #503), not measured here: there
# is no Brave key on the development box.
PREPAID = {
    "X-RateLimit-Limit": "50, 0",
    "X-RateLimit-Remaining": "0, 0",
    "X-RateLimit-Reset": "1, 2523327",
}


def test_a_prepaid_keys_empty_month_is_not_a_used_up_quota(tmp_path, nowhere) -> None:
    time = FakeTime()
    brave = brave_answers(PREPAID, None)
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
        health = client.get("/healthz").json()
    # Too soon, so tried once more after the second's reset, and answered.
    assert response.status_code == 200, response.text
    assert response.json()["results"]
    assert health["details"]["lastSearchOk"] is True
    assert len(brave.seen) == 2 and time.slept == [1.0]


def test_a_prepaid_key_refused_twice_is_told_too_fast_not_quota(tmp_path, nowhere) -> None:
    time = FakeTime()
    brave = brave_answers(PREPAID)
    with brave_client(tmp_path, brave, nowhere, time=time) as client:
        response = search(client)
        health = client.get("/healthz").json()
    assert response.status_code == 429 and response.headers["Retry-After"] == "1"
    problem = response.json()
    assert problem["code"] == "rate_limited"
    assert "faster than its plan allows" in problem["detail"]
    assert "quota" not in problem["detail"] and "days" not in problem["detail"]
    assert health["details"]["code"] == "rate_limited"
    assert len(brave.seen) == 2 and time.slept == [1.0]


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
