"""Google Search as a search account (`google-search-account.md`, GS1-GS9).

The answer is the real `generateContent` reply captured on 2026-10-09 from
`gemini-3.5-flash-lite` with the `googleSearch` tool
(`fixtures/google-grounding-answer.json`). What Google's redirect host
answers, the model listing and every error body are played from Google's
documented shapes: the box has one Gemini key and it was used once.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_tool_driver import providers
from eugene_plexus_tool_driver.config import FIELDS, ConfigStore, as_schema
from eugene_plexus_tool_driver.providers import PROVIDERS, choose_model

from .test_web_search import BRAVE, SEARXNG, FakeTime, Upstream, _client, search

GROUNDED = json.loads(
    (Path(__file__).parent / "fixtures" / "google-grounding-answer.json").read_text(
        encoding="utf-8"
    )
)
KEY = "AIzaSy-test-key-do-not-leak-0123456789"
GOOGLE = "generativelanguage.googleapis.com"
REDIRECT_HOST = "vertexaisearch.cloud.google.com"
ANSWER = GROUNDED["candidates"][0]["content"]["parts"][0]["text"]
SUGGESTIONS = GROUNDED["candidates"][0]["groundingMetadata"]["searchEntryPoint"]["renderedContent"]
CHUNKS = GROUNDED["candidates"][0]["groundingMetadata"]["groundingChunks"]
URIS = [c["web"]["uri"] for c in CHUNKS]
TITLES = [c["web"]["title"] for c in CHUNKS]
#: Where Google's redirect sends each source; None is a redirect that fails.
DESTINATIONS: list[str | None] = [
    "https://www.pishop.us/product/raspberry-pi-5-8gb/",
    "https://www.centralcomputer.com/raspberry-pi-5-8gb.html",
    "https://www.raspberrypi.com/products/raspberry-pi-5/",
    None,
    "https://dev.to/someone/raspberry-pi-5-price",
]


def listing_page(*models: tuple[str, list[str]], next_token: str | None = None) -> dict[str, Any]:
    page: dict[str, Any] = {
        "models": [
            {"name": f"models/{name}", "supportedGenerationMethods": methods}
            for name, methods in models
        ]
    }
    if next_token:
        page["nextPageToken"] = next_token
    return page


GEN = ["generateContent", "countTokens"]
PAGES = [
    listing_page(
        ("gemini-2.5-flash", GEN),
        ("gemini-embedding-001", ["embedContent"]),
        # Named in our preference list but not able to generate: skipped.
        ("gemini-3.5-flash-lite", ["countTokens"]),
        next_token="page-2",
    ),
    listing_page(("gemini-3.1-flash-lite", GEN), ("gemini-3-flash", GEN)),
]


def error(status: int, text: str, code: int, **extra: Any) -> httpx.Response:
    return httpx.Response(
        code, json={"error": {"code": code, "message": text, "status": status, **extra}}
    )


class FakeGoogle:
    """Google's API and its redirect host, recording every request."""

    def __init__(self) -> None:
        self.grounded: dict[str, Any] = GROUNDED
        self.pages = PAGES
        self.generate: Callable[[httpx.Request], httpx.Response] | None = None
        self.listing: Callable[[httpx.Request], httpx.Response] | None = None
        self.redirect_delay = 0.0
        self.now = self.peak = 0
        self.upstream = Upstream(self.handle)

    @property
    def seen(self) -> list[httpx.Request]:
        return self.upstream.seen

    def hosts(self) -> set[str]:
        return {r.url.host for r in self.seen}

    def generated(self) -> list[httpx.Request]:
        return [r for r in self.seen if r.url.path.endswith(":generateContent")]

    def listed(self) -> list[httpx.Request]:
        return [r for r in self.seen if r.url.host == GOOGLE and r.method == "GET"]

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == GOOGLE:
            if request.method == "POST":
                if self.generate is not None:
                    return self.generate(request)
                return httpx.Response(200, json=self.grounded)
            if self.listing is not None:
                return self.listing(request)
            token = request.url.params.get("pageToken")
            return httpx.Response(200, json=self.pages[1 if token == "page-2" else 0])
        if request.url.host == REDIRECT_HOST:
            self.now += 1
            self.peak = max(self.peak, self.now)
            await asyncio.sleep(self.redirect_delay)
            self.now -= 1
            for uri, destination in zip(URIS, DESTINATIONS, strict=True):
                if request.url.path == httpx.URL(uri).path:
                    if destination is None:
                        return httpx.Response(500, text="boom")
                    return httpx.Response(302, headers={"Location": destination})
        # The destination page, or anything else: recorded, and a 200 so a
        # follower of redirects would carry on.
        return httpx.Response(200, text="<html>a page</html>")


@pytest.fixture(autouse=True)
def fresh_picks() -> Iterator[None]:
    providers.forget_models()
    yield
    providers.forget_models()


@pytest.fixture
def google() -> FakeGoogle:
    return FakeGoogle()


@pytest.fixture
def lan() -> Upstream:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the internal transport must not be used: {request.url}")

    return Upstream(refuse)


def google_client(
    tmp_path: Path, fake: FakeGoogle, lan: Upstream, *, time: FakeTime | None = None, **config: Any
) -> TestClient:
    return _client(
        tmp_path, {"provider": "google", "apiKey": KEY, **config}, lan, fake.upstream, time=time
    )


def run(
    tmp_path: Path,
    fake: FakeGoogle,
    lan: Upstream,
    config: dict[str, Any] | None = None,
    **body: Any,
) -> httpx.Response:
    with google_client(tmp_path, fake, lan, **(config or {})) as client:
        return search(client, **body)


# --- the request ------------------------------------------------------------


def test_a_search_is_one_generate_content_call_with_the_key_in_a_header(
    tmp_path, google, lan
) -> None:
    response = run(tmp_path, google, lan, {"searchModel": "gemini-3.5-flash-lite"})
    assert response.status_code == 200, response.text
    assert len(google.generated()) == 1 and not google.listed()
    sent = google.generated()[0]
    assert sent.method == "POST"
    assert str(sent.url) == (
        f"https://{GOOGLE}/v1beta/models/gemini-3.5-flash-lite:generateContent"
    )
    assert sent.headers["x-goog-api-key"] == KEY
    assert json.loads(sent.content) == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Search the web for: llama.cpp rpc server. "
                        "Answer briefly from what you find."
                    }
                ],
            }
        ],
        "tools": [{"googleSearch": {}}],
    }
    # The key travels in a header only: not in a URL, not in a body.
    for request in google.seen:
        assert KEY not in str(request.url) and KEY.encode() not in request.content
    assert response.json()["provider"] == "google"


def test_the_site_hint_and_the_language_are_in_the_instruction(tmp_path, google, lan) -> None:
    run(
        tmp_path,
        google,
        lan,
        {"searchModel": "gemini-3.5-flash-lite", "language": "de"},
        allowedDomains=["github.com"],
    )
    text = json.loads(google.generated()[0].content)["contents"][0]["parts"][0]["text"]
    assert text == (
        "Search the web for: llama.cpp rpc server site:github.com. "
        "Answer briefly from what you find. Answer in de."
    )


def test_the_key_is_not_sent_to_googles_redirect_host(tmp_path, google, lan) -> None:
    run(tmp_path, google, lan, {"searchModel": "gemini-3.5-flash-lite"})
    redirects = [r for r in google.seen if r.url.host == REDIRECT_HOST]
    assert len(redirects) == len(URIS)
    assert all("x-goog-api-key" not in r.headers for r in redirects)


def test_a_search_without_a_key_is_not_set_up(tmp_path, google, lan) -> None:
    with _client(tmp_path, {"provider": "google"}, lan, google.upstream) as client:
        response = search(client)
    assert response.status_code == 503 and response.json()["code"] == "not_configured"
    assert "Google Search (Gemini API key)" in response.json()["detail"]
    assert not google.seen


# --- the model (GS3) --------------------------------------------------------


def test_the_model_is_chosen_from_the_keys_paginated_listing(tmp_path, google, lan) -> None:
    with google_client(tmp_path, google, lan) as client:
        assert search(client).status_code == 200
        listed = google.listed()
        assert [r.url.path for r in listed] == ["/v1beta/models", "/v1beta/models"]
        assert listed[0].url.params["pageSize"] == "1000"
        assert "pageToken" not in listed[0].url.params
        assert listed[1].url.params["pageToken"] == "page-2"
        assert all(r.headers["x-goog-api-key"] == KEY for r in listed)
        # gemini-3.5-flash-lite is on the list but cannot generate; the next
        # preference is used.
        assert google.generated()[0].url.path.endswith(
            "/models/gemini-3.1-flash-lite:generateContent"
        )
        # The pick is kept for the hour: a second search does not list again.
        assert search(client).status_code == 200
        assert len(google.listed()) == 2
        assert len(google.generated()) == 2


def test_an_expired_pick_is_listed_again(tmp_path, google, lan) -> None:
    with google_client(tmp_path, google, lan) as client:
        search(client)
        for key in list(providers._PICKS):
            model, _ = providers._PICKS[key]
            providers._PICKS[key] = (model, 0.0)
        search(client)
    assert len(google.listed()) == 4


def test_a_pick_is_per_key_and_per_address(tmp_path, google, lan) -> None:
    with google_client(tmp_path, google, lan) as client:
        search(client)
    with google_client(tmp_path, google, lan, apiKey="another-key") as client:
        search(client)
    assert len(google.listed()) == 4


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"], "gemini-3.5-flash-lite"),
        (["gemini-3-flash", "gemini-3.1-flash-lite"], "gemini-3.1-flash-lite"),
        (["gemini-3-flash", "gemini-4-flash-lite", "gemini-3.7-flash-lite"], "gemini-4-flash-lite"),
        (["gemini-3.7-flash-lite", "gemini-3.10-flash-lite"], "gemini-3.10-flash-lite"),
        (["gemini-2.5-flash", "gemini-3-flash", "gemini-pro-latest"], "gemini-3-flash"),
        (["gemini-3-flash-image", "gemini-3.5-flash-lite-tts", "gemini-3-pro"], None),
        ([], None),
    ],
)
def test_the_preference_for_the_cheapest_grounding_model(ids, expected) -> None:
    assert choose_model(ids) == expected


def test_the_setting_overrides_the_listing(tmp_path, google, lan) -> None:
    response = run(tmp_path, google, lan, {"searchModel": "models/gemini-3-flash"})
    assert response.status_code == 200
    assert not google.listed()
    assert google.generated()[0].url.path == "/v1beta/models/gemini-3-flash:generateContent"


def test_a_key_that_lists_no_model_that_can_search_names_the_setting(tmp_path, google, lan) -> None:
    google.pages = [listing_page(("gemini-embedding-001", ["embedContent"]))]
    response = run(tmp_path, google, lan)
    assert response.status_code == 502
    assert "searchModel" in response.json()["detail"]
    assert not google.generated()


def test_a_model_the_key_cannot_use_names_the_model_and_the_setting(tmp_path, google, lan) -> None:
    google.generate = lambda r: error("NOT_FOUND", "models/gemini-9 is not found", 404)
    response = run(tmp_path, google, lan, {"searchModel": "gemini-9"})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "gemini-9" in detail and "searchModel" in detail and "404" in detail


def test_a_chosen_model_that_goes_missing_is_chosen_again_next_time(tmp_path, google, lan) -> None:
    with google_client(tmp_path, google, lan) as client:
        search(client)
        assert providers.known_model(KEY, None) == "gemini-3.1-flash-lite"
        google.generate = lambda r: error("NOT_FOUND", "gone", 404)
        assert search(client).status_code == 502
        assert providers.known_model(KEY, None) is None


def _field(values: dict[str, Any], key: str) -> Any:
    return next(f for f in as_schema(values=values).fields if f.key == key)


def test_the_unset_model_says_what_is_in_effect_once_it_is_known(tmp_path, google, lan) -> None:
    values = {"provider": "google", "apiKey": KEY}
    before = _field(values, "searchModel")
    assert "chosen from the key's own listing at the first search" in before.unsetMeans
    assert before.unsetResolvesTo is None
    run(tmp_path, google, lan)
    after = _field(values, "searchModel")
    assert "gemini-3.1-flash-lite" in after.unsetMeans
    assert after.unsetResolvesTo == "gemini-3.1-flash-lite"
    # Another key's page does not claim this key's pick.
    other = _field({"provider": "google", "apiKey": "other"}, "searchModel")
    assert "gemini-3.1" not in other.unsetMeans
    # A model the person set is the value, not an unset.
    assert _field({**values, "searchModel": "gemini-3-flash"}, "searchModel").unsetMeans is None


# --- what comes back --------------------------------------------------------


def test_the_answer_is_googles_text_verbatim_with_its_sources(tmp_path, google, lan) -> None:
    body = run(tmp_path, google, lan, {"searchModel": "m"}, maxResults=10).json()
    assert body["answer"] == ANSWER
    assert [r["title"] for r in body["results"]] == TITLES
    urls = [r["url"] for r in body["results"]]
    # Resolved to the real address; the one redirect that failed keeps Google's link.
    assert urls == [d or u for d, u in zip(DESTINATIONS, URIS, strict=True)]
    assert urls[3].startswith(f"https://{REDIRECT_HOST}/grounding-api-redirect/")
    supports = GROUNDED["candidates"][0]["groundingMetadata"]["groundingSupports"]
    snippets = [r["snippet"] for r in body["results"]]
    assert snippets[0] == supports[0]["segment"]["text"] == snippets[1]
    assert snippets[2] == supports[1]["segment"]["text"]
    assert snippets[3] == snippets[4] == supports[2]["segment"]["text"]
    assert all(r["publishedAt"] is None for r in body["results"])
    assert body["searchSuggestions"] == SUGGESTIONS
    assert not body.get("ignored")


def test_no_destination_page_is_ever_fetched(tmp_path, google, lan) -> None:
    run(tmp_path, google, lan, {"searchModel": "m"}, maxResults=10)
    assert google.hosts() == {GOOGLE, REDIRECT_HOST}


def test_every_redirect_is_asked_at_once(tmp_path, google, lan) -> None:
    google.redirect_delay = 0.02
    run(tmp_path, google, lan, {"searchModel": "m"})
    assert google.peak == len(URIS)


def test_a_redirect_that_is_not_a_clean_redirect_keeps_googles_link(tmp_path, google, lan) -> None:
    answers: list[Callable[[], httpx.Response]] = [
        lambda: httpx.Response(200, text="no redirect here"),
        lambda: httpx.Response(302, headers={"Location": "/relative/path"}),
        lambda: httpx.Response(302, headers={"Location": "javascript:alert(1)"}),
        lambda: httpx.Response(302),
        lambda: httpx.Response(404),
    ]
    paths = {httpx.URL(u).path: a for u, a in zip(URIS, answers, strict=True)}

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == REDIRECT_HOST:
            return paths[request.url.path]()
        return httpx.Response(200, json=GROUNDED)

    upstream = Upstream(handle)
    with _client(
        tmp_path,
        {"provider": "google", "apiKey": KEY, "searchModel": "m"},
        google.upstream,
        upstream,
    ) as client:
        body = search(client, maxResults=10).json()
    assert [r["url"] for r in body["results"]] == URIS


def test_a_redirect_that_times_out_or_cannot_connect_keeps_googles_link(
    tmp_path, google, lan
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == REDIRECT_HOST:
            if request.url.path == httpx.URL(URIS[0]).path:
                raise httpx.ReadTimeout("slow", request=request)
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=GROUNDED)

    with _client(
        tmp_path,
        {"provider": "google", "apiKey": KEY, "searchModel": "m"},
        google.upstream,
        Upstream(handle),
    ) as client:
        response = search(client, maxResults=10)
    assert response.status_code == 200
    assert [r["url"] for r in response.json()["results"]] == URIS


def test_a_redirect_gets_at_most_three_seconds_and_no_more_than_the_search_has(
    tmp_path, google, lan
) -> None:
    run(tmp_path, google, lan, {"searchModel": "m"})
    redirect = next(r for r in google.seen if r.url.host == REDIRECT_HOST)
    assert redirect.extensions["timeout"]["read"] == 3.0
    short = FakeGoogle()
    run(tmp_path, short, lan, {"searchModel": "m", "timeoutSeconds": 2})
    redirect = next(r for r in short.seen if r.url.host == REDIRECT_HOST)
    assert 0 < redirect.extensions["timeout"]["read"] <= 2.0


def _with_parts(*parts: dict[str, Any]) -> dict[str, Any]:
    grounded = copy.deepcopy(GROUNDED)
    grounded["candidates"][0]["content"]["parts"] = list(parts)
    return grounded


def test_thoughts_are_not_the_answer_and_text_is_joined_untouched(tmp_path, google, lan) -> None:
    google.grounded = _with_parts(
        {"text": "Let me look this up.", "thought": True},
        {"text": "  First **part**\n", "thoughtSignature": "abc"},
        {"text": "second part.  "},
    )
    body = run(tmp_path, google, lan, {"searchModel": "m"}).json()
    assert body["answer"] == "  First **part**\nsecond part.  "


def test_a_source_the_supports_do_not_cite_has_no_snippet_and_a_long_one_is_trimmed(
    tmp_path, google, lan
) -> None:
    grounded = copy.deepcopy(GROUNDED)
    meta = grounded["candidates"][0]["groundingMetadata"]
    long_text = "word " * 200
    meta["groundingSupports"] = [
        {"segment": {"text": long_text}, "groundingChunkIndices": [0, 99, "x"]},
        {"segment": {"text": "Second."}, "groundingChunkIndices": [0]},
        {"segment": {"text": "Second."}, "groundingChunkIndices": [0]},
    ]
    google.grounded = grounded
    body = run(tmp_path, google, lan, {"searchModel": "m"}, contextSize="low").json()
    first = body["results"][0]["snippet"]
    assert first.endswith("…") and len(first) <= 241
    assert not body["results"][1]["snippet"]
    # Joined, and a repeated segment is said once.
    body = run(tmp_path, google, lan, {"searchModel": "m"}, contextSize="high").json()
    assert body["results"][0]["snippet"].count("Second.") == 1


def test_a_chunk_without_an_address_is_skipped(tmp_path, google, lan) -> None:
    grounded = copy.deepcopy(GROUNDED)
    grounded["candidates"][0]["groundingMetadata"]["groundingChunks"] = [
        {"retrievedContext": {"uri": "x"}},
        {"web": {"title": "no address"}},
        CHUNKS[2],
    ]
    google.grounded = grounded
    body = run(tmp_path, google, lan, {"searchModel": "m"}).json()
    assert [r["title"] for r in body["results"]] == ["raspberrypi.com"]


# --- domain filters (GS8) ---------------------------------------------------


def test_a_blocked_domain_drops_the_source_by_its_resolved_address_and_the_answer(
    tmp_path, google, lan
) -> None:
    # `dev.to`'s title is its domain, but it is the resolved address that is judged.
    body = run(
        tmp_path, google, lan, {"searchModel": "m"}, blockedDomains=["dev.to"], maxResults=10
    ).json()
    assert [r["title"] for r in body["results"]] == TITLES[:4]
    assert body["answer"] is None
    # Google's suggestions are still shown with what remains.
    assert body["searchSuggestions"] == SUGGESTIONS


def test_an_allowed_domain_keeps_only_matching_sources_and_drops_the_answer(
    tmp_path, google, lan
) -> None:
    body = run(
        tmp_path,
        google,
        lan,
        {"searchModel": "m"},
        allowedDomains=["raspberrypi.com", "pishop.us"],
        maxResults=10,
    ).json()
    assert [r["title"] for r in body["results"]] == ["pishop.us", "raspberrypi.com"]
    assert body["answer"] is None
    # An unresolved source is judged on Google's link, which is nobody's domain.
    body = run(tmp_path, google, lan, {"searchModel": "m"}, allowedDomains=["canakit.com"]).json()
    assert body["results"] == [] and body["answer"] is None


def test_without_filters_the_answer_stays(tmp_path, google, lan) -> None:
    assert run(tmp_path, google, lan, {"searchModel": "m"}).json()["answer"] == ANSWER


# --- suggestions, location, nothing grounded --------------------------------


def test_the_other_providers_answer_null_suggestions(tmp_path, lan) -> None:
    brave = Upstream(lambda request: httpx.Response(200, json=BRAVE))
    with _client(tmp_path, {"provider": "brave", "apiKey": "k"}, lan, brave) as client:
        body = search(client).json()
    assert "searchSuggestions" in body and body["searchSuggestions"] is None
    searx = Upstream(lambda request: httpx.Response(200, json=SEARXNG))
    unused = Upstream(lambda request: httpx.Response(500))
    with _client(
        tmp_path, {"provider": "searxng", "baseUrl": "http://192.168.1.20:8888"}, searx, unused
    ) as client:
        body = search(client).json()
    assert "searchSuggestions" in body and body["searchSuggestions"] is None


def test_google_without_suggestions_answers_null(tmp_path, google, lan) -> None:
    grounded = copy.deepcopy(GROUNDED)
    del grounded["candidates"][0]["groundingMetadata"]["searchEntryPoint"]
    google.grounded = grounded
    body = run(tmp_path, google, lan, {"searchModel": "m"}).json()
    assert body["searchSuggestions"] is None and body["answer"] == ANSWER


@pytest.mark.parametrize(
    "location",
    [
        {"type": "approximate", "approximate": {"country": "DE"}},
        {"type": "approximate", "approximate": {"city": "Berlin"}},
    ],
)
def test_a_location_is_named_as_ignored_and_not_sent(tmp_path, google, lan, location) -> None:
    body = run(tmp_path, google, lan, {"searchModel": "m"}, userLocation=location).json()
    assert body["ignored"] == ["userLocation"]
    sent = google.generated()[0].content.decode()
    assert "latLng" not in sent and "Berlin" not in sent and "DE" not in sent.replace("Answer", "")


@pytest.mark.parametrize(
    "grounded",
    [
        {"candidates": [{"content": {"parts": [{"text": "From memory: it is 175 dollars."}]}}]},
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "From memory: it is 175 dollars."}]},
                    "groundingMetadata": {"groundingChunks": []},
                }
            ]
        },
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {},
    ],
)
def test_a_search_google_did_not_run_has_no_hits_and_no_answer(
    tmp_path, google, lan, caplog, grounded
) -> None:
    google.grounded = grounded
    caplog.set_level(logging.DEBUG)
    response = run(tmp_path, google, lan, {"searchModel": "m"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["answer"] is None
    # The model's own words from memory are never logged, nor handed on.
    assert "175 dollars" not in caplog.text and "175 dollars" not in response.text


# --- failures ---------------------------------------------------------------


def _fail(tmp_path, google, lan, response: httpx.Response, **config: Any) -> httpx.Response:
    google.generate = lambda r: response
    return run(tmp_path, google, lan, {"searchModel": "m", **config})


@pytest.mark.parametrize(
    "refusal",
    [
        error(
            "INVALID_ARGUMENT",
            "API key not valid. Please pass a valid API key.",
            400,
            details=[
                {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}
            ],
        ),
        error("PERMISSION_DENIED", "Your API key was reported as leaked.", 403),
        error("UNAUTHENTICATED", "Request had invalid authentication credentials.", 401),
    ],
)
def test_a_refused_key_is_our_credential(tmp_path, google, lan, refusal) -> None:
    response = _fail(tmp_path, google, lan, refusal)
    assert response.status_code == 502
    problem = response.json()
    assert problem["code"] == "upstream_auth_error"
    assert problem["detail"].startswith("Google refused this account's Gemini API key")
    assert "Google AI Studio" in problem["detail"]


def test_a_429_carries_googles_wait_and_is_retried_only_when_there_is_one(
    tmp_path, google, lan
) -> None:
    retry = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}
    busy = error("RESOURCE_EXHAUSTED", "You exceeded your current quota.", 429, details=[retry])
    time = FakeTime()
    google.generate = lambda r: busy
    with google_client(tmp_path, google, lan, searchModel="m", time=time) as client:
        response = search(client)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "34"
    problem = response.json()
    assert problem["code"] == "rate_limited" and problem["retryAfterSeconds"] == 34
    assert "wait 34 seconds" in problem["detail"] and "quota" in problem["detail"]
    # 34 s does not fit in the search's 15: handed back, not retried.
    assert len(google.generated()) == 1 and time.slept == []

    # No wait given: handed back at once, and no Retry-After invented.
    google.seen.clear()
    google.generate = lambda r: error("RESOURCE_EXHAUSTED", "Quota exceeded.", 429)
    time = FakeTime()
    with google_client(tmp_path, google, lan, searchModel="m", time=time) as client:
        response = search(client)
    assert response.status_code == 429 and "Retry-After" not in response.headers
    assert len(google.generated()) == 1 and time.slept == []


def test_a_short_wait_from_googles_error_or_header_is_waited_for_once(
    tmp_path, google, lan
) -> None:
    retry = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}
    google.generate = lambda r: error("RESOURCE_EXHAUSTED", "Slow down.", 429, details=[retry])
    time = FakeTime()
    with google_client(tmp_path, google, lan, searchModel="m", time=time) as client:
        response = search(client)
    assert response.status_code == 429
    assert "tried once more after 2s" in response.json()["detail"]
    assert len(google.generated()) == 2 and time.slept == [2.0]

    google.seen.clear()
    google.generate = lambda r: httpx.Response(
        429, headers={"Retry-After": "3"}, json={"error": {"message": "Slow down."}}
    )
    time = FakeTime()
    with google_client(tmp_path, google, lan, searchModel="m", time=time) as client:
        response = search(client)
    assert response.headers["Retry-After"] == "3" and time.slept == [3.0]


def test_a_region_google_does_not_serve(tmp_path, google, lan) -> None:
    response = _fail(
        tmp_path,
        google,
        lan,
        error("FAILED_PRECONDITION", "User location is not supported for the API use.", 400),
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "does not serve the region" in detail and "User location is not supported" in detail


def test_any_other_refusal_gives_googles_own_message(tmp_path, google, lan) -> None:
    response = _fail(
        tmp_path, google, lan, error("INTERNAL", "An internal error has occurred.", 500)
    )
    assert response.status_code == 502 and response.json()["code"] == "upstream_error"
    assert "HTTP 500" in response.json()["detail"]
    assert "An internal error has occurred." in response.json()["detail"]
    bad = _fail(tmp_path, google, lan, httpx.Response(200, text="<html>not json</html>"))
    assert bad.status_code == 502 and "not JSON" in bad.json()["detail"]
    html = _fail(tmp_path, google, lan, httpx.Response(503, text="<html>down</html>"))
    assert html.status_code == 502 and "HTTP 503" in html.json()["detail"]


def test_a_slow_or_absent_google_is_a_timeout_or_unreachable(tmp_path, google, lan) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    google.generate = slow
    response = run(tmp_path, google, lan, {"searchModel": "m", "timeoutSeconds": 3})
    assert response.status_code == 504
    assert response.json()["code"] == "timeout" and "3s" in response.json()["detail"]

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    google.generate = refused
    response = run(tmp_path, google, lan, {"searchModel": "m"})
    assert response.status_code == 502 and response.json()["code"] == "unreachable"
    assert GOOGLE in response.json()["detail"] and "ConnectError" in response.json()["detail"]


def test_the_listing_failing_is_named_too(tmp_path, google, lan) -> None:
    google.listing = lambda r: error("PERMISSION_DENIED", "denied", 403)
    response = run(tmp_path, google, lan)
    assert response.status_code == 502 and response.json()["code"] == "upstream_auth_error"
    assert not google.generated()

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    google.listing = slow
    assert run(tmp_path, google, lan).status_code == 504


# --- the key is never shown -------------------------------------------------

ECHOING: list[tuple[str, Callable[[], httpx.Response]]] = [
    (
        "auth",
        lambda: error(
            "INVALID_ARGUMENT",
            f"API key not valid: {KEY}",
            400,
            details=[{"reason": "API_KEY_INVALID"}],
        ),
    ),
    ("quota", lambda: error("RESOURCE_EXHAUSTED", f"Quota exceeded for key {KEY}.", 429)),
    (
        "region",
        lambda: error("FAILED_PRECONDITION", f"User location is not supported ({KEY})", 400),
    ),
    ("other", lambda: error("INTERNAL", f"Broke while serving {KEY}", 500)),
    ("plain", lambda: httpx.Response(502, text=f"bad gateway for {KEY}")),
]


@pytest.mark.parametrize(("name", "make"), ECHOING, ids=[n for n, _ in ECHOING])
def test_an_error_that_echoes_the_key_is_scrubbed(
    tmp_path, google, lan, caplog, name, make
) -> None:
    caplog.set_level(logging.DEBUG)
    response = _fail(tmp_path, google, lan, make())
    assert response.status_code >= 400
    assert KEY not in response.text and KEY not in caplog.text
    assert KEY not in json.dumps(dict(response.headers))


def test_a_success_never_shows_the_key(tmp_path, google, lan, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    response = run(tmp_path, google, lan)
    assert response.status_code == 200
    assert KEY not in response.text and KEY not in caplog.text
    assert all(KEY not in str(r.url) for r in google.seen)


def test_the_key_is_not_in_the_config_document(tmp_path, google, lan) -> None:
    store = ConfigStore(tmp_path / "c.yaml")
    store.load()
    from eugene_plexus_tool_driver._generated.models import ConfigUpdateRequest

    store.apply_patch(ConfigUpdateRequest.model_validate({"provider": "google", "apiKey": KEY}))
    assert KEY not in store.as_document().model_dump_json()


# --- the account's page and /v1/info ----------------------------------------


def test_the_provider_is_registered_as_a_paid_search_with_no_probe() -> None:
    google = PROVIDERS["google"]
    assert google.label == "Google Search (Gemini API key)"
    assert google.default_base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert google.needs_key and not google.probes
    assert google.billing == "per_search" and google.search_interval == 0
    assert "quota" in google.search_interval_reason


def test_info_names_the_account(tmp_path, google, lan) -> None:
    with google_client(tmp_path, google, lan) as client:
        info = client.get("/v1/info").json()
    assert info["provider"] == "google"
    assert info["label"] == "Google Search (Gemini API key)"
    assert info["billing"] == "per_search" and info["tools"] == ["web_search"]
    assert not google.seen


def test_the_settings_page_for_google() -> None:
    by_key = {f.key: f for f in FIELDS}
    provider = by_key["provider"]
    assert "google" in provider.enumValues
    assert provider.enumLabels[provider.enumValues.index("google")] == (
        "Google Search (Gemini API key)"
    )
    api_key = by_key["apiKey"]
    assert api_key.showWhen is not None and api_key.showWhen.equals == ["brave", "google"]
    assert "Google AI Studio" in api_key.description
    assert (
        "Google's terms for Grounding with Google Search bind the key's owner: results are "
        "shown unmodified, with Google's Search Suggestions, to the person who asked. "
        "Workbench shows them; other apps may not."
    ) in api_key.description
    model = by_key["searchModel"]
    assert model.showWhen is not None and model.showWhen.equals == ["google"]
    assert not model.requiresRestart
    # Safe search does nothing for Google, so it is not offered there.
    safe = by_key["safeSearch"]
    assert safe.showWhen is not None and "google" not in safe.showWhen.equals
    assert by_key["probeMinutes"].showWhen.equals == ["searxng"]


def test_the_unset_facts_for_a_google_account() -> None:
    values = {"provider": "google"}
    base = _field(values, "baseUrl")
    assert "Google's own address" in base.unsetMeans
    assert base.unsetResolvesTo == "https://generativelanguage.googleapis.com/v1beta"
    interval = _field(values, "searchIntervalSeconds")
    assert interval.unsetResolvesTo == 0
    assert "as they come" in interval.unsetMeans and "project's quota" in interval.unsetMeans
    assert "Gemini API key" in _field(values, "apiKey").unsetMeans
    # SearXNG's own wording is unchanged.
    assert "quota" not in _field({"provider": "searxng"}, "searchIntervalSeconds").unsetMeans


def _grounded_with(chunks: list[dict[str, Any]], supports: list[dict[str, Any]]) -> dict[str, Any]:
    answer = copy.deepcopy(GROUNDED)
    meta = answer["candidates"][0]["groundingMetadata"]
    meta["groundingChunks"] = chunks
    meta["groundingSupports"] = supports
    return answer


def test_a_redirect_on_the_accounts_own_origin_is_resolved_and_no_other_host_is_asked(
    tmp_path, google, lan
) -> None:
    """A proxy or fixture in front of Google answers its redirects at the
    account's own `baseUrl` origin; a link on any other host is kept as
    given and never requested."""
    own = f"https://{GOOGLE}/grounding-api-redirect/own"
    other = "https://elsewhere.example/grounding-api-redirect/other"
    google.grounded = _grounded_with(
        [
            {"web": {"uri": own, "title": "a.example"}},
            {"web": {"uri": other, "title": "b.example"}},
        ],
        [],
    )

    def listing(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/grounding-api-redirect/own", request.url
        return httpx.Response(302, headers={"Location": "https://a.example/page"})

    google.listing = listing
    response = run(tmp_path, google, lan, {"searchModel": "gemini-3.5-flash-lite"})
    assert response.status_code == 200, response.text
    assert [r["url"] for r in response.json()["results"]] == ["https://a.example/page", other]
    assert "elsewhere.example" not in google.hosts()


def test_overlapping_segments_keep_the_longer_text_once(tmp_path, google, lan) -> None:
    """Google's segments can repeat one another; a source's snippet carries
    each sentence once, the longest form of it."""
    google.grounded = _grounded_with(
        [{"web": {"uri": URIS[0], "title": "a.example"}}],
        [
            {"segment": {"text": "The Pi 5 costs $80."}, "groundingChunkIndices": [0]},
            {
                "segment": {"text": "The Pi 5 costs $80. Resellers stock it."},
                "groundingChunkIndices": [0],
            },
            {"segment": {"text": "Resellers stock it."}, "groundingChunkIndices": [0]},
        ],
    )
    response = run(tmp_path, google, lan, {"searchModel": "gemini-3.5-flash-lite"})
    assert response.json()["results"][0]["snippet"] == "The Pi 5 costs $80. Resellers stock it."
