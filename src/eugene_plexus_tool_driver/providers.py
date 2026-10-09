"""The search providers: SearXNG (free, self-hosted), Brave (keyed) and Google (a Gemini key).

Design call #4 (`server-run-tools.md` §9). Each provider turns the one
`WebSearchRequest` into its own wire request and its answer back into
`search.Found`; everything they share -- filtering, excerpt length, the
`site:` hint -- lives in `search.py` so the two cannot disagree about it.

A provider's failure is raised as `search.SearchFailure` carrying the
status this component answers and a sentence naming the cause, because
that sentence is what the model is handed and then what a person reads.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from .search import Found, Hit, SearchFailure, host_of, keep, plain, trim, with_site

log = logging.getLogger(__name__)

BRAVE_URL = "https://api.search.brave.com"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta"

_SAFE_SEARXNG = {"off": "0", "moderate": "1", "strict": "2"}


@dataclass(frozen=True)
class Query:
    """One search, with the account's settings already applied."""

    query: str
    max_results: int
    allowed: list[str] | None
    blocked: list[str] | None
    country: str | None
    context_size: str | None
    safe_search: str
    language: str | None
    #: Seconds this request may take: what is left of the search's timeout.
    timeout: float
    #: The request carried a `userLocation` (of any kind), whether or not it
    #: named a country; a provider that cannot use one names it as ignored.
    located: bool = False


@dataclass(frozen=True)
class Account:
    """The account's saved settings, read when a search runs."""

    base_url: str
    api_key: str | None
    #: `searchModel`: the Gemini model that runs a Google search. None lets
    #: the provider choose from the key's own listing.
    search_model: str | None = None


SearchFn = Callable[[httpx.AsyncClient, Account, Query], Awaitable[Found]]


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    #: Where the provider lives when the operator gives no address.
    default_base_url: str | None
    needs_key: bool
    #: Free to ask on a timer. A paid provider never is: every probe is a
    #: search the operator pays for.
    probes: bool
    search: SearchFn
    #: `free` or `per_search`, reported on `/v1/info` so the gateway tries
    #: a free account before one that bills.
    billing: str = "per_search"
    #: Seconds from one answer to the next search when the operator sets
    #: none (`searchIntervalSeconds`); 0 sends searches as they come.
    search_interval: float = 0.0
    #: Whose pace `search_interval` is, finishing the settings page's
    #: "Not set: 1s between searches, ..." (settings never lie).
    search_interval_reason: str = ""

    def missing(self, account: Account) -> str | None:
        """What the account still needs before it can search, or None."""
        if not account.base_url:
            return "no address is set: give the SearXNG instance's address"
        if self.needs_key and not account.api_key:
            return f"no API key is set: add your {self.label} key"
        return None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            return None
    return None


async def _get(
    client: httpx.AsyncClient,
    who: str,
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str] | None,
    seconds: float,
) -> httpx.Response:
    try:
        return await client.get(url, params=params, headers=headers, timeout=seconds)
    except httpx.TimeoutException:
        # `seconds` is what was left of the search's timeout, so it is
        # rarely a round number.
        raise SearchFailure(
            504, "timeout", f"{who} did not answer within {round(seconds, 1):g}s"
        ) from None
    except httpx.HTTPError as exc:
        raise SearchFailure(
            502, "unreachable", f"{who} could not be reached at {url} ({type(exc).__name__})"
        ) from None


# --- SearXNG ---------------------------------------------------------------


async def search_searxng(client: httpx.AsyncClient, account: Account, query: Query) -> Found:
    """`GET /search?format=json`, as measured against a real instance.

    SearXNG has no location parameter, so a `userLocation` is named as
    ignored rather than silently dropped. It has no result count either:
    it answers every result its engines found, and the list is cut here.
    """
    base = account.base_url.rstrip("/")
    params: dict[str, Any] = {
        "q": with_site(query.query, query.allowed),
        "format": "json",
        "safesearch": _SAFE_SEARXNG.get(query.safe_search, "1"),
    }
    if query.language:
        params["language"] = query.language
    response = await _get(
        client, "SearXNG", f"{base}/search", params=params, headers=None, seconds=query.timeout
    )
    if response.status_code == 403:
        # Measured: a format the instance has not enabled is a 403 with an
        # HTML body, and nothing else in the answer says why.
        raise SearchFailure(
            502,
            "json_disabled",
            f"SearXNG at {base} refused JSON output (HTTP 403). Add `json` to "
            "`search.formats` in its settings.yml and restart it.",
        )
    if response.status_code == 429:
        raise SearchFailure(
            429,
            "rate_limited",
            f"SearXNG at {base} is limiting requests (HTTP 429). Its `server.limiter` "
            "setting refuses repeated searches from one address.",
            retry_after=_retry_after(response),
        )
    if response.status_code >= 400:
        raise SearchFailure(
            502, "upstream_error", f"SearXNG at {base} answered HTTP {response.status_code}"
        )
    try:
        body = response.json()
    except ValueError:
        raise SearchFailure(
            502, "upstream_error", f"SearXNG at {base} answered something that is not JSON"
        ) from None
    if not isinstance(body, dict):
        raise SearchFailure(502, "upstream_error", f"SearXNG at {base} answered an odd shape")
    hits = []
    for result in body.get("results") or []:
        if not isinstance(result, dict):
            continue
        url, title = result.get("url"), result.get("title")
        if not isinstance(url, str) or not url:
            continue
        hits.append(
            Hit(
                url=url,
                title=plain(title) or url,
                snippet=trim(plain(result.get("content")), query.context_size),
                published_at=result.get("publishedDate")
                if isinstance(result.get("publishedDate"), str)
                else None,
            )
        )
    answer = _searxng_answer(body.get("answers"))
    ignored = ["userLocation"] if query.country else []
    return Found(hits=hits, answer=answer, ignored=ignored)


def _searxng_answer(answers: Any) -> str | None:
    """SearXNG's instant answers, as text. Older instances answer strings,
    newer ones objects carrying `answer`; either is kept."""
    texts: list[str] = []
    for answer in answers or []:
        if isinstance(answer, str):
            texts.append(plain(answer))
        elif isinstance(answer, dict) and isinstance(answer.get("answer"), str):
            texts.append(plain(answer["answer"]))
    joined = " ".join(t for t in texts if t)
    return joined or None


# --- Brave -----------------------------------------------------------------


async def search_brave(client: httpx.AsyncClient, account: Account, query: Query) -> Found:
    """`GET /res/v1/web/search`, from Brave's API documentation.

    Not measured live: there is no Brave key on the development box. The
    fixture plays what the documentation says, and the record says so.
    """
    base = (account.base_url or BRAVE_URL).rstrip("/")
    params: dict[str, Any] = {
        "q": with_site(query.query, query.allowed),
        # Brave pages at 20. Asked for a few more than the caller wants
        # when a domain filter will remove results after the fact.
        "count": min(20, query.max_results + (5 if query.allowed or query.blocked else 0)),
        "safesearch": query.safe_search,
    }
    if query.country and len(query.country) == 2:
        params["country"] = query.country.upper()
    if query.language:
        params["search_lang"] = query.language.split("-")[0].split("_")[0].lower()
    if query.context_size == "high":
        params["extra_snippets"] = "true"
    response = await _get(
        client,
        "Brave Search",
        f"{base}/res/v1/web/search",
        params=params,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": account.api_key or "",
        },
        seconds=query.timeout,
    )
    if response.status_code in (401, 403):
        # Our credential, not the caller's request (R3.4's rule): the
        # model is told the search could not run, and the person is told
        # which key to fix.
        raise SearchFailure(
            502,
            "upstream_auth_error",
            f"Brave Search refused this account's API key (HTTP {response.status_code}). "
            "Check the key on the search account's settings.",
        )
    if response.status_code == 429:
        raise _brave_limited(response)
    if response.status_code == 422:
        raise SearchFailure(
            502,
            "upstream_error",
            f"Brave Search refused the search's parameters (HTTP 422): {_brave_detail(response)}",
        )
    if response.status_code >= 400:
        raise SearchFailure(
            502, "upstream_error", f"Brave Search answered HTTP {response.status_code}"
        )
    try:
        body = response.json()
    except ValueError:
        raise SearchFailure(
            502, "upstream_error", "Brave Search answered something that is not JSON"
        ) from None
    web = body.get("web") if isinstance(body, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    hits = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        if not isinstance(url, str) or not url:
            continue
        snippet = plain(result.get("description"))
        extras = result.get("extra_snippets")
        if query.context_size == "high" and isinstance(extras, list):
            snippet = " ".join([snippet, *(plain(e) for e in extras if isinstance(e, str))]).strip()
        published = result.get("page_age") or result.get("age")
        hits.append(
            Hit(
                url=url,
                title=plain(result.get("title")) or url,
                snippet=trim(snippet, query.context_size),
                published_at=published if isinstance(published, str) else None,
            )
        )
    return Found(hits=hits, answer=None, ignored=[])


def _brave_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("detail") or error.get("code") or error)[:200]
    return str(body)[:200]


def _brave_limited(response: httpx.Response) -> SearchFailure:
    """Brave's 429, read for whether a short wait will do.

    Brave documents its limits as headers holding one comma-separated
    value per window, shortest first: `X-RateLimit-Remaining: 0, 1500` is
    no search left this second and 1,500 this month, and
    `X-RateLimit-Reset: 1, 1419704` the seconds until each window starts
    again. From its documentation, not measured: there is no Brave key on
    the development box, so the first live 429 is the check on this.

    * A window after the first with nothing left is the monthly quota,
      used up. No wait inside a search ends it, so it is not retried, and
      the person reads "quota", not "too fast".
    * Except a window `X-RateLimit-Limit` gives no allocation at all.
      Brave stopped issuing free keys on 2026-02-12, and a prepaid key
      since reports `X-RateLimit-Limit: 50, 0` / `X-RateLimit-Remaining:
      49, 0`: a month with a limit of 0 always reads 0 left, so read as a
      quota it would turn every too-fast refusal into "used up, resets in
      about 29 days", never retried. Taken from a third party's header
      capture (nicobailon/pi-web-access#501), not measured here.
    * Anything else is too fast, and worth one retry after the first
      window's reset or the `Retry-After`, whichever is longer.
    * A header that is missing, or does not read as a number for every
      window, is treated as absent. With no `Remaining`, nothing says the
      month is used up; with no wait at all, the retry waits one pacing
      interval (`SearchService._paced`). With no `Limit` that pairs with
      `Remaining`, every window is read as allocated, as before Brave sent
      one (a legacy free key's `1, 2000` reads the same either way).
    """
    remaining = _per_window(response, "x-ratelimit-remaining")
    reset = _per_window(response, "x-ratelimit-reset")
    if remaining is not None and reset is not None and len(remaining) != len(reset):
        # Two headers that disagree on how many windows there are cannot
        # be paired, so neither is believed.
        remaining = reset = None
    limit = _per_window(response, "x-ratelimit-limit")
    unallocated = (
        {i for i, allowed in enumerate(limit) if allowed == 0}
        if limit is not None and remaining is not None and len(limit) == len(remaining)
        else set()
    )
    retry_after = _retry_after(response)
    used_up = [i for i, left in enumerate(remaining or []) if i > 0 and left == 0]
    used_up = [i for i in used_up if i not in unallocated]
    if used_up:
        resets_in = max(reset[i] for i in used_up) if reset is not None else retry_after
        when = f"; it resets in {_in_words(resets_in)}" if resets_in is not None else ""
        return SearchFailure(
            429,
            "quota_exhausted",
            f"Brave Search says this key's monthly quota is used up (HTTP 429){when}.",
            retry_after=resets_in,
        )
    waits = [w for w in (retry_after, reset[0] if reset is not None else None) if w is not None]
    # One window says nothing about the month; two with the later one not
    # used up say it was the pace.
    cause = (
        "is searching faster than its plan allows"
        if remaining is not None and len(remaining) > 1
        else "is over its rate or monthly limit"
    )
    return SearchFailure(
        429,
        "rate_limited",
        f"Brave Search says this key {cause} (HTTP 429).",
        retry_after=max(waits) if waits else None,
        worth_a_retry=True,
    )


def _per_window(response: httpx.Response, name: str) -> list[float] | None:
    """A comma-separated `X-RateLimit-*` header as one number per window.

    None when it is missing or any part is not a finite number of zero or
    more: a header read halfway is not one to decide on.
    """
    value = response.headers.get(name)
    if not value:
        return None
    try:
        numbers = [float(part) for part in value.split(",")]
    except ValueError:
        return None
    if not all(math.isfinite(n) and n >= 0 for n in numbers):
        return None
    return numbers


def _in_words(seconds: float) -> str:
    """A wait as a person says it: `about 16 days`, `about 3 hours`, `40 seconds`."""
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            count = round(seconds / size)
            return f"about {count} {unit}{'' if count == 1 else 's'}"
    count = math.ceil(seconds)
    return f"{count} second{'' if count == 1 else 's'}"


# --- Google (Grounding with Google Search, through a Gemini key) ------------

#: How long a model chosen from a key's listing is trusted.
MODEL_PICK_SECONDS = 3600.0
#: The longest one redirect lookup may take.
REDIRECT_SECONDS = 3.0
#: The only host whose links are asked for a `Location`: Google's own
#: redirect for a grounding source. Any other address in an answer is kept
#: as it is, so no page is ever fetched.
_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_PREFERRED_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
_FLASH_LITE = re.compile(r"^gemini-(\d+(?:\.\d+)*)-flash-lite$")
_FLASH = re.compile(r"^gemini-(\d+(?:\.\d+)*)-flash$")

#: (sha256 of the key, base address) -> (model, perf_counter() it expires at).
_PICKS: dict[tuple[str, str], tuple[str, float]] = {}


def _pick_key(api_key: str, base: str) -> tuple[str, str]:
    return hashlib.sha256(api_key.encode()).hexdigest(), base.rstrip("/")


def known_model(api_key: str | None, base_url: str | None) -> str | None:
    """The model an unset `searchModel` is using now, if one was chosen
    within the hour: what the settings page says is in effect."""
    if not api_key:
        return None
    entry = _PICKS.get(_pick_key(api_key, base_url or GOOGLE_URL))
    if entry is None or entry[1] <= time.perf_counter():
        return None
    return entry[0]


def forget_models() -> None:
    _PICKS.clear()


def choose_model(ids: list[str]) -> str | None:
    """The cheapest grounding model among `ids`: our two named flash-lites,
    else the newest flash-lite, else the newest flash."""
    for wanted in _PREFERRED_MODELS:
        if wanted in ids:
            return wanted
    for pattern in (_FLASH_LITE, _FLASH):
        found = [(i, m.group(1)) for i in ids if (m := pattern.match(i))]
        if found:
            return max(found, key=lambda pair: tuple(int(p) for p in pair[1].split(".")))[0]
    return None


def _scrub(text: str, key: str | None) -> str:
    """`text` with the key gone: Google can echo request details in an error."""
    return text.replace(key, "[key]") if key else text


def _time_left(seconds: float, started: float) -> float:
    """What is left of the search's time, or the timeout it has become."""
    left = seconds - (time.perf_counter() - started)
    if left <= 0:
        raise SearchFailure(504, "timeout", f"Google did not answer within {round(seconds, 1):g}s")
    return left


async def _post(
    client: httpx.AsyncClient,
    who: str,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    seconds: float,
) -> httpx.Response:
    try:
        return await client.post(url, json=json, headers=headers, timeout=seconds)
    except httpx.TimeoutException:
        raise SearchFailure(
            504, "timeout", f"{who} did not answer within {round(seconds, 1):g}s"
        ) from None
    except httpx.HTTPError as exc:
        raise SearchFailure(
            502, "unreachable", f"{who} could not be reached at {url} ({type(exc).__name__})"
        ) from None


def _google_error(response: httpx.Response) -> tuple[str, float | None]:
    """Google's `error.message`, and the wait its `RetryInfo` gives."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300], None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return str(body)[:300], None
    delay: float | None = None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            raw = str(detail.get("retryDelay") or "").strip().removesuffix("s")
            try:
                delay = max(0.0, float(raw))
            except ValueError:
                delay = None
    return str(error.get("message") or error.get("status") or "")[:300], delay


def _google_failure(response: httpx.Response, key: str | None, model: str | None) -> SearchFailure:
    """Google's refusal as the sentence a person and a model are handed."""
    status = response.status_code
    message, delay = _google_error(response)
    message = _scrub(message, key)
    if "location is not supported" in message.lower():
        return SearchFailure(
            502,
            "upstream_error",
            f"Google does not serve the region this tool driver runs in (HTTP {status}: "
            f"{message}). Run the search account on a node in a region Google supports.",
        )
    if status in (401, 403) or (status == 400 and "API_KEY_INVALID" in response.text):
        return SearchFailure(
            502,
            "upstream_auth_error",
            f"Google refused this account's Gemini API key (HTTP {status}). Check the key "
            "on the search account's settings; a new one comes from Google AI Studio.",
        )
    if status == 429:
        header = _retry_after(response)
        wait = header if header is not None else delay
        when = f"; Google says to wait {_in_words(wait)}" if wait is not None else ""
        return SearchFailure(
            429,
            "rate_limited",
            f"Google says this key's project is over its quota or searching too fast "
            f"(HTTP 429{when}): {message}",
            retry_after=wait,
            worth_a_retry=wait is not None,
        )
    if status == 404 and model:
        return SearchFailure(
            502,
            "upstream_error",
            f"Google has no model {model!r} this key can use (HTTP 404). Set `searchModel` "
            "on the search account to one the key lists, or clear it to let the account choose.",
        )
    return SearchFailure(502, "upstream_error", f"Google answered HTTP {status}: {message}")


async def _list_model_ids(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    key: str,
    seconds: float,
    started: float,
) -> list[str]:
    """Every generateContent model the key lists, following `nextPageToken`."""
    ids: list[str] = []
    token: str | None = None
    for _ in range(20):
        params: dict[str, Any] = {"pageSize": 1000}
        if token:
            params["pageToken"] = token
        response = await _get(
            client,
            "Google",
            f"{base}/models",
            params=params,
            headers=headers,
            seconds=_time_left(seconds, started),
        )
        if response.status_code >= 400:
            raise _google_failure(response, key, None)
        try:
            body = response.json()
        except ValueError:
            raise SearchFailure(502, "upstream_error", "Google's model list was not JSON") from None
        if not isinstance(body, dict):
            break
        for entry in body.get("models") or []:
            if not isinstance(entry, dict):
                continue
            name, methods = entry.get("name"), entry.get("supportedGenerationMethods")
            if isinstance(name, str) and isinstance(methods, list) and "generateContent" in methods:
                ids.append(name.removeprefix("models/"))
        token = body.get("nextPageToken")
        if not isinstance(token, str) or not token:
            break
    return ids


async def _model_for(
    client: httpx.AsyncClient,
    account: Account,
    base: str,
    headers: dict[str, str],
    seconds: float,
    started: float,
) -> tuple[str, bool]:
    """The model to search with, and whether it came from the key's listing."""
    if account.search_model:
        return account.search_model.removeprefix("models/"), False
    key = account.api_key or ""
    known = known_model(key, base)
    if known:
        return known, True
    chosen = choose_model(await _list_model_ids(client, base, headers, key, seconds, started))
    if chosen is None:
        raise SearchFailure(
            502,
            "upstream_error",
            "This Gemini key lists no flash-lite or flash model that can search the web. "
            "Set `searchModel` on the search account to a model the key can use.",
        )
    _PICKS[_pick_key(key, base)] = (chosen, time.perf_counter() + MODEL_PICK_SECONDS)
    return chosen, True


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    return parts.scheme, (parts.hostname or "").lower(), parts.port


async def _resolve(client: httpx.AsyncClient, uri: str, seconds: float, base: str) -> str:
    """Google's redirect for a source, followed one step and no further.

    Asks only for the `Location`; the page it names is never fetched. Any
    failure keeps Google's own link, which still opens the source. Only
    Google's redirect host is asked, or the account's own `baseUrl` origin
    (a proxy or a fixture in front of Google): never a host an answer names.
    """
    if (host_of(uri) != _REDIRECT_HOST and _origin(uri) != _origin(base)) or seconds <= 0:
        return uri
    try:
        response = await client.get(uri, follow_redirects=False, timeout=seconds)
    except Exception:
        return uri
    if 300 <= response.status_code < 400:
        location = str(response.headers.get("location", ""))
        parts = urlsplit(location)
        if parts.scheme in ("http", "https") and parts.netloc:
            return location
    return uri


def _candidate_text(candidate: dict[str, Any]) -> str:
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    return "".join(
        p["text"]
        for p in parts or []
        if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")
    )


async def search_google(client: httpx.AsyncClient, account: Account, query: Query) -> Found:
    """One `generateContent` call with the `googleSearch` tool.

    Google has no plain search endpoint for a Gemini key: a model runs the
    searches and answers from them (`google-search-account.md`). What the
    candidate carries is passed on as Google's terms require (GS1): the
    answer's text is never changed, and its Search Suggestions are handed
    on untouched. Shape measured 2026-10-09 against `gemini-3.5-flash-lite`.
    """
    started = time.perf_counter()
    base = (account.base_url or GOOGLE_URL).rstrip("/")
    key = account.api_key or ""
    headers = {"x-goog-api-key": key}
    model, from_listing = await _model_for(client, account, base, headers, query.timeout, started)
    instruction = f"Search the web for: {with_site(query.query, query.allowed)}. "
    instruction += "Answer briefly from what you find."
    if query.language:
        instruction += f" Answer in {query.language}."
    response = await _post(
        client,
        "Google",
        f"{base}/models/{model}:generateContent",
        json={
            "contents": [{"role": "user", "parts": [{"text": instruction}]}],
            "tools": [{"googleSearch": {}}],
        },
        headers=headers,
        seconds=_time_left(query.timeout, started),
    )
    if response.status_code >= 400:
        if response.status_code == 404 and from_listing:
            _PICKS.pop(_pick_key(key, base), None)
        raise _google_failure(response, key, model)
    try:
        body = response.json()
    except ValueError:
        raise SearchFailure(
            502, "upstream_error", "Google answered something that is not JSON"
        ) from None
    candidates = body.get("candidates") if isinstance(body, dict) else None
    candidate = candidates[0] if isinstance(candidates, list) and candidates else None
    candidate = candidate if isinstance(candidate, dict) else {}
    meta = candidate.get("groundingMetadata")
    meta = meta if isinstance(meta, dict) else {}
    chunks = meta.get("groundingChunks")
    ignored = ["userLocation"] if query.located else []
    if not isinstance(chunks, list) or not chunks:
        # Google ran no search. Only the reason is logged, never the model's words.
        log.debug("Google grounded nothing (finishReason=%s)", candidate.get("finishReason"))
        return Found(hits=[], answer=None, ignored=ignored)

    snippets: dict[int, list[str]] = {}
    for support in meta.get("groundingSupports") or []:
        segment = support.get("segment") if isinstance(support, dict) else None
        text = segment.get("text") if isinstance(segment, dict) else None
        if not isinstance(text, str) or not text:
            continue
        for index in support.get("groundingChunkIndices") or []:
            if isinstance(index, int):
                bucket = snippets.setdefault(index, [])
                # Google's segments can repeat one another (a later one
                # starting with an earlier one's text): keep the longer.
                if any(text in kept for kept in bucket):
                    continue
                bucket[:] = [kept for kept in bucket if kept not in text]
                bucket.append(text)

    sources: list[tuple[str, str, str]] = []
    for index, chunk in enumerate(chunks):
        web = chunk.get("web") if isinstance(chunk, dict) else None
        uri = web.get("uri") if isinstance(web, dict) else None
        if not isinstance(web, dict) or not isinstance(uri, str) or not uri:
            continue
        title = web.get("title")
        sources.append(
            (
                uri,
                title if isinstance(title, str) and title else uri,
                " ".join(snippets.get(index, [])),
            )
        )
    seconds = min(REDIRECT_SECONDS, query.timeout - (time.perf_counter() - started))
    urls = await asyncio.gather(*(_resolve(client, uri, seconds, base) for uri, _, _ in sources))
    hits = [
        Hit(url=url, title=title, snippet=trim(snippet, query.context_size))
        for url, (_, title, snippet) in zip(urls, sources, strict=True)
    ]
    # Google's terms forbid editing what it grounded, so when the request
    # filters domains (the answer may lean on pages the filter removes) the
    # answer is left out rather than trimmed (GS8).
    filtered = bool(query.allowed or query.blocked)
    hits = keep(hits, query.allowed, query.blocked)
    answer = None if filtered else (_candidate_text(candidate) or None)
    entry = meta.get("searchEntryPoint")
    rendered = entry.get("renderedContent") if isinstance(entry, dict) else None
    return Found(
        hits=hits,
        answer=answer,
        ignored=ignored,
        search_suggestions=rendered if isinstance(rendered, str) and rendered else None,
    )


PROVIDERS: dict[str, Provider] = {
    "searxng": Provider(
        key="searxng",
        label="SearXNG",
        default_base_url=None,
        needs_key=False,
        probes=True,
        search=search_searxng,
        billing="free",
    ),
    "brave": Provider(
        key="brave",
        label="Brave Search",
        default_base_url=BRAVE_URL,
        needs_key=True,
        probes=False,
        search=search_brave,
        # One search a second is what a Brave free key allows; a free key
        # met it live on 2026-10-01 (tool-driver#3). Brave stopped issuing
        # free keys on 2026-02-12, but keys issued before still search, so
        # their pace stays the default: it is the one every key accepts. A
        # prepaid key reports 50 a second (a third party's header capture,
        # nicobailon/pi-web-access#501; there is no Brave key on the
        # development box), and its owner can set the gap lower.
        search_interval=1.0,
        search_interval_reason=(
            "the pace a Brave free key allows. Brave stopped issuing free keys on "
            "2026-02-12, and a prepaid key allows more: set a shorter wait, or 0, for one."
        ),
    ),
    "google": Provider(
        key="google",
        label="Google Search (Gemini API key)",
        default_base_url=GOOGLE_URL,
        needs_key=True,
        # Every search is billed (a query the model runs costs, past the
        # free allowance), so nothing asks one on a timer.
        probes=False,
        search=search_google,
        billing="per_search",
        search_interval=0.0,
        search_interval_reason=(
            "because Google limits a Gemini key by its project's quota, not by a pace "
            "between searches."
        ),
    ),
}


def get_provider(key: str) -> Provider:
    try:
        return PROVIDERS[key]
    except KeyError:
        raise SearchFailure(
            503,
            "not_configured",
            f"unknown search provider {key!r}; choose one of {list(PROVIDERS)}",
        ) from None
