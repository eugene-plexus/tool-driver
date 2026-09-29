"""The search providers: SearXNG (free, self-hosted) and Brave (keyed).

Design call #4 (`server-run-tools.md` §9). Each provider turns the one
`WebSearchRequest` into its own wire request and its answer back into
`search.Found`; everything they share -- filtering, excerpt length, the
`site:` hint -- lives in `search.py` so the two cannot disagree about it.

A provider's failure is raised as `search.SearchFailure` carrying the
status this component answers and a sentence naming the cause, because
that sentence is what the model is handed and then what a person reads.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .search import Found, Hit, SearchFailure, plain, trim, with_site

log = logging.getLogger(__name__)

BRAVE_URL = "https://api.search.brave.com"

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
    timeout: float


@dataclass(frozen=True)
class Account:
    """The account's saved settings, read when a search runs."""

    base_url: str
    api_key: str | None


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
        raise SearchFailure(504, "timeout", f"{who} did not answer within {seconds:g}s") from None
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
        raise SearchFailure(
            429,
            "rate_limited",
            "Brave Search says this key is over its rate or monthly limit (HTTP 429).",
            retry_after=_retry_after(response),
        )
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
