"""Web search providers: one request shape in, one answer shape out.

A provider owns its own wire protocol and its quirks; the gateway sees
`WebSearchResponse` and nothing else. Two providers today, chosen by
design call #4 (`server-run-tools.md` §9): **SearXNG**, self-hosted and
keyless, **Brave**, the first keyed one, and **Google** (a Gemini key; the
model runs the search). Each was read against the real thing where this box
could reach it:

* SearXNG, measured 2026-09-29 against an instance started from
  `searxng/searxng@4e2c1ea` in WSL2 with `search.formats: [html, json]`:
  `GET /search?q=…&format=json` answers `results` (each with `url`,
  `title`, `content`, `publishedDate`, `engines`, `score`), `answers`,
  `infoboxes`, `suggestions`, `corrections` and `unresponsive_engines`
  (pairs, e.g. `["duckduckgo", "CAPTCHA"]`), and **no**
  `number_of_results`. A format that is not enabled is **403 with an
  HTML body**, so the one message a SearXNG user needs — switch JSON on
  — has to be inferred from that status. No query is 400
  `{"error": "No query"}`.
* Brave, from its API documentation (no key on this box):
  `GET https://api.search.brave.com/res/v1/web/search` with
  `X-Subscription-Token`; `count` at most 20; `web.results[]` with
  `title`, `url`, `description`, `age`, `page_age`, `extra_snippets`;
  401 for a bad key, 422 for bad parameters, 429 when rate limited, with
  the limits in `X-RateLimit-*` headers (`providers._brave_limited`).
* Google, measured 2026-10-09 with a Gemini key: `generateContent` with the
  `googleSearch` tool answers text plus `groundingMetadata`
  (`groundingChunks`, `groundingSupports`, `searchEntryPoint`); see
  `providers.search_google` and `docs/design/google-search-account.md`.

**Filtering is ours, always.** A provider may be told about allowed
domains (a `site:` operator when there is exactly one), but what comes
back is filtered here whatever it did, so a result outside the allowed
list never reaches a model.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

#: Snippet length per `contextSize`.
SNIPPET_CHARS = {"low": 240, "medium": 600, "high": 1600}

_TAG = re.compile(r"<[^>]+>")


class SearchFailure(Exception):
    """A provider failed, with the status the tool-driver answers and why.

    `code` is the problem's machine name (`upstream_auth_error`,
    `rate_limited`, `quota_exhausted`, `json_disabled`, `upstream_error`,
    `timeout`, `not_configured`); `retry_after` is how long the provider
    asked us to wait, in seconds, when it said. `worth_a_retry` is the
    provider saying a short wait will do: refused for coming too soon,
    not for a quota that is used up.
    """

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        retry_after: float | None = None,
        *,
        worth_a_retry: bool = False,
    ):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.retry_after = retry_after
        self.worth_a_retry = worth_a_retry


@dataclass
class Hit:
    url: str
    title: str
    snippet: str = ""
    published_at: str | None = None


@dataclass
class Found:
    hits: list[Hit]
    answer: str | None = None
    ignored: list[str] = field(default_factory=list)
    #: HTML the provider's terms require shown with the results (Google's
    #: Search Suggestions), verbatim; None for providers with none.
    search_suggestions: str | None = None


def plain(text: Any) -> str:
    """A provider's excerpt as plain text: tags gone, entities decoded, spaces collapsed.

    Brave marks the query's words with `<strong>`; a model given that
    reads markup, not the page.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(html.unescape(_TAG.sub("", text)).split())


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def domain_matches(host: str, domain: str) -> bool:
    """A domain matches itself and its subdomains, never a suffix that is not one.

    `example.com` matches `docs.example.com` and not `notexample.com`. A
    domain given with a scheme or a path (`https://example.com/docs`) is
    read for its host, since that is what a person pastes.
    """
    wanted = domain.strip().lower()
    if "://" in wanted:
        wanted = host_of(wanted)
    wanted = wanted.split("/", 1)[0].lstrip("*.").rstrip(".")
    if not wanted or not host:
        return False
    return host == wanted or host.endswith("." + wanted)


def keep(hits: list[Hit], allowed: list[str] | None, blocked: list[str] | None) -> list[Hit]:
    kept = []
    for hit in hits:
        host = host_of(hit.url)
        if allowed and not any(domain_matches(host, d) for d in allowed):
            continue
        if blocked and any(domain_matches(host, d) for d in blocked):
            continue
        kept.append(hit)
    return kept


def with_site(query: str, allowed: list[str] | None) -> str:
    """The query with a `site:` operator when exactly one domain is allowed.

    A hint only — both providers honour the operator — so that the
    filter afterwards has something to keep. Two or more domains are not
    joined with `OR`: engines disagree on that syntax, and a query the
    engine misreads returns nothing to filter.
    """
    if allowed and len(allowed) == 1:
        domain = allowed[0].strip()
        if "://" in domain:
            domain = host_of(domain)
        domain = domain.split("/", 1)[0]
        if domain and "site:" not in query:
            return f"{query} site:{domain}"
    return query


def trim(text: str, context_size: str | None) -> str:
    limit = SNIPPET_CHARS.get(context_size or "medium", SNIPPET_CHARS["medium"])
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"
