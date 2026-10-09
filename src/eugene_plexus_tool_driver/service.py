"""One search, end to end: the account's settings, the provider, the filter.

Also the account's health. `/healthz` must never run a search per read
-- a paid provider counts every one -- so health is the outcome of the
most recent search, the operator's Test included, and for a free SearXNG
a probe search on a timer so a broken instance is known before a model
needs it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ._generated.models import WebSearchRequest, WebSearchResponse, WebSearchResult
from ._http import egress_client, internal_client, is_internal
from .config import ConfigStore
from .pacing import Clock, Pacer, Sleep
from .providers import Account, Provider, Query, get_provider
from .search import Found, SearchFailure, keep

log = logging.getLogger(__name__)

#: The fixed query a Test and a probe search for. Something with results
#: everywhere and nothing personal in it.
PROBE_QUERY = "Apache License 2.0"


def _strings(items: Any) -> list[str] | None:
    """A generated list of constrained strings as plain strings.

    Each entry is a pydantic `RootModel` (the schema bounds its length),
    and a `RootModel` is not a `str`: handed to the domain filter as it
    is, no host would ever match it.
    """
    out = [str(getattr(item, "root", item)) for item in items or []]
    return out or None


@dataclass
class Outcome:
    ok: bool
    at: float
    error: str | None = None
    code: str | None = None


class SearchService:
    def __init__(
        self,
        store: ConfigStore,
        *,
        transports: tuple[httpx.AsyncBaseTransport, httpx.AsyncBaseTransport] | None = None,
        timing: tuple[Clock, Sleep] | None = None,
    ) -> None:
        self.store = store
        # Two clients for the life of the process, never one per search
        # (R1.1): a SearXNG on the LAN is dialled without the user's proxy,
        # a public provider through it -- that is how those users reach the
        # internet at all. `transports` (internal, egress) is the test seam.
        internal_transport, egress_transport = transports or (None, None)
        self._internal = internal_client(
            timeout=httpx.Timeout(30.0, connect=5.0), transport=internal_transport
        )
        self._egress = egress_client(
            timeout=httpx.Timeout(30.0, connect=10.0), transport=egress_transport
        )
        # `timing` (clock, sleep) is the test seam for waits measured in seconds.
        self._clock, self._sleep = timing or (time.perf_counter, asyncio.sleep)
        self._pacer = Pacer(self._clock, self._sleep)
        self.last: Outcome | None = None
        self._probe_task: asyncio.Task[None] | None = None

    # -- the account -------------------------------------------------------

    def resolve(self, overrides: dict[str, Any] | None = None) -> tuple[Provider, Account]:
        values = self.store.snapshot()
        values.update(overrides or {})
        provider = get_provider(str(values.get("provider") or "searxng"))
        base_url = str(values.get("baseUrl") or provider.default_base_url or "").strip()
        api_key = values.get("apiKey")
        model = values.get("searchModel")
        account = Account(
            base_url=base_url,
            api_key=api_key if isinstance(api_key, str) else None,
            search_model=model.strip() if isinstance(model, str) and model.strip() else None,
        )
        return provider, account

    def configured(self) -> bool:
        try:
            provider, account = self.resolve()
        except SearchFailure:
            return False
        return provider.missing(account) is None

    def provider_key(self) -> str:
        return str(self.store.get("provider") or "searxng")

    # -- one search --------------------------------------------------------

    async def web_search(
        self, request: WebSearchRequest, *, overrides: dict[str, Any] | None = None
    ) -> WebSearchResponse:
        started = time.perf_counter()
        values = self.store.snapshot()
        values.update(overrides or {})
        provider, account = self.resolve(overrides)
        missing = provider.missing(account)
        if missing is not None:
            raise SearchFailure(
                503, "not_configured", f"This search account is not set up: {missing}."
            )
        max_results = request.maxResults or int(values.get("maxResults") or 5)
        location = request.userLocation.approximate if request.userLocation else None
        query = Query(
            query=request.query,
            max_results=max_results,
            allowed=_strings(request.allowedDomains),
            blocked=_strings(request.blockedDomains),
            country=location.country if location is not None else None,
            context_size=request.contextSize.value if request.contextSize else None,
            safe_search=str(values.get("safeSearch") or "moderate"),
            language=str(values["language"]) if values.get("language") else None,
            timeout=float(values.get("timeoutSeconds") or 15),
            located=request.userLocation is not None,
        )
        configured = values.get("searchIntervalSeconds")
        interval = provider.search_interval if configured is None else float(configured)
        client = self._internal if is_internal(account.base_url) else self._egress
        try:
            found = await self._paced(provider, client, account, query, interval)
        except SearchFailure as failure:
            self.last = Outcome(False, time.time(), failure.detail, failure.code)
            raise
        hits = keep(found.hits, query.allowed, query.blocked)[:max_results]
        self.last = Outcome(True, time.time())
        return WebSearchResponse(
            results=[
                WebSearchResult(
                    url=h.url, title=h.title, snippet=h.snippet or None, publishedAt=h.published_at
                )
                for h in hits
            ],
            answer=found.answer,
            provider=provider.key,
            latencyMs=round((time.perf_counter() - started) * 1000, 1),
            ignored=found.ignored or None,
            searchSuggestions=found.search_suggestions,
        )

    async def _paced(
        self,
        provider: Provider,
        client: httpx.AsyncClient,
        account: Account,
        query: Query,
        interval: float,
    ) -> Found:
        """The provider's search in this account's turn, retried once when
        the provider says a short wait will do (tool-driver#3).

        The whole search -- its turn, the gap before it, the request and
        the one retry -- lives inside `timeoutSeconds`, because that is how
        long the model waits on it. The retry waits at least `interval`
        too, so it never lands closer to the refusal than any search would.
        """
        deadline = self._clock() + query.timeout
        async with self._pacer.turn(interval, deadline):
            sent = self._clock()
            try:
                return await provider.search(client, account, self._with_time_left(query, deadline))
            except SearchFailure as failure:
                if not failure.worth_a_retry:
                    raise
                wait = max(failure.retry_after or 0.0, interval)
                # Room for the retry's own answer, as long as the refusal took.
                took = self._clock() - sent
                if self._clock() + wait + took >= deadline:
                    raise
                log.info(
                    "%s refused a search as too soon; trying once more in %gs",
                    provider.label,
                    wait,
                )
                await self._sleep(wait)
            try:
                return await provider.search(client, account, self._with_time_left(query, deadline))
            except SearchFailure as again:
                # Whatever the second answer was -- refused again, the month
                # now used up, no answer in the time left -- it was the second.
                again.detail = (
                    f"{again.detail.rstrip('.')}. It was tried once more after {wait:g}s."
                )
                raise

    def _with_time_left(self, query: Query, deadline: float) -> Query:
        return dataclasses.replace(query, timeout=deadline - self._clock())

    # -- health ------------------------------------------------------------

    def start_probe(self) -> None:
        if self._probe_task is None:
            self._probe_task = asyncio.create_task(self._probe_loop(), name="search-probe")

    async def _probe_loop(self) -> None:
        # A first check soon after start, then on the configured interval.
        await asyncio.sleep(2)
        while True:
            minutes = int(self.store.get("probeMinutes") or 0)
            try:
                provider, _ = self.resolve()
                if minutes > 0 and provider.probes and self.configured():
                    await self.web_search(WebSearchRequest(query=PROBE_QUERY, maxResults=1))
            except SearchFailure as failure:
                log.warning("search account check failed: %s", failure.detail)
            except Exception:  # never let the probe take the process down
                log.exception("search account check raised")
            await asyncio.sleep(max(60, minutes * 60) if minutes > 0 else 60)

    async def aclose(self) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._probe_task
        for client in (self._internal, self._egress):
            with contextlib.suppress(Exception):
                await client.aclose()
