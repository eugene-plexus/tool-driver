"""One account's searches, spaced so its provider does not refuse them.

Brave's free plan allows one search a second, and the gateway sends a
turn's searches one after another with no gap (tool-driver#3), so the
second reached Brave inside the same second and was refused. The gap is
kept here rather than in the gateway's loop because every search on an
account passes through this process, two chats' searches included.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from .search import SearchFailure

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class Pacer:
    """Searches take turns, `interval` seconds from one answer to the next request.

    Measured from the answer rather than from the request: a request's
    arrival at the provider is somewhere between the two, so spacing the
    starts by a second can still put two arrivals inside one. An interval
    of 0 takes no turns at all, and searches run side by side as before.
    """

    def __init__(self, clock: Clock, sleep: Sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last: float | None = None

    @asynccontextmanager
    async def turn(self, interval: float, deadline: float) -> AsyncIterator[None]:
        if interval <= 0:
            try:
                yield
            finally:
                self._last = self._clock()
            return
        async with self._lock:
            gap = 0.0 if self._last is None else self._last + interval - self._clock()
            if self._clock() + max(gap, 0.0) >= deadline:
                # Sent now, it would have no time left to be answered in.
                raise SearchFailure(
                    429,
                    "rate_limited",
                    f"This search account leaves {interval:g}s between searches, and this "
                    "search's turn did not come within its timeout.",
                    retry_after=max(gap, interval),
                )
            if gap > 0:
                await self._sleep(gap)
            try:
                yield
            finally:
                self._last = self._clock()
