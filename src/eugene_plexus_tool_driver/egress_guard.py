"""Where a remote MCP call may connect (row 59b, call R59-MCP).

An API client names the MCP server on its Responses request, so this
process dials an address a stranger chose, from wherever the hub sits:
beside the owner's NAS, router admin page and cloud metadata service.
This module is the one rule about that, applied to **every connection**
an MCP client makes, a redirect's included:

1. The host is resolved **once**, here.
2. **Every** address it resolves to is classified, and one the reach does
   not allow refuses the whole connection: a name that answers a public
   address and a private one is not half allowed.
3. The connection goes to an address that was checked, as a literal, so
   nothing resolves the name a second time. DNS rebinding (a first answer
   that passes, a second that points inside) has no second answer to use.
   TLS still checks the certificate against the name: httpcore hands the
   original host to `start_tls` as the server name.

The classes, measured against Python 3.12's `ipaddress` (whose `is_global`
alone is not a safe predicate: it is True for `224.0.0.1`, `ff02::1`,
`fec0::1` and `64:ff9b::7f00:1`, NAT64's form of 127.0.0.1):

* `NEVER`: the unspecified address, link-local (169.254.0.0/16 and
  fe80::/10, where cloud metadata services listen), multicast, reserved
  ranges (240.0.0.0/4, the broadcast address, IPv4-compatible IPv6),
  and the metadata addresses outside link-local (Alibaba's
  100.100.100.200, AWS's fd00:ec2::254). Refused under every reach.
* `LOOPBACK`: this machine. Refused under every reach too: this install's
  own services listen there, and some of them trust a loopback caller.
* `NETWORK`: the owner's network -- RFC 1918, unique-local and site-local
  IPv6, the carrier range 100.64.0.0/10 (where a tailnet lives), and the
  other ranges `is_global` says are not on the internet.
* `GLOBAL`: the internet.

An IPv6 address that carries an IPv4 one is judged by that IPv4 address
too, and by the stricter of the two: IPv4-mapped (`::ffff:a.b.c.d`, which a
dual-stack socket sends to `a.b.c.d`), NAT64 (`64:ff9b::/96`), 6to4
(`2002::/16`) and Teredo's client address.

`internet` (the default) allows `GLOBAL`; `network` allows `GLOBAL` and
`NETWORK` over https, and `NETWORK` only over plain http.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import anyio
import httpcore

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

GLOBAL, NETWORK, LOOPBACK, NEVER = 0, 1, 2, 3

#: Metadata services that are not link-local. Refused under every reach.
_METADATA = (
    ipaddress.ip_network("100.100.100.200/32"),
    ipaddress.ip_network("fd00:ec2::254/128"),
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

#: What a reach allows, by scheme. `internet` has no `http` entry: a plain
#: request to the internet would carry the caller's credential in the clear.
ALLOWED: dict[tuple[str, str], frozenset[int]] = {
    ("internet", "https"): frozenset({GLOBAL}),
    ("network", "https"): frozenset({GLOBAL, NETWORK}),
    ("network", "http"): frozenset({NETWORK}),
}


class AddressRefused(httpcore.ConnectError):
    """A connection the reach does not allow. `reason` names the rule and the
    address it found, in words a caller can act on."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _plain_kind(address: IPAddress) -> tuple[int, str]:
    if any(address in network for network in _METADATA):
        return NEVER, "a cloud metadata service's address"
    if address.is_unspecified:
        return NEVER, "the unspecified address"
    if address.is_link_local:
        return NEVER, "link-local, where cloud metadata services listen"
    if address.is_multicast:
        return NEVER, "a multicast group"
    if address.is_loopback:
        return LOOPBACK, "this machine's loopback"
    if address.is_reserved:
        return NEVER, "in a reserved range"
    if isinstance(address, ipaddress.IPv6Address) and address.is_site_local:
        return NETWORK, "on a private network"
    if address.is_global:
        return GLOBAL, "on the internet"
    return NETWORK, "on a private network"


def _carried(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 one carries, for the forms that deliver to it."""
    if address in _NAT64:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    return None


def classify(address: IPAddress) -> tuple[int, str]:
    """`(kind, words)`: which class an address is in, and why, for a message."""
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None:
            # A dual-stack socket sends this to the IPv4 address itself.
            kind, words = _plain_kind(mapped)
            return kind, f"{words} (it is {mapped}, written as IPv6)"
        packed = int(address)
        if 1 < packed < 2**32:
            return NEVER, "an IPv4-compatible IPv6 address, a form no host uses"
        inner = _carried(address)
        if inner is not None:
            if address in _NAT64:
                kind, words = _plain_kind(inner)
                return kind, f"{words} (NAT64 delivers it to {inner})"
            outer_kind, outer_words = _plain_kind(address)
            inner_kind, inner_words = _plain_kind(inner)
            if inner_kind > outer_kind:
                return inner_kind, f"{inner_words} (it carries {inner})"
            return outer_kind, outer_words
    return _plain_kind(address)


def refusal(address: IPAddress, reach: str, scheme: str) -> str | None:
    """Why `reach` may not connect to `address` over `scheme`, or None."""
    allowed = ALLOWED.get((reach, scheme))
    if allowed is None:
        return f"{scheme} is not allowed under the reach {reach!r}"
    kind, words = classify(address)
    if kind in allowed:
        return None
    if kind in (NEVER, LOOPBACK):
        return f"{address} is {words}, and a remote MCP server is never reached there"
    if kind == NETWORK:
        return (
            f"{address} is {words}, and this install lets API clients reach MCP servers on "
            "the internet only (the owner's Remote MCP setting)"
        )
    # GLOBAL refused: plain http under `network` reach.
    return (
        f"{address} is {words}, and plain http is allowed only to addresses on this "
        "network: use https"
    )


Resolver = Callable[[str, int], Awaitable[list[IPAddress]]]


async def resolve(host: str, port: int) -> list[IPAddress]:
    """Every address `host` resolves to, in the resolver's order, once each."""
    infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    found: list[IPAddress] = []
    for _family, _type, _proto, _name, sockaddr in infos:
        text = str(sockaddr[0]).split("%", 1)[0]
        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            continue
        if address not in found:
            found.append(address)
    return found


class GuardedBackend(httpcore.AsyncNetworkBackend):
    """httpcore's network backend, with the check before every connection.

    One per (reach, scheme): a connection pool keyed by origin must never
    hand a connection checked under one rule to a request under another.
    `resolver` and `inner` are the test seams: a name answered as a test
    says, and a connector that records the literal it was given.
    """

    def __init__(
        self,
        reach: str,
        scheme: str,
        *,
        resolver: Resolver | None = None,
        inner: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self.reach, self.scheme = reach, scheme
        self._resolve = resolver or resolve
        self._inner = inner or httpcore.AnyIOBackend()

    async def _addresses(self, host: str, port: int, seconds: float | None) -> list[IPAddress]:
        try:
            with anyio.fail_after(seconds):
                return await self._resolve(host, port)
        except TimeoutError:
            raise httpcore.ConnectTimeout(f"{host} did not resolve in time") from None
        except OSError as exc:
            raise httpcore.ConnectError(f"{host} did not resolve ({exc.strerror or exc})") from None

    def checked(self, host: str, addresses: Iterable[IPAddress]) -> list[IPAddress]:
        """`addresses`, if every one is allowed; else `AddressRefused` naming one."""
        kept = list(addresses)
        if not kept:
            raise httpcore.ConnectError(f"{host} resolved to no address")
        for address in kept:
            why = refusal(address, self.reach, self.scheme)
            if why is not None:
                named = (
                    "" if host.strip("[]") == str(address) else f"{host} resolves to {address}: "
                )
                raise AddressRefused(f"{named}{why}")
        return kept

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = self.checked(host, await self._addresses(host, port, timeout))
        last: Exception | None = None
        for address in addresses:
            try:
                # A literal: nothing below this line looks the name up again.
                return await self._inner.connect_tcp(
                    str(address), port, timeout, local_address, socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last = exc
        assert last is not None
        raise last

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        raise AddressRefused("a remote MCP server is never reached over a Unix socket")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def refused_reason(exc: BaseException) -> str | None:
    """The rule an httpx error came from, when the guard refused it."""
    seen: BaseException | None = exc
    for _ in range(8):
        if seen is None:
            return None
        if isinstance(seen, AddressRefused):
            return seen.reason
        seen = seen.__cause__ or seen.__context__
    return None
