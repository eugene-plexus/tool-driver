"""Remote MCP, the tool driver's half (row 59b).

An API client names an MCP server on a Responses request; the gateway runs
the loop and asks this process to list the server's tools and to call one.
Each of those is one short session over MCP's Streamable HTTP transport:
`initialize`, `notifications/initialized`, then `tools/list` (page by page)
or `tools/call`, and a `DELETE` that ends the session. Nothing is kept
between two calls.

The rules, each for a reason:

* **Where it may connect** is `egress_guard`'s: every connection, every
  address the name resolves to, the checked address dialled as a literal.
  The user's proxy is never used, because through a proxy the proxy would
  resolve the name and the check would mean nothing.
* **A redirect** is followed only as a 307 or 308 to the same origin: the
  caller's headers go to the server it named and to no other. Anything
  else is a failure that names the redirect.
* **The caller's headers** are sent with each request of the call and
  then dropped. They are never logged, and every text that leaves this
  module -- an error's words, a tool's description, a tool's output -- has
  each header value (eight characters or more) replaced, so a server that
  echoes a token cannot send it on to a log, a metrics row or a stored
  response. An error names the server by its origin only: a path or a
  query can carry a key too.
* **Limits**, each named when hit: 30 s for the whole call, 8 MiB for each
  answer, 64 tools after `allowedTools`, 32 pages of tools. Answers are
  asked for uncompressed and read raw, so the 8 MiB counts what arrives.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpcore
import httpx

from . import __version__
from ._http import internal_client, ssl_context
from .egress_guard import ALLOWED, GuardedBackend, Resolver, refused_reason
from .search import SearchFailure

log = logging.getLogger(__name__)

LIMIT_SECONDS = 30.0
LIMIT_BYTES = 8 * 1024 * 1024
LIMIT_TOOLS = 64
LIMIT_PAGES = 32
LIMIT_REDIRECTS = 3
LIMIT_HEADERS = 32
LIMIT_HEADER_VALUE = 8192
#: How long ending the session may take; it is a courtesy to the server.
END_SECONDS = 2.0
PROTOCOL_VERSION = "2025-06-18"
#: A header value shorter than this is not scrubbed from text: `X-Version: 2`
#: would otherwise turn every 2 in a tool's output into a placeholder.
SCRUB_MIN = 8

#: Headers that frame the request, and so are this module's, never a caller's.
FRAMING = frozenset(
    {
        "host",
        "content-length",
        "content-type",
        "content-encoding",
        "accept",
        "accept-encoding",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "expect",
        "proxy-authorization",
        "proxy-authenticate",
        "proxy-connection",
        "mcp-session-id",
        "mcp-protocol-version",
        "last-event-id",
    }
)
_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]{1,128}$")
_TOOL_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")


class McpFailure(SearchFailure):
    """A remote MCP call that failed, with the status this component answers,
    a machine name for the cause and the words a person reads."""


# --------------------------------------------------------------------------- #
# The call's inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Target:
    """The server as the caller named it, checked before anything is dialled."""

    url: str
    scheme: str
    origin: str


def target(server_url: str, reach: str) -> Target:
    """The server URL, refused before any lookup when its form is wrong."""
    try:
        url = httpx.URL(server_url.strip())
    except (httpx.InvalidURL, ValueError):
        raise McpFailure(400, "invalid_url", "server_url is not a web address") from None
    scheme = url.scheme.lower()
    if scheme not in ("http", "https") or not url.host:
        raise McpFailure(400, "invalid_url", "server_url must be an https:// address with a host")
    if url.userinfo:
        raise McpFailure(
            400,
            "invalid_url",
            "server_url carries a user name or password; send credentials in headers or "
            "authorization instead",
        )
    host = f"[{url.host}]" if ":" in url.host else url.host
    origin = f"{scheme}://{host}" + (f":{url.port}" if url.port else "")
    if (reach, scheme) not in ALLOWED:
        if reach == "internet" and scheme == "http":
            raise McpFailure(
                400,
                "scheme_refused",
                f"{origin} is plain http; a remote MCP server is reached over https, so the "
                "request's credentials are never sent in the clear",
            )
        raise McpFailure(400, "invalid_reach", f"unknown reach {reach!r}")
    return Target(url=str(url.copy_with(fragment=None)), scheme=scheme, origin=origin)


def caller_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """The caller's headers, checked. A refusal names the header, never its value."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in (headers or {}).items():
        if len(out) >= LIMIT_HEADERS:
            raise McpFailure(400, "invalid_header", f"more than {LIMIT_HEADERS} headers")
        if not isinstance(name, str) or not _TOKEN.match(name):
            raise McpFailure(400, "invalid_header", "a header name is not a valid HTTP token")
        lowered = name.lower()
        if lowered in FRAMING:
            raise McpFailure(
                400,
                "invalid_header",
                f"headers.{name}: Eugene sets this header itself; it cannot come from a request",
            )
        if lowered in seen:
            raise McpFailure(400, "invalid_header", f"headers.{name}: given twice")
        if not isinstance(value, str) or any(c in value for c in "\r\n\x00"):
            raise McpFailure(400, "invalid_header", f"headers.{name}: not a single-line value")
        if len(value) > LIMIT_HEADER_VALUE:
            raise McpFailure(
                400,
                "invalid_header",
                f"headers.{name}: longer than {LIMIT_HEADER_VALUE} characters",
            )
        seen.add(lowered)
        out[name] = value
    return out


@dataclass
class Scrubber:
    """Takes the caller's header values out of any text that leaves this module."""

    secrets: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, headers: Mapping[str, str]) -> Scrubber:
        found: set[str] = set()
        for value in headers.values():
            parts = [value, *value.split(None, 1)[1:]]  # "Bearer xyz" -> also "xyz"
            found.update(p.strip() for p in parts if len(p.strip()) >= SCRUB_MIN)
        return cls(sorted(found, key=len, reverse=True))

    def text(self, value: str) -> str:
        for secret in self.secrets:
            if secret in value:
                value = value.replace(secret, "[redacted]")
        return value

    def deep(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.deep(v) for v in value]
        if isinstance(value, dict):
            return {self.text(str(k)): self.deep(v) for k, v in value.items()}
        return value


# --------------------------------------------------------------------------- #
# One session
# --------------------------------------------------------------------------- #


@dataclass
class _Answer:
    message: dict[str, Any] | None
    session_id: str | None


class _Session:
    """One MCP session: JSON-RPC over POST, answered as JSON or as an event stream."""

    def __init__(
        self, client: httpx.AsyncClient, where: Target, headers: dict[str, str], scrub: Scrubber
    ) -> None:
        self.client = client
        self.where = where
        self.url = where.url
        self.headers = headers
        self.scrub = scrub
        self.session_id: str | None = None
        self.version: str | None = None
        self._next_id = 0

    def _fail(self, status: int, code: str, words: str) -> McpFailure:
        return McpFailure(
            status, code, self.scrub.text(f"The MCP server at {self.where.origin} {words}")
        )

    def _request_headers(self) -> dict[str, str]:
        headers = {
            **self.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "identity",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.version:
            headers["MCP-Protocol-Version"] = self.version
        return headers

    async def open(self) -> None:
        result = await self.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "eugene-plexus", "version": __version__},
            },
        )
        version = result.get("protocolVersion")
        self.version = version if isinstance(version, str) and version else PROTOCOL_VERSION
        await self.notify("notifications/initialized")

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        ident = self._next_id
        answer = await self._post(
            {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}, ident
        )
        if answer.session_id and self.session_id is None:
            self.session_id = answer.session_id
        message = answer.message
        if message is None:
            raise self._fail(502, "upstream_error", f"sent no answer to {method}")
        error = message.get("error")
        if isinstance(error, dict):
            words = str(error.get("message") or "no message")[:300]
            raise self._fail(
                502, "server_error", f"answered {method} with error {error.get('code')}: {words}"
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise self._fail(502, "upstream_error", f"answered {method} with no result")
        return result

    async def notify(self, method: str) -> None:
        await self._post({"jsonrpc": "2.0", "method": method}, None)

    async def end(self) -> None:
        """Ends the session, as the transport asks; any failure is the server's."""
        if not self.session_id:
            return
        with contextlib.suppress(Exception):
            async with asyncio.timeout(END_SECONDS):
                await self.client.delete(self.url, headers=self._request_headers())

    async def _post(self, payload: dict[str, Any], ident: int | None) -> _Answer:
        url = self.url
        for _ in range(LIMIT_REDIRECTS + 1):
            try:
                async with self.client.stream(
                    "POST", url, json=payload, headers=self._request_headers()
                ) as response:
                    if response.status_code in (307, 308):
                        url = self._same_origin(url, response)
                        continue
                    answer = await self._read(response, ident)
                    # Later requests of this session go where the redirect ended.
                    self.url = url
                    return answer
            except httpx.TimeoutException:
                raise self._fail(504, "timeout", "did not answer in time") from None
            except httpx.HTTPError as exc:
                refused = refused_reason(exc)
                if refused is not None:
                    raise McpFailure(403, "address_refused", self.scrub.text(refused)) from None
                raise self._fail(
                    502, "unreachable", f"could not be reached ({type(exc).__name__})"
                ) from None
        raise self._fail(502, "redirect_refused", f"redirected more than {LIMIT_REDIRECTS} times")

    def _same_origin(self, url: str, response: httpx.Response) -> str:
        location = response.headers.get("location") or ""
        try:
            moved = httpx.URL(urljoin(url, location))
        except (httpx.InvalidURL, ValueError):
            moved = None
        here = httpx.URL(url)
        if moved is None or (moved.scheme, moved.host, moved.port) != (
            here.scheme,
            here.host,
            here.port,
        ):
            raise self._fail(
                502,
                "redirect_refused",
                f"redirected (HTTP {response.status_code}) to another server; Eugene sends "
                "the request's headers only to the server it names, so give the address the "
                "redirect ends at",
            )
        return str(moved.copy_with(fragment=None))

    async def _read(self, response: httpx.Response, ident: int | None) -> _Answer:
        status = response.status_code
        session_id = response.headers.get("mcp-session-id")
        if 300 <= status < 400:
            raise self._fail(
                502,
                "redirect_refused",
                f"redirected (HTTP {status}); give the address the redirect ends at",
            )
        if status in (401, 403):
            raise self._fail(
                424,
                "server_refused_credentials",
                f"refused the request's credentials (HTTP {status}); check the tool's headers "
                "or authorization",
            )
        if status >= 400:
            raise self._fail(502, "upstream_error", f"answered HTTP {status}")
        if ident is None:
            # Read (bounded) so the connection can serve the next request.
            await self._body(response)
            return _Answer(None, session_id)
        encoding = (response.headers.get("content-encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise self._fail(
                502, "upstream_error", f"compressed its answer ({encoding}) though asked not to"
            )
        kind = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if kind == "text/event-stream":
            return _Answer(await self._from_events(response, ident), session_id)
        if kind == "application/json":
            body = await self._body(response)
            try:
                parsed = json.loads(body)
            except ValueError:
                raise self._fail(
                    502, "upstream_error", "answered JSON that does not parse"
                ) from None
            return _Answer(_matching(parsed, ident), session_id)
        raise self._fail(
            502,
            "upstream_error",
            f"answered {kind or 'no content type'}, not JSON or an event stream; is it a "
            "Streamable HTTP MCP endpoint?",
        )

    async def _chunks(self, response: httpx.Response) -> AsyncIterator[bytes]:
        read = 0
        async for chunk in response.aiter_raw():
            read += len(chunk)
            if read > LIMIT_BYTES:
                raise self._fail(
                    502,
                    "too_large",
                    f"sent an answer over the {LIMIT_BYTES // (1024 * 1024)} MiB limit",
                )
            yield chunk

    async def _body(self, response: httpx.Response) -> bytes:
        return b"".join([chunk async for chunk in self._chunks(response)])

    async def _from_events(self, response: httpx.Response, ident: int) -> dict[str, Any]:
        """The JSON-RPC answer to `ident`, from a server-sent event stream.

        Notifications and requests the server sends first are skipped: this
        client offers no capability a server could ask it to use.
        """
        pending = bytearray()
        data: list[str] = []
        async for chunk in self._chunks(response):
            # Only the new bytes are searched for a line's end, so a long line
            # costs its length once, not once per chunk.
            scan = len(pending)
            pending += chunk
            while True:
                cut = pending.find(b"\n", scan)
                if cut < 0:
                    break
                line = bytes(pending[:cut]).rstrip(b"\r").decode("utf-8", "replace")
                del pending[: cut + 1]
                scan = 0
                if line == "":
                    found = _event(data, ident)
                    data = []
                    if found is not None:
                        return found
                elif line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
        found = _event(data, ident)
        if found is not None:
            return found
        raise self._fail(502, "upstream_error", "ended its event stream without answering")


def _matching(parsed: Any, ident: int) -> dict[str, Any] | None:
    """The response to `ident` in a JSON-RPC message or batch."""
    messages = parsed if isinstance(parsed, list) else [parsed]
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("id") == ident
            and ("result" in message or "error" in message)
        ):
            return message
    return None


def _event(data: list[str], ident: int) -> dict[str, Any] | None:
    if not data:
        return None
    try:
        parsed = json.loads("\n".join(data))
    except ValueError:
        return None
    return _matching(parsed, ident)


# --------------------------------------------------------------------------- #
# What the gateway asks for
# --------------------------------------------------------------------------- #


@dataclass
class McpTool:
    name: str
    description: str | None
    input_schema: dict[str, Any]
    annotations: dict[str, Any] | None


@dataclass
class McpCalled:
    output: str | None
    error: str | None


def _output(result: dict[str, Any]) -> str:
    """A tool's content as text: text parts joined; others named, not carried."""
    lines: list[str] = []
    for part in result.get("content") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            lines.append(part["text"])
        elif kind == "resource" and isinstance(part.get("resource"), dict):
            resource = part["resource"]
            if isinstance(resource.get("text"), str):
                lines.append(resource["text"])
            else:
                lines.append(f"[the tool returned a resource, {resource.get('uri')}, not carried]")
        elif kind == "resource_link":
            lines.append(f"[the tool linked a resource: {part.get('uri')}]")
        elif isinstance(kind, str):
            lines.append(f"[the tool returned {kind} content, which is not carried]")
    if not lines and result.get("structuredContent") is not None:
        lines.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(lines)


class RemoteMcp:
    """Lists and calls tools on remote MCP servers, for the gateway.

    One client per (reach, scheme), each built at first use and kept for
    the life of the process (R1.1), each on its own guarded backend, so a
    pooled connection checked under one rule never serves another.
    `resolver` and `connector` are the test seams for the guard;
    `transports` replaces a client's transport outright (protocol tests).
    """

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        connector: Any = None,
        transports: Mapping[tuple[str, str], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        self._resolver = resolver
        self._connector = connector
        self._transports = dict(transports or {})
        self._clients: dict[tuple[str, str], httpx.AsyncClient] = {}

    def _client(self, reach: str, scheme: str) -> httpx.AsyncClient:
        key = (reach, scheme)
        client = self._clients.get(key)
        if client is None:
            transport = self._transports.get(key)
            if transport is None:
                # httpx's own transport, its pool rebuilt on the guarded
                # backend (`_pool` is the one private attribute touched, as
                # `_http.MachineTransport` does; a test fails if httpx moves it).
                guarded = httpx.AsyncHTTPTransport(verify=ssl_context(), trust_env=False)
                guarded._pool = httpcore.AsyncConnectionPool(
                    ssl_context=ssl_context(),
                    max_connections=20,
                    max_keepalive_connections=5,
                    keepalive_expiry=5.0,
                    network_backend=GuardedBackend(
                        reach, scheme, resolver=self._resolver, inner=self._connector
                    ),
                )
                transport = guarded
            # `internal_client` for what it guarantees here: no proxy from the
            # environment, and the transport given kept as it is. Where this
            # client may connect is the guard's rule, not "internal".
            client = internal_client(
                transport=transport,
                follow_redirects=False,
                timeout=httpx.Timeout(LIMIT_SECONDS, connect=10.0),
            )
            self._clients[key] = client
        return client

    async def _run(
        self, server_url: str, reach: str, headers: Mapping[str, str] | None, work: Any
    ) -> Any:
        where = target(server_url, reach)
        sent = caller_headers(headers)
        scrub = Scrubber.of(sent)
        session = _Session(self._client(reach, where.scheme), where, sent, scrub)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(LIMIT_SECONDS):
                await session.open()
                return await work(session, scrub)
        except TimeoutError:
            raise McpFailure(
                504,
                "timeout",
                f"The MCP server at {where.origin} did not finish within {LIMIT_SECONDS:g}s",
            ) from None
        except McpFailure as failure:
            # The origin and the cause only: never a header, a path or a query.
            log.info("remote MCP at %s failed: %s", where.origin, failure.code)
            raise
        finally:
            await session.end()
            log.debug(
                "remote MCP at %s took %.0f ms",
                where.origin,
                (time.perf_counter() - started) * 1000,
            )

    async def list_tools(
        self,
        server_url: str,
        *,
        reach: str,
        headers: Mapping[str, str] | None,
        allowed: list[str] | None,
    ) -> list[McpTool]:
        wanted = set(allowed) if allowed is not None else None

        async def work(session: _Session, scrub: Scrubber) -> list[McpTool]:
            tools: list[McpTool] = []
            names: set[str] = set()
            cursor: str | None = None
            for _ in range(LIMIT_PAGES):
                result = await session.call("tools/list", {"cursor": cursor} if cursor else {})
                for entry in result.get("tools") or []:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name")
                    if not isinstance(name, str) or not _TOOL_NAME.match(name) or name in names:
                        continue
                    if wanted is not None and name not in wanted:
                        continue
                    names.add(name)
                    if len(names) > LIMIT_TOOLS:
                        raise session._fail(
                            413,
                            "too_many_tools",
                            f"lists more than {LIMIT_TOOLS} tools; name the ones to use in "
                            "the tool's allowed_tools",
                        )
                    schema = entry.get("inputSchema")
                    described = entry.get("description")
                    annotations = entry.get("annotations")
                    tools.append(
                        McpTool(
                            name=name,
                            description=scrub.text(described)
                            if isinstance(described, str)
                            else None,
                            input_schema=scrub.deep(schema)
                            if isinstance(schema, dict)
                            else {"type": "object", "properties": {}},
                            annotations=scrub.deep(annotations)
                            if isinstance(annotations, dict)
                            else None,
                        )
                    )
                nxt = result.get("nextCursor")
                if not isinstance(nxt, str) or not nxt:
                    return tools
                cursor = nxt
            raise session._fail(
                502, "too_many_pages", f"did not end its tool list within {LIMIT_PAGES} pages"
            )

        result: list[McpTool] = await self._run(server_url, reach, headers, work)
        return result

    async def call(
        self,
        server_url: str,
        *,
        reach: str,
        headers: Mapping[str, str] | None,
        name: str,
        arguments: dict[str, Any],
    ) -> McpCalled:
        async def work(session: _Session, scrub: Scrubber) -> McpCalled:
            result = await session.call("tools/call", {"name": name, "arguments": arguments})
            text = scrub.text(_output(result))
            if result.get("isError") is True:
                return McpCalled(output=None, error=text or "the tool reported a failure")
            return McpCalled(output=text, error=None)

        called: McpCalled = await self._run(server_url, reach, headers, work)
        return called

    async def aclose(self) -> None:
        for client in self._clients.values():
            with contextlib.suppress(Exception):
                await client.aclose()
