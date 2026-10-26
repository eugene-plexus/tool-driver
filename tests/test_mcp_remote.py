"""Remote MCP in the tool driver (row 59b): where it may connect, and what it keeps.

Two kinds of test. The address rules run through the real guarded backend:
an IP literal needs no lookup, and a name is answered by a test resolver,
so nothing here reaches a network. The protocol (sessions, pages, event
streams, limits, the caller's headers) runs against `mcp_fixture`'s server
in-process, through the client's transport seam, and once through the
guard to a real socket.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpcore
import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, Response

from eugene_plexus_tool_driver import mcp_remote
from eugene_plexus_tool_driver.app import create_app
from eugene_plexus_tool_driver.egress_guard import (
    GLOBAL,
    LOOPBACK,
    NETWORK,
    NEVER,
    AddressRefused,
    GuardedBackend,
    classify,
)
from eugene_plexus_tool_driver.settings import Settings

from .conftest import FakeInstall
from .mcp_fixture import FixtureServer, serving, tool

TOKEN = "mcp-caller-token-do-not-leak-0123456789"
SERVER = "https://mcp.example.com/mcp"


class NeverConnect:
    """A connector that fails the test if anything gets past the guard."""

    def __init__(self) -> None:
        self.hosts: list[str] = []

    async def connect_tcp(self, host: str, port: int, *args: Any, **kwargs: Any) -> Any:
        self.hosts.append(host)
        raise AssertionError(f"the guard let a connection through to {host}")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ToFixture:
    """A connector that records the literal it was given and dials the fixture,
    as if that literal were the fixture's address."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.hosts: list[str] = []
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        *args: Any,
        **kwargs: Any,
    ) -> httpcore.AsyncNetworkStream:
        self.hosts.append(host)
        return await self._inner.connect_tcp("127.0.0.1", self.port, timeout)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class Resolver:
    """Answers names from a table, and counts how often it was asked."""

    def __init__(self, answers: dict[str, list[list[str]]]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def __call__(self, host: str, port: int) -> list[Any]:
        self.asked.append(host)
        queue = self.answers[host]
        found = queue.pop(0) if len(queue) > 1 else queue[0]
        return [ipaddress.ip_address(a) for a in found]


def make_client(
    tmp_path: Path,
    *,
    resolver: Any = None,
    connector: Any = None,
    fixture: FixtureServer | None = None,
) -> TestClient:
    app = create_app(Settings(config_file=tmp_path / "td.yaml"))
    app.state.mcp_network = {"resolver": resolver, "connector": connector or NeverConnect()}
    if fixture is not None:
        transport = httpx.ASGITransport(app=fixture.app)
        app.state.mcp_transports = {
            ("internet", "https"): transport,
            ("network", "https"): transport,
            ("network", "http"): transport,
        }
    return TestClient(app)


def list_tools(client: TestClient, url: str = SERVER, **body: Any) -> httpx.Response:
    return client.post(
        "/v1/tools/mcp/list_tools", json={"serverUrl": url, "reach": "internet", **body}
    )


def call(client: TestClient, url: str = SERVER, **body: Any) -> httpx.Response:
    return client.post(
        "/v1/tools/mcp/call",
        json={"serverUrl": url, "reach": "internet", "name": "echo", **body},
    )


# --------------------------------------------------------------------------- #
# Where it may connect
# --------------------------------------------------------------------------- #

PRIVATE = [
    ("127.0.0.1", "loopback"),
    ("127.8.9.10", "loopback"),
    ("10.0.0.5", "private network"),
    ("172.16.0.1", "private network"),
    ("192.168.1.1", "private network"),
    ("100.64.0.1", "private network"),
    ("169.254.169.254", "link-local"),
    ("169.254.170.2", "link-local"),
    ("100.100.100.200", "metadata"),
    ("0.0.0.0", "unspecified"),
    ("224.0.0.1", "multicast"),
    ("240.0.0.1", "reserved"),
    ("255.255.255.255", "reserved"),
    ("::1", "loopback"),
    ("::", "unspecified"),
    ("fe80::1", "link-local"),
    ("fc00::1", "private network"),
    ("fd00:ec2::254", "metadata"),
    ("fec0::1", "private network"),
    ("ff02::1", "multicast"),
    ("::ffff:127.0.0.1", "loopback"),
    ("::ffff:169.254.169.254", "link-local"),
    ("::ffff:10.0.0.1", "private network"),
    ("::7f00:1", "IPv4-compatible"),
    ("64:ff9b::7f00:1", "loopback"),
    ("64:ff9b::a9fe:a9fe", "link-local"),
    ("2002:7f00:1::1", "loopback"),
    ("2001:0:4136:e378:8000:63bf:80ff:fffe", "loopback"),
]


@pytest.mark.parametrize(("address", "words"), PRIVATE)
def test_a_private_address_is_refused(tmp_path, address: str, words: str) -> None:
    """Each range, as an IP literal (no lookup), under the default reach."""
    never = NeverConnect()
    host = f"[{address}]" if ":" in address else address
    with make_client(tmp_path, connector=never) as client:
        response = list_tools(client, f"https://{host}/mcp", headers={"Authorization": TOKEN})
    assert response.status_code == 403, response.text
    problem = response.json()
    assert problem["code"] == "address_refused"
    assert words in problem["detail"], problem["detail"]
    assert never.hosts == []
    assert TOKEN not in response.text


@pytest.mark.parametrize(("address", "words"), PRIVATE)
def test_the_same_address_behind_a_name_is_refused(tmp_path, address: str, words: str) -> None:
    resolver = Resolver({"mcp.example.com": [[address]]})
    never = NeverConnect()
    with make_client(tmp_path, resolver=resolver, connector=never) as client:
        response = list_tools(client)
    assert response.status_code == 403, response.text
    assert (
        f"mcp.example.com resolves to {ipaddress.ip_address(address)}"
        in (response.json()["detail"])
    )
    assert never.hosts == []


def test_every_address_a_name_answers_is_checked(tmp_path) -> None:
    """One public answer does not carry a private one through with it."""
    for answer in (["93.184.216.34", "10.0.0.5"], ["10.0.0.5", "93.184.216.34"]):
        resolver = Resolver({"mcp.example.com": [answer]})
        never = NeverConnect()
        with make_client(tmp_path, resolver=resolver, connector=never) as client:
            response = list_tools(client)
        assert response.status_code == 403, response.text
        assert "10.0.0.5" in response.json()["detail"]
        assert never.hosts == []


async def test_rebinding_cannot_switch_the_address() -> None:
    """The guard resolves once and hands the connector the address it checked.

    The resolver answers a public address first and loopback after: the
    first connection goes to the literal that was checked, and the next
    lookup's answer is checked again, and refused, before any connection.
    """
    resolver = Resolver({"rebind.example": [["93.184.216.34"], ["127.0.0.1"]]})
    seen: list[str] = []

    class Recorder:
        async def connect_tcp(self, host: str, port: int, *args: Any, **kwargs: Any) -> Any:
            seen.append(host)
            return object()

        async def sleep(self, seconds: float) -> None:
            return None

    backend = GuardedBackend("internet", "https", resolver=resolver, inner=Recorder())  # type: ignore[arg-type]
    await backend.connect_tcp("rebind.example", 443, timeout=5)
    assert seen == ["93.184.216.34"] and resolver.asked == ["rebind.example"]
    with pytest.raises(AddressRefused) as refused:
        await backend.connect_tcp("rebind.example", 443, timeout=5)
    assert "127.0.0.1" in refused.value.reason
    assert seen == ["93.184.216.34"]


def test_rebinding_through_a_real_session(tmp_path) -> None:
    """End to end, through httpx's pool and a real socket: every connection
    goes to the checked literal, and a later lookup that answers loopback is
    refused before anything is dialled."""
    fixture = FixtureServer()
    resolver = Resolver({"mcp.lan": [["10.20.30.40"]]})
    with serving(fixture) as port:
        dial = ToFixture(port)
        with make_client(tmp_path, resolver=resolver, connector=dial) as client:
            first = list_tools(client, "http://mcp.lan:9000/mcp", reach="network")
            assert first.status_code == 200, first.text
            assert [t["name"] for t in first.json()["tools"]] == ["echo", "shout"]
            assert dial.hosts and set(dial.hosts) == {"10.20.30.40"}
            # Kept alive: one lookup and one connection served the whole session.
            assert resolver.asked == ["mcp.lan"] and len(dial.hosts) == 1
            # The name now answers loopback. The pooled connection may serve
            # the next call; drop it, so the next one must resolve again, as
            # after a keep-alive expiry.
            resolver.answers["mcp.lan"] = [["127.0.0.1"]]
            client.app.state.mcp._clients.clear()  # type: ignore[attr-defined]
            second = list_tools(client, "http://mcp.lan:9000/mcp", reach="network")
    assert second.status_code == 403, second.text
    assert "127.0.0.1" in second.json()["detail"]
    assert "127.0.0.1" not in dial.hosts


def test_http_is_refused(tmp_path) -> None:
    resolver = Resolver({"mcp.example.com": [["93.184.216.34"]]})
    with make_client(tmp_path, resolver=resolver) as client:
        response = list_tools(client, "http://mcp.example.com/mcp")
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "scheme_refused"
    assert "https" in response.json()["detail"]
    assert resolver.asked == []


def test_the_expert_reach_allows_this_network_only(tmp_path) -> None:
    """`network`: a LAN address over http or https; never loopback, link-local or
    metadata; and plain http never to the internet."""
    fixture = FixtureServer()
    resolver = Resolver(
        {
            "nas.lan": [["192.168.1.20"]],
            "public.example": [["93.184.216.34"]],
            "metadata.lan": [["169.254.169.254"]],
            "self.lan": [["127.0.0.1"]],
        }
    )
    with serving(fixture) as port:
        dial = ToFixture(port)
        with make_client(tmp_path, resolver=resolver, connector=dial) as client:
            ok = list_tools(client, "http://nas.lan:9000/mcp", reach="network")
            assert ok.status_code == 200, ok.text
            assert dial.hosts[0] == "192.168.1.20"
            for url, words in (
                ("http://public.example/mcp", "use https"),
                ("http://metadata.lan/mcp", "link-local"),
                ("https://metadata.lan/mcp", "link-local"),
                ("http://self.lan/mcp", "loopback"),
            ):
                refused = list_tools(client, url, reach="network")
                assert refused.status_code == 403, (url, refused.text)
                assert words in refused.json()["detail"], refused.json()["detail"]
    assert set(dial.hosts) == {"192.168.1.20"}


def test_the_classes_of_the_internet() -> None:
    assert classify(ipaddress.ip_address("93.184.216.34"))[0] == GLOBAL
    assert classify(ipaddress.ip_address("2606:4700::1111"))[0] == GLOBAL
    assert classify(ipaddress.ip_address("::ffff:93.184.216.34"))[0] == GLOBAL
    # NAT64 of a public address is the internet; of loopback it is not.
    assert classify(ipaddress.ip_address("64:ff9b::5db8:d822"))[0] == GLOBAL
    assert classify(ipaddress.ip_address("64:ff9b::7f00:1"))[0] == LOOPBACK
    assert classify(ipaddress.ip_address("100.64.0.1"))[0] == NETWORK
    assert classify(ipaddress.ip_address("169.254.169.254"))[0] == NEVER


@pytest.mark.skipif(sys.platform == "win32", reason="Windows' resolver does not read these forms")
@pytest.mark.parametrize("host", ["2130706433", "0x7f000001", "127.1", "017700000001"])
def test_numeric_spellings_of_loopback_are_refused(tmp_path, host: str) -> None:
    """The real resolver reads these as 127.0.0.1, and its answer is what is checked."""
    never = NeverConnect()
    with make_client(tmp_path, connector=never) as client:
        response = list_tools(client, f"https://{host}/mcp")
    assert response.status_code in (403, 502), response.text
    if response.status_code == 403:
        assert "127.0.0.1" in response.json()["detail"]
    assert never.hosts == []


def test_a_url_with_a_password_is_refused(tmp_path) -> None:
    with make_client(tmp_path) as client:
        response = list_tools(client, "https://user:secret@mcp.example.com/mcp")
    assert response.status_code == 400
    assert "secret" not in response.text


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #


@pytest.fixture
def fixture() -> FixtureServer:
    return FixtureServer()


@pytest.fixture
def mcp(tmp_path, fixture) -> Iterator[TestClient]:
    with make_client(tmp_path, fixture=fixture) as client:
        yield client


def test_list_tools_and_a_call_round_trip(mcp, fixture) -> None:
    listed = list_tools(mcp, headers={"Authorization": f"Bearer {TOKEN}", "X-Team": "blue"})
    assert listed.status_code == 200, listed.text
    tools = listed.json()["tools"]
    assert [t["name"] for t in tools] == ["echo", "shout"]
    assert tools[0]["description"] == "Says the text back."
    assert tools[0]["inputSchema"]["required"] == ["text"]
    assert tools[0]["annotations"] == {"readOnlyHint": True}
    # One session: initialize, initialized, a page at a time, then ended.
    assert fixture.methods() == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/list",
    ]
    assert fixture.ended == ["session-1"]
    for _, _, headers, _ in fixture.seen:
        assert headers["authorization"] == f"Bearer {TOKEN}"
        assert headers["x-team"] == "blue"
        assert headers["accept"] == "application/json, text/event-stream"
    assert fixture.seen[-1][2]["mcp-protocol-version"] == "2025-06-18"
    called = call(mcp, arguments={"text": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert called.status_code == 200, called.text
    body = called.json()
    assert body["error"] is None
    assert body["output"] == "hi\n[the tool returned image content, which is not carried]"
    sent = fixture.seen[-1][3]
    assert sent["method"] == "tools/call"
    assert sent["params"] == {"name": "echo", "arguments": {"text": "hi"}}


def test_allowed_tools_filter_the_list(mcp) -> None:
    listed = list_tools(mcp, allowedTools=["shout"])
    assert [t["name"] for t in listed.json()["tools"]] == ["shout"]


def test_an_event_stream_answer_is_read(mcp, fixture) -> None:
    fixture.answer_as = "sse"
    listed = list_tools(mcp)
    assert listed.status_code == 200, listed.text
    assert [t["name"] for t in listed.json()["tools"]] == ["echo", "shout"]
    assert call(mcp, arguments={"text": "x"}).json()["output"].startswith("x")


def test_a_tool_that_fails_says_so(mcp) -> None:
    body = call(mcp, name="fail", arguments={}).json()
    assert body["output"] is None and body["error"] == "it broke"


def test_a_redirect_to_the_same_server_is_followed(mcp, fixture) -> None:
    def moved(request: Any, message: dict[str, Any]) -> Response | None:
        if request.url.path == "/mcp":
            return Response(status_code=307, headers={"location": "/moved"})
        return None

    fixture.hook = moved
    listed = list_tools(mcp)
    assert listed.status_code == 200, listed.text
    assert {path for path, *_ in fixture.seen} == {"/mcp", "/moved"}


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_to_another_server_is_not_followed(mcp, fixture, status: int) -> None:
    fixture.hook = lambda request, message: Response(
        status_code=status, headers={"location": "https://elsewhere.example/mcp"}
    )
    response = list_tools(mcp, headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 502, response.text
    assert response.json()["code"] == "redirect_refused"
    assert len(fixture.seen) == 1


def test_a_refused_credential_is_named_without_it(mcp, fixture, caplog) -> None:
    fixture.hook = lambda request, message: JSONResponse(
        {"error": f"bad token {TOKEN}"}, status_code=401
    )
    caplog.set_level(logging.DEBUG)
    response = list_tools(mcp, headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 424, response.text
    assert response.json()["code"] == "server_refused_credentials"
    assert TOKEN not in response.text
    assert TOKEN not in caplog.text


def test_headers_never_reach_errors_logs_or_output(mcp, fixture, caplog) -> None:
    """A server that echoes the caller's token gets it back as [redacted],
    in a JSON-RPC error, in a tool's description and in a tool's output."""
    caplog.set_level(logging.DEBUG)
    fixture.tools = [tool("echo", f"Uses {TOKEN} to work.")]
    listed = list_tools(mcp, headers={"Authorization": f"Bearer {TOKEN}"})
    assert listed.status_code == 200
    assert listed.json()["tools"][0]["description"] == "Uses [redacted] to work."
    echoed = call(mcp, arguments={"text": f"got {TOKEN}"}, headers={"X-Key": TOKEN})
    assert echoed.json()["output"].startswith("got [redacted]")

    def error(request: Any, message: dict[str, Any]) -> Response | None:
        if message.get("method") == "tools/call":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32000, "message": f"denied for {TOKEN}"},
                }
            )
        return None

    fixture.hook = error
    failed = call(mcp, arguments={"text": "x"}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert failed.status_code == 502
    assert "denied for [redacted]" in failed.json()["detail"]
    for response in (listed, echoed, failed):
        assert TOKEN not in response.text
    assert TOKEN not in caplog.text
    # The server's path is never in an error's words, only its origin.
    assert "mcp.example.com/mcp" not in failed.json()["detail"]


@pytest.mark.parametrize(
    "name",
    ["Host", "Content-Length", "content-type", "Accept", "Mcp-Session-Id", "Transfer-Encoding"],
)
def test_framing_headers_are_refused_by_name(mcp, fixture, name: str) -> None:
    response = list_tools(mcp, headers={name: TOKEN})
    assert response.status_code == 400
    assert name in response.json()["detail"]
    assert TOKEN not in response.text
    assert fixture.seen == []


def test_a_header_with_a_line_break_is_refused(mcp, fixture) -> None:
    response = list_tools(mcp, headers={"X-Key": "a\r\nHost: evil"})
    assert response.status_code == 400 and fixture.seen == []


# --------------------------------------------------------------------------- #
# Limits, each named
# --------------------------------------------------------------------------- #


def test_more_than_64_tools_is_named(mcp, fixture) -> None:
    fixture.tools = [tool(f"t{i}") for i in range(65)]
    fixture.page_size = 50
    response = list_tools(mcp)
    assert response.status_code == 413, response.text
    assert response.json()["code"] == "too_many_tools"
    assert "64 tools" in response.json()["detail"]
    fixture.tools = [tool(f"t{i}") for i in range(64)]
    assert len(list_tools(mcp).json()["tools"]) == 64


def test_an_answer_over_the_size_limit_is_named(mcp, fixture, monkeypatch) -> None:
    monkeypatch.setattr(mcp_remote, "LIMIT_BYTES", 4096)
    fixture.tools = [tool("big", "x" * 8000)]
    response = list_tools(mcp)
    assert response.status_code == 502, response.text
    assert response.json()["code"] == "too_large"
    assert "MiB limit" in response.json()["detail"]


def test_a_server_slower_than_the_limit_is_named(mcp, fixture, monkeypatch) -> None:
    monkeypatch.setattr(mcp_remote, "LIMIT_SECONDS", 0.3)
    fixture.slow = 1.0
    response = list_tools(mcp)
    assert response.status_code == 504, response.text
    assert response.json()["code"] == "timeout"


def test_a_tool_list_that_never_ends_is_named(mcp, fixture, monkeypatch) -> None:
    monkeypatch.setattr(mcp_remote, "LIMIT_PAGES", 3)
    fixture.tools = [tool(f"t{i}") for i in range(10)]
    fixture.page_size = 1
    response = list_tools(mcp)
    assert response.status_code == 502
    assert response.json()["code"] == "too_many_pages"


# --------------------------------------------------------------------------- #
# Who may ask, and what the driver says it runs
# --------------------------------------------------------------------------- #


def test_only_the_gateway_may_call(tmp_path, install: FakeInstall) -> None:
    app = create_app(Settings(config_file=tmp_path / "td.yaml"))
    app.state.auth_state = install.auth_state()
    app.state.mcp_network = {"connector": NeverConnect()}
    with TestClient(app) as client:

        def ask(token: str) -> httpx.Response:
            return client.post(
                "/v1/tools/mcp/list_tools",
                json={"serverUrl": "https://127.0.0.1/mcp", "reach": "network"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert ask(install.session()).status_code == 401
        assert ask(install.service("agent")).status_code == 403
        # The gateway's token passes auth and meets the address rule.
        gateway = ask(install.service("gateway"))
        assert gateway.status_code == 403 and gateway.json()["code"] == "address_refused"
        info = client.get("/v1/info", headers={"Authorization": f"Bearer {install.session()}"})
        assert "mcp" in info.json()["tools"]


def test_httpx_still_keeps_its_pool_where_the_guard_replaces_it() -> None:
    assert isinstance(httpx.AsyncHTTPTransport()._pool, httpcore.AsyncConnectionPool)
