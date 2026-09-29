"""The two cross-repo rules, checked here as every component checks them."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_tool_driver"


def test_durations_are_measured_with_perf_counter() -> None:
    """`time.monotonic()` is `GetTickCount64` on Windows/CPython 3.12 -- a
    15.6 ms grid -- and 3.12 is what both installers provision. Read as
    text, because the defect is which function the source names."""
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"


def test_no_http_client_is_built_outside_the_one_module_that_owns_them() -> None:
    """One client per instance, never per call (R1.1): constructing one
    parses certifi's bundle on the event loop, ~105 ms. Every client here
    is built through `_http.py`, once, by `SearchService`."""
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if "_generated" not in path.parts
        and path.name != "_http.py"
        and "httpx.AsyncClient(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these build their own client: {offenders}"
