"""Settings never lie (Troy, 2026-09-29: fundamental).

For a search account's config trio: a null in the file is the default
(`probeMinutes: null` turned the probe off while the schema said every ten
minutes); an unset address says whose it is, or that there is none; an empty
key is no key; and a restart is pending only while the saved value differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eugene_plexus_tool_driver._generated.models import ConfigUpdateRequest
from eugene_plexus_tool_driver.config import ConfigStore, as_schema


def test_a_null_in_the_file_is_the_default(tmp_path: Path) -> None:
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump({"probeMinutes": None, "maxResults": None}), encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    assert store.get("probeMinutes") == 10 and store.get("maxResults") == 5


def test_an_unset_address_says_whose_it_is(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "search.yaml")
    store.load()
    fields = {f.key: f for f in as_schema(values=store.snapshot()).fields}
    assert "searches nothing" in fields["baseUrl"].unsetMeans
    store.apply_patch(ConfigUpdateRequest.model_validate({"provider": "brave"}))
    brave = {f.key: f for f in as_schema(values=store.snapshot()).fields}
    assert brave["baseUrl"].unsetResolvesTo == "https://api.search.brave.com"


def test_an_unset_search_interval_says_the_providers_own_pace(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "search.yaml")
    store.load()

    def interval() -> Any:
        return next(
            f for f in as_schema(values=store.snapshot()).fields if f.key == "searchIntervalSeconds"
        )

    searxng = interval()
    assert searxng.default is None and searxng.unsetResolvesTo == 0
    assert "as they come" in searxng.unsetMeans
    store.apply_patch(ConfigUpdateRequest.model_validate({"provider": "brave"}))
    brave = interval()
    assert brave.unsetResolvesTo == 1 and "1s between searches" in brave.unsetMeans
    # Whose pace it is, in words that are true for a key someone can still
    # get: Brave stopped issuing free keys on 2026-02-12, and a prepaid key
    # allows more than one search a second. "The pace Brave documents for
    # its free plan" described a plan a new user cannot sign up for.
    assert "free plan" not in brave.unsetMeans
    assert "free key" in brave.unsetMeans and "2026-02-12" in brave.unsetMeans
    assert "prepaid key allows more" in brave.unsetMeans
    # 0 is a choice, shown as one, not as unset.
    store.apply_patch(ConfigUpdateRequest.model_validate({"searchIntervalSeconds": 0}))
    assert store.get("searchIntervalSeconds") == 0 and interval().unsetMeans is None


def test_an_empty_key_is_no_key_and_restarts_are_per_patch(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "search.yaml")
    store.load()
    store.apply_patch(ConfigUpdateRequest.model_validate({"apiKey": ""}))
    assert store.as_document().model_dump().get("apiKey") is None
    first = store.apply_patch(ConfigUpdateRequest.model_validate({"logLevel": "DEBUG"}))
    assert first.pendingRestart == ["logLevel"]
    live = store.apply_patch(ConfigUpdateRequest.model_validate({"maxResults": 7}))
    assert live.requiresRestart is False
    field = next(
        f for f in as_schema(pending=store.pending_restart()).fields if f.key == "logLevel"
    )
    assert field.pendingRestart is True and field.inEffect == "INFO"


def test_a_restart_key_set_back_to_the_running_value_needs_nothing(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "search.yaml")
    store.load()
    store.apply_patch(ConfigUpdateRequest.model_validate({"logLevel": "DEBUG"}))
    back = store.apply_patch(ConfigUpdateRequest.model_validate({"logLevel": "INFO"}))
    assert back.requiresRestart is False and back.pendingRestart == []
    assert store.pending_restart() == {}
