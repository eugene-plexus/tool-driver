# Eugene Plexus — `tool-driver`

[![CI](https://github.com/eugene-plexus/tool-driver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/tool-driver/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

Runs the tools the [Eugene Plexus](https://github.com/eugene-plexus) hub runs
itself — **web search** today — for **one** tool provider account.

OpenAI's and Anthropic's APIs define `web_search` as a tool the *provider*
runs. Point Codex or Claude Code at a local model and nobody runs it: Codex
silently loses its search and Claude Code's WebSearch fails. With a
tool-driver in the install, the [`gateway`](https://github.com/eugene-plexus/gateway)
offers the model a search tool, sends each search the model makes here, and
hands the results back, so the model answers with sources.

An install runs **zero or more instances, one per search account**:

- **SearXNG** — free and self-hosted. Give it the instance's address. The
  instance must allow JSON output: add `json` to `search.formats` in its
  `settings.yml` (it answers `403` otherwise, and this component says so).
- **Brave Search** — a paid service with its own index. Give it your API key.

Either way the words searched for leave your machine: a SearXNG on your LAN
forwards them to public engines. That is why a client key marked local-only
never has a search run for it.

## What this service is — and what it isn't

It holds the account's secret (sealed at rest, like an inference-driver's API
key) and serves `POST /v1/tools/web_search`. It does **not** run the loop
that decides when to search, how many times, or what the model is told — that
is the gateway's, so every door (chat completions, Responses, Anthropic
messages) behaves the same. It never runs tools that act on your machine;
files, shells and your own MCP servers stay in apps and clients.

The contract is [`openapi/tool-driver.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/tool-driver.yaml)
in `specs`; the design is
[`docs/design/server-run-tools.md`](https://github.com/eugene-plexus/specs/blob/main/docs/design/server-run-tools.md).

## Running it

The agent supervises it: add a search account from the console
(**Backends → Add a search account**), or declare a component of kind
`tool-driver`. Standalone, for development:

```bash
python -m eugene_plexus_tool_driver   # binds 127.0.0.1:8190, no auth
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python scripts/codegen.py   # regenerate models at the pinned SPECS_REF
ruff check . && ruff format --check . && mypy src/ && pytest
```

Apache-2.0. Contributions are signed off (DCO), not assigned.
