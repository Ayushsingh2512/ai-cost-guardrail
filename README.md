# ai-cost-guardrail

Building a gateway that sits in front of LLM API calls and handles the stuff you actually need in production but never see in tutorials — rate limits, cost tracking, PII scrubbing, a circuit breaker for when the provider goes down.

Basically: I kept reading about companies burning through OpenAI credits with no per-user limits, or a single flaky API call taking down an entire feature, and wanted to actually build something that solves that instead of just reading about it.

## What it does (or will do)

- Auth + tenant/user identification via JWT
- Per-tenant rate limiting using Redis — distinct `429` response so clients know to back off, not that their request was invalid
- Checks incoming requests for PII, prompt injection attempts, and leaked secrets before they go anywhere
- Reserves budget for a request *before* calling the LLM, so two concurrent requests from the same tenant can't both slip through and blow the budget
- Circuit breaker around the LLM call — if the provider starts failing, stop hammering it and fail fast instead
- Caches repeated queries so identical requests don't hit the LLM twice
- A separate path for file uploads — extract text, chunk it, embed it, store it for retrieval later

## Architecture

![Architecture diagram](./docs/architecture_diagram.png)

This is the target design, not what's built yet — see the checklist below for actual progress.

## Where it's at right now

Core gateway is working and tested end to end. Still building out the production-hardening pieces.

- [x] Project setup, folder structure, dependencies
- [x] `/chat` endpoint with real JWT auth (tested: valid/invalid/tampered/expired tokens, missing-token rejection)
- [x] Guardrail engine — token limits, model policy, budget reservation — built as a standalone, unit-tested service
- [x] Redis integration (tested under connection failure and recovery)
- [x] PostgreSQL integration (tested under connection failure and recovery)
- [x] Tenant/User models + Alembic migrations, real tables verified
- [x] Budget tracking backed by real Postgres data — survives a server restart, no longer in-memory
- [x] pytest coverage for core guardrail logic
- [x] Docker compose so the whole app (not just Postgres/Redis) is a one-command run
- [x] Distributed rate limiter (Redis) — tested under real 429 conditions
- [ ] Security checks on incoming requests (PII, prompt injection)
- [ ] Circuit breaker
- [ ] Caching layer
- [ ] File ingestion pipeline

I'll update this as things get built instead of pretending it's all done.

## Stack

**In use:** FastAPI, PostgreSQL + SQLAlchemy + Alembic, Redis, JWT, Google Gemini, uv, pytest

**Planned:** PGVector (semantic caching), Celery (async file processing)

## Running it

```bash
docker-compose up --build     # starts everything — API + Postgres + Redis, migrations run automatically
```

Visit `http://127.0.0.1:8000/docs` for the interactive API once it's up. You'll need a JWT to hit `/chat` — generate one via `POST /token` (dev-only, not how real auth would work).

For local development without rebuilding the container on every code change:
```bash
docker-compose up -d postgres redis   # just the dependencies
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

## Why

Wanted a project that goes past "call an LLM API and return the response" and actually deals with the things that break in production — concurrency bugs in budget tracking, what happens when a provider times out, that kind of thing. Not trying to make this look finished before it is.
