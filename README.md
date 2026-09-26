# ai-cost-guardrail

A backend/system-design learning project for building an LLM Cost & Guardrail Gateway — a service that sits between applications and LLM providers and handles the infrastructure around LLM calls: authentication, tenant isolation, rate limiting, budget enforcement, cost accounting, usage auditing, and failure handling.

The goal is not to pretend this is a complete enterprise AI platform. The goal is to build and understand the engineering problems that appear around real LLM workloads.

## How the request flows

```
Client
  │
  ▼
JWT Authentication
  │
  ▼
Tenant / User Identification
  │
  ▼
Guardrails
 ├── Model Policy
 ├── Output Token Limit
 └── Redis Rate Limit
  │
  ▼
Provider Token Counting
  │
  ▼
Cost Reservation
  │
  ▼
PostgreSQL Budget Check + Row Lock
  │
  ▼
Circuit Breaker
  │
  ▼
LLM Provider
  │
  ├───────────────┐
  ▼               ▼
Success         Failure
  │               │
  ▼               ▼
Actual Usage    Release Reservation
  │
  ▼
Cost Settlement
  │
  ▼
Usage Record
  │
  ▼
Response
```

## How the money works

Budget reservation is deliberately separated from actual spending.

For every request:

1. Count the estimated input tokens using the provider's token-counting API.
2. Combine estimated input tokens with the client's maximum requested output tokens.
3. Calculate the maximum expected cost using the model's pricing configuration.
4. Lock the tenant row with `SELECT ... FOR UPDATE`.
5. Verify that the tenant has enough remaining budget.
6. Reserve the estimated amount and create a `usage_records` row with `status="reserved"`.
7. Call the LLM.
8. Read the provider's actual usage metadata.
9. Calculate the actual cost.
10. Settle the reservation:
    - actual cost lower than reservation → refund the difference
    - actual cost higher than reservation → increase spend to the actual cost
    - request failure → release the reservation

Money is stored using PostgreSQL `NUMERIC(12,6)` and handled with Python `Decimal` rather than floating-point arithmetic.

## Concurrency

Budget enforcement is backed by PostgreSQL rather than an in-memory counter.

The tenant row is locked during reservation:

```
Request A ─────┐
               │
               ▼
          Lock tenant
               │
          Check budget
               │
           Reserve
               │
            Commit
               │
          Release lock
               │
Request B ──────────────────► Lock tenant
                              │
                         Check updated budget
                              │
                            ...
```

This prevents concurrent requests from independently seeing the same remaining budget and both spending it.

## Circuit breaker

The gateway uses a circuit breaker around the upstream LLM call.

```
CLOSED
  │
  │ repeated failures
  ▼
OPEN
  │
  │ recovery timeout
  ▼
HALF_OPEN
  │
  ├── success ──► CLOSED
  │
  └── failure ─► OPEN
```

The current circuit breaker is per-process in-memory state. A distributed circuit breaker shared across multiple API instances is future work.

## API failure mapping

The gateway translates known failure conditions into controlled HTTP responses:

| Condition | Response |
|---|---|
| Missing / invalid authentication, invalid token claims | 401 |
| Invalid guardrail request, unsupported model | 400 |
| Rate limit exceeded | 429 |
| Redis/rate-limiter unavailable, circuit breaker open | 503 |
| Upstream LLM failure | 502 |

## Current status

Core gateway functionality is implemented and tested.

Current test suite: 35 tests passing.

### Implemented

- [x] Project setup and dependency management
- [x] FastAPI application structure
- [x] `/api/v1/chat` endpoint
- [x] JWT authentication with tenant/user identification
- [x] Authentication failure handling
- [x] Model policy and output-token limits
- [x] Redis-backed per-tenant rate limiting (incl. Redis-unavailable handling)
- [x] PostgreSQL tenant/user models + Alembic migrations
- [x] PostgreSQL-backed budget tracking with row-locked reservation
- [x] Provider-native input token counting
- [x] Model-aware CostEngine
- [x] Reservation/settlement lifecycle with actual usage-based costs
- [x] Failed-request reservation release
- [x] NUMERIC(12,6) money storage with Python Decimal calculations
- [x] Circuit breaker: closed/open/half-open, unit + integration tested
- [x] HTTP authentication and guardrail tests
- [x] Docker Compose development stack

### Next

- [ ] Reservation edge-case tests (actual usage above reservation, breaker failures restoring spend)
- [ ] Explicit failure handling for provider `count_tokens()` calls
- [ ] Upstream LLM timeout handling
- [ ] Atomic Redis rate limiting using Lua
- [ ] Stale reservation cleanup after process failure
- [ ] Security checks for PII, prompt injection, and leaked secrets
- [ ] Observability: structured logs, metrics, and request tracing
- [ ] Harden `/token` and `/tenants` development endpoints
- [ ] RAG/document ingestion workload
- [ ] Caching layer
- [ ] Distributed circuit breaker

## RAG workload

RAG is part of the broader planned scope of the gateway, but it is not yet implemented.

The planned flow:

```
Documents → Text Extraction → Chunking → Embeddings → Vector Storage
        → Retrieval → Relevant Context → LLM Gateway → Response
```

The gateway is intended to provide the infrastructure around the RAG workload — authentication, tenant isolation, rate limiting, token/cost accounting, budget enforcement, and provider reliability.

RAG implementation is intentionally scheduled after the core gateway is stable.

## Scope

This is a backend and system-design learning project, not an attempt to build a complete enterprise AI platform. The primary focus is everything around the LLM call — authentication, tenant isolation, rate limiting, guardrails, cost reservation, reliability, usage accounting, and settlement.

PostgreSQL provides durable state and accounting. Redis provides fast, ephemeral state.

RAG, caching, additional providers behind a common interface, distributed rate limiting and breaker state, stale reservation recovery, PII/secret handling, prompt-injection detection, background processing, and advanced observability are extensions built only when they support the project's learning objectives. Features that are not necessary for demonstrating the gateway's core engineering concepts will remain future work rather than being added simply to make the project appear larger.

## Stack

**In use:** Python, FastAPI, PostgreSQL, SQLAlchemy, Alembic, Redis, JWT, Google Gemini, uv, pytest, Docker / Docker Compose

**Planned:** PGVector, additional LLM provider, RAG retrieval pipeline, semantic caching, background processing

## Running locally

### Docker Compose

```bash
docker compose up --build
```

This starts the API, PostgreSQL, and Redis, and runs database migrations automatically.

API documentation: `http://127.0.0.1:8000/docs`

The development `/token` endpoint can be used to obtain a JWT for local testing. It is development tooling and is not intended to represent a production authentication system.

### Local development

Start the dependencies:

```bash
docker compose up -d postgres redis
```

Install/sync Python dependencies:

```bash
uv sync
```

Apply migrations:

```bash
uv run alembic upgrade head
```

Start the API:

```bash
uv run uvicorn app.main:app --reload
```

### Run tests

```bash
uv run pytest tests -v
```

## Why this project?

Calling an LLM API is the easy part.

The interesting engineering problems appear around it:

- What happens when many users share one budget?
- How do you prevent concurrent requests from overspending?
- How do you account for actual token usage?
- What happens when the provider fails?
- How do you stop repeatedly sending traffic to a failing provider?
- How do you isolate tenants?
- How do you rate-limit requests?
- How do you recover a reservation if a process crashes?
- How do you eventually support RAG without losing control of cost and reliability?

This project is an attempt to build those systems, test them, understand their trade-offs, and document what is actually implemented rather than pretending unfinished infrastructure is production-ready.