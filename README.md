# Pullsmith

**Turns a GitHub issue into a reviewed pull request.**

Point it at a repository and an issue. It reads the issue, indexes the codebase, retrieves the
relevant code with hybrid search, plans a fix, edits the files, runs the test suite, analyses
failures and retries, scores the risk, and opens a pull request **only after a human approves
the exact diff**.

Not a chatbot, and not a document-RAG demo. The agent loop, the tool contract, the state machine
and the approval gate are all written here rather than pulled from a framework.

> **Status: Phases 1–6A and CI built. 369 backend + 14 frontend tests passing, ruff and mypy
> clean.**
>
> Working and tested: the API, durable job queue, run state machine, event timeline, worker,
> live dashboard, repository ingestion, syntax-aware chunking, embeddings, pgvector storage with
> an HNSW index, hybrid retrieval fused by RRF, the tool layer with its safety boundaries, the
> LLM abstraction and agent loop, issue analysis and planning, write tools, the test runner, the
> test-driven repair loop, diff generation, risk scoring, the human approval gate, and — new in
> 6A — the orchestrator that drives all of it, including resuming a run after a crash.
>
> A full run goes end to end against a real git repository: clone, analyse, index, plan, edit,
> test, diff, score risk, park for a human. Nine integration tests drive it, with the model and
> sandbox scripted so the results are reproducible.
>
> **Against the real Gemini API**, three live runs verified everything up to and including
> retrieval and the tool loop, and found four real bugs that the test suite could not: a retired
> model name, a reasoning model spending its whole output budget on thinking, the model
> requesting another tool instead of returning the plan, and incremental indexing that counted
> unchanged files without carrying their chunks forward. All four are fixed and now covered by
> tests. Details in [BUILD_PLAN.md](./BUILD_PLAN.md) section 20.
>
> **What is honestly incomplete:**
>
> **It works, and there are two real pull requests to prove it:**
>
> - **[agent-sandbox#3](https://github.com/DIKSHAKUMA/agent-sandbox/pull/3) — fully autonomous.**
>   The real Gemini model read the issue, worked out the root cause, chose the file, wrote the
>   two-line fix, ran the tests, self-verified and scored risk. A human approved; the pull
>   request opened. Verified by independent clone: 4 of 4 tests pass on the branch, 2 of 4 fail
>   on `main`.
> - [agent-sandbox#2](https://github.com/DIKSHAKUMA/agent-sandbox/pull/2) — the same pipeline
>   with the model scripted, used to prove the publish path while the model quota was exhausted.
>
> **What is honestly incomplete:**
>
> 1. **One solved task is not a success rate.** That bug was small and well-specified, in a
>    five-file repository whose tests already named the expected behaviour. It proves the
>    pipeline works. It says nothing about performance on a real codebase, and no percentage will
>    be published until Phase 6B measures it with N stated.
> 3. **No remote sandbox.** Only the local subprocess runner exists, which is process
>    confinement and not isolation. Tests would run unisolated, and configuration validation
>    refuses this backend outside development.
> 4. **Retrieval quality is unmeasured** — the Gemini free-tier embedding quota was exhausted
>    during setup, so live runs used a deterministic fake embedding provider. Storage, indexing,
>    vector search, trigram search and fusion are proven; semantic ranking quality is not. No
>    Recall@K number is published until it is real.
> 5. **Retrieval is stubbed in the orchestrator tests**, because similarity search needs
>    Postgres and those tests run on SQLite. Retrieval has its own Postgres tests, and the live
>    runs exercised both together, but no automated test covers the pair.
> 6. **A run sent back for revision cannot later be approved**, because the schema allows one
>    decision per run. Fixing it means versioning approvals the way plans are versioned.
>
> Full plan and per-phase detail in [BUILD_PLAN.md](./BUILD_PLAN.md).

---

## Architecture

```
Vite React frontend
        │  REST + SSE
FastAPI API process
        │
   ┌────┴──────────────┬──────────────────┐
 Auth/session      Run manager        Read APIs
                       │ enqueue (Postgres job table)
        Neon Postgres + pgvector
                       ▲ claim (FOR UPDATE SKIP LOCKED)
              Worker process
                       │
            Agent orchestrator (state machine)
                       │
 Issue │ Repo map │ Code RAG │ Planner │ Tools │ Verifier │ Risk
                       │
        ┌──────────────┴──────────────┐
  SandboxRunner (remote)        GitHub client
```

Two OS processes. The API stays responsive; the worker does the long agent runs. Both
share Postgres, so a worker crash does not lose a run.

## Agent state machine

```
CREATED → CLONING_REPOSITORY → ANALYZING_ISSUE → EXPLORING_REPOSITORY
  → INDEXING_REPOSITORY → RETRIEVING_CONTEXT → PLANNING
  → [WAITING_FOR_PLAN_REVIEW] → IMPLEMENTING → TESTING
       ├─ pass → VERIFYING → WAITING_FOR_APPROVAL → PR_CREATED → COMPLETED
       └─ fail → ANALYZING_FAILURE → REVISING → TESTING   (≤ MAX_ITERATIONS)
  any → FAILED (categorised) | CANCELLED
```

Legal transitions are declared in one table (`app/agent/states.py`) and enforced on every
change. `PR_CREATED` is reachable from exactly one state, `WAITING_FOR_APPROVAL` — a test
asserts this, because it is the hinge of the safety model.

## Code RAG: why chunking is the hard part

The default RAG recipe splits text every N characters. For code that is actively harmful:
a fixed window cuts a function in half, so one chunk holds a signature with no body and
the next holds a body with no name. Neither answers "where is email validation handled?"

Instead, tree-sitter parses each file into a syntax tree and we cut at declaration
boundaries — functions, methods, classes, interfaces. Every chunk is a unit a developer
would recognise, and carries metadata:

```
symbol          update
symbol_kind     function
parent_symbol   ProfileService        → qualified name ProfileService.update
path            app/services/profile.py
lines           40-72
imports         [...]                 → what this code depends on
is_test         false
strategy        syntax | syntax-split | lines
```

That metadata is what makes retrieval precise rather than merely plausible: exact-symbol
lookup for stack traces, filtering tests in or out, and citing real line numbers.

Measured on this repository's own backend: **270 chunks from 55 files, 98% with a real
symbol name.** Inspect any repository yourself, with no embedding spend:

```powershell
.\venv\Scripts\python.exe -m scripts.index_preview <path>
```

Incremental indexing works by SHA-256 per file: re-indexing after a one-line change
re-embeds one file, not thousands. Everything is scoped to a commit SHA, so the agent can
never be handed a chunk of code that no longer exists in the version it is editing.

## Retrieval: why hybrid, not just vectors

Semantic search alone is not enough for code, and neither is keyword search.

| Strategy | Finds | Fails at |
|---|---|---|
| Semantic (pgvector, cosine) | code by meaning — "email validation" matches `check_address()` | exact identifiers: every `update` method looks alike |
| Lexical (pg_trgm) | exact symbols, error strings, paths — what a stack trace contains | anything phrased differently from the code |

Both run, then results are fused with **Reciprocal Rank Fusion**:

```
score = Σ  weight / (60 + rank_in_that_list)
```

Fusing by *rank* rather than score is deliberate: a cosine distance and a trigram
similarity are not on a comparable scale, so normalising one against the other would be an
invented conversion. RRF also rewards agreement — a chunk both strategies liked outranks
one that a single strategy loved.

Indexes that make it work, created in the Phase 2 migration:

```sql
CREATE INDEX ... USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ... USING gin (symbol  gin_trgm_ops);
CREATE INDEX ... USING gin (content gin_trgm_ops);
```

Try it against any indexed directory:

```powershell
.\venv\Scripts\python.exe -m scripts.rag_demo app --limit 5
```

## Sandbox honesty

The development machine cannot run Docker Desktop (2 cores, 5.9 GB RAM). `SandboxRunner`
is therefore an interface with three backends:

| Backend | Isolation | Use |
|---|---|---|
| `github_actions` | ephemeral GitHub-hosted VM | default |
| `e2b` | remote microVM | fast repair loop |
| `local_unsafe` | **process confinement only, not isolation** | development only |

`local_unsafe` is rejected by configuration validation outside development and is never
described as a sandbox. Backends land in Phase 3.

## Tech stack

Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2.0 async · Alembic ·
Neon Postgres + pgvector · Postgres-backed job queue · tree-sitter (Phase 2) ·
Vite + React + TypeScript + Tailwind + Zustand (Phase 1B) · SSE · GitHub Actions CI

Deliberately **not** used: Docker Desktop, Kubernetes, Redis, Celery, LangChain,
LangGraph, local embedding models. Reasons are recorded in `BUILD_PLAN.md`.

---

## Local setup

Requires Python 3.12 and Git.

```powershell
cd backend

# Dependencies. uv is used instead of pip: on a low-memory machine pip's downloader
# fails with an out-of-memory fault. See BUILD_PLAN.md section 9.
python -m venv venv
..\.tools\uv.exe pip install --python .\venv\Scripts\python.exe -r requirements-dev.txt

# Config
copy .env.example .env
# then set SESSION_SECRET and TOKEN_ENCRYPTION_KEY:
#   python -c "import secrets;print(secrets.token_urlsafe(48))"
#   python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"

# Schema (local development)
.\venv\Scripts\python.exe -m scripts.init_db

# Fixture user + repository + issue, prints a session cookie
.\venv\Scripts\python.exe -m scripts.dev_seed
```

Run the two processes in separate terminals:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
.\venv\Scripts\python.exe -m app.worker.main
```

API docs: `http://localhost:8000/docs`

### Real database (Neon Postgres)

```powershell
.\venv\Scripts\python.exe -m scripts.db_check      # connectivity + enable pgvector
.\venv\Scripts\python.exe -m alembic upgrade head  # apply migrations
.\venv\Scripts\python.exe -m scripts.db_inspect    # confirm tables and indexes
```

Verified against Neon: PostgreSQL 18.6, pgvector 0.8.6, 8 tables, 20 indexes, revision
`8c0540c85e61`.

Note on the connection string: Neon's console gives a libpq URL. It needs
`postgresql+asyncpg://`, `sslmode=require` changed to `ssl=require`, `channel_binding`
dropped, and `prepared_statement_cache_size=0` added because the pooled endpoint is
PgBouncer in transaction mode. Details in `BUILD_PLAN.md` section 11.

### Frontend

```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api/*` to the backend on port 8000. This is deliberate: the
session cookie is `SameSite=Lax`, so a cross-origin fetch would not send it. Proxying
keeps the browser on a single origin and removes CORS from the development path.

## Running a real end-to-end agent run

`scripts/e2e_setup.py` builds a throwaway git repository containing a genuine bug and a test
suite that fails because of it, seeds a repository and issue pointing at it, and queues a run.
Everything else is real: the clone, the indexing, the model, the tool layer, the sandbox runner,
the test runner, the diff, the risk score and the approval gate.

```powershell
cd backend
.\venv\Scripts\Activate.ps1      # required: the sandbox passes PATH to the child, and
                                  # the agent needs to find pytest to run the fixture's tests

python -m scripts.e2e_setup --reset     # prints RUN_ID and a session cookie
python -m app.worker.main                # in a second terminal, with the venv activated
python -m scripts.e2e_watch <RUN_ID>     # in a third; add --approve to approve the diff
```

The setup script refuses to start and tells you exactly what is wrong rather than failing
halfway. It checks for `SANDBOX_BACKEND=local_unsafe` (the remote backends raise rather than
silently downgrading), a Gemini API key, an embedding provider that will not hit an exhausted
quota, and `pytest` on PATH.

To watch it in the dashboard instead, run the API and frontend and open
`http://localhost:5173`:

```powershell
cd backend;  .\venv\Scripts\python.exe -m uvicorn app.main:app --reload
cd frontend; npm run dev
```

Sign in with a personal access token (the form appears when `APP_ENV` is not `production`), then
**Link from GitHub** to pick a repository, sync its issues, and start a run on one. Set
`GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` to get the "Continue with GitHub" button instead;
the sign-in screen only offers methods the server reports it can honour.

A run parks at `WAITING_FOR_APPROVAL` and goes no further without a human.

### Opening a real pull request

The local fixture never exercises the publish path. `scripts/e2e_github.py` uses a real
repository so approving actually opens a pull request. No OAuth app needed — a fine-grained
personal access token is enough, and it is stored exactly as an OAuth token would be:
Fernet-encrypted in `github_connection`, decrypted only inside the worker.

You need a **throwaway public** repository you own, and a token with Contents, Issues and Pull
requests set to read and write on it ([create one here](https://github.com/settings/personal-access-tokens/new)).
Public is required because the agent clones without credentials by design.

```powershell
$env:GITHUB_PAT = "github_pat_..."        # environment, not an argument, so it stays out of history
python -m scripts.e2e_github --repo yourname/agent-sandbox --seed --reset
python -m app.worker.main                  # second terminal
python -m scripts.e2e_watch <RUN_ID> --approve
```

`--seed` commits the buggy fixture files and opens the issue in your repository, which also
proves the token has the write permission the pull request will need before an agent run is spent
finding out. Drop the flag on later runs.

If the model quota is exhausted, `scripts/e2e_publish.py` runs the same thing with the model
scripted and everything else real, which is how
[pull request #2](https://github.com/DIKSHAKUMA/agent-sandbox/pull/2) was opened:

```powershell
python -m scripts.e2e_publish --repo yourname/agent-sandbox
```

It prints the diff, approves it, and opens the pull request. It proves the plumbing, not the
model's ability to solve a bug, and says so in its own docstring.

## Tests

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\ruff.exe check app tests scripts

cd ..\frontend
npm test
npm run typecheck
```

Backend: 339 tests covering the state machine, secret redaction, event ordering and resume,
queue claiming and crash recovery, config validation, token encryption, the API contract
including cross-user isolation, file exclusion rules, incremental change detection,
repository-map detection, code chunking across Python/TypeScript/TSX, the safety boundary, the
tool registry, the agent loop and structured-output repair, write tools, the test-output parser,
the repair loop's bounds, diff construction, risk scoring, the approval gate, and eight
end-to-end orchestrator runs against a real git repository on disk.

Frontend: 14 tests covering event merge ordering, duplicate suppression on SSE replay,
history/stream overlap, run-state presentation and the diff viewer.

Backend tests run against in-memory SQLite, which is what keeps the suite fast on a two-core
machine. The vector column is a `TypeDecorator` that emits `vector(n)` on Postgres and JSON
elsewhere, so storage is portable; similarity search is not, so retrieval tests require
Postgres and the orchestrator tests substitute a retriever.

The integration tests start real `git` subprocesses. On a memory-constrained machine a
subprocess can fail to start at all (Windows `0xC0000142`), and the fixture skips with that
exact reason rather than reporting a false failure. Re-run
`pytest tests/test_orchestrator_integration.py` on its own if that happens.

## Environment variables

| Name | Purpose |
|---|---|
| `APP_ENV` | `development` / `test` / `production`; gates unsafe defaults |
| `DATABASE_URL` | async SQLAlchemy URL; use an absolute path for SQLite |
| `SESSION_SECRET` | signs the session cookie |
| `TOKEN_ENCRYPTION_KEY` | Fernet key encrypting GitHub tokens at rest |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | OAuth app credentials |
| `FRONTEND_ORIGIN` | CORS allow-list entry |
| `MAX_ITERATIONS`, `MAX_TOOL_CALLS`, `MAX_RUN_SECONDS`, `MAX_RETRIEVED_CHUNKS` | agent cost caps |
| `SANDBOX_BACKEND` | `github_actions` / `e2b` / `local_unsafe` |
| `AGENT_WORKSPACE_ROOT` | where repositories are checked out, one directory per run |

Secrets are never logged: a redaction filter in the logging pipeline strips GitHub
tokens, bearer headers, provider keys and inline database credentials.

## The approval gate

The project's central safety claim:

> No push and no pull request without a stored approval row whose hash matches the diff being
> pushed.

```python
approval = await assert_push_allowed(session, run_id=run.id, diff_text=diff)
# raises ApprovalRequiredError unless:
#   an approval row exists, AND
#   its decision is "approved", AND
#   its diff_hash equals hash(diff_text)
```

Four properties worth knowing:

- **Enforced in the service layer, never the UI.** A UI-only guard is bypassed by one `curl`.
- **A persisted row, not a flag.** It survives restarts and records who decided what, when.
- **It raises rather than returning `False`,** so a caller cannot ignore it by forgetting to
  check a return value.
- **Bound to a content hash.** Approving a one-line fix cannot authorise pushing a different
  diff. If the workspace changed after review, the push is refused.

The database backs this up: `uq_approval_per_run` makes a second decision a conflict rather than
an overwrite, and `pull_request.approval_id` is `ON DELETE RESTRICT` so the authorising approval
cannot be deleted while the pull request exists.

A credential detected in the diff sets `risk_blocking`, and a blocked change cannot be approved
at all — the API refuses and the UI shows no approve button.

Three decisions exist, not two: **approve**, **reject** (terminal), and **request revision**
(returns to planning, keeps the work, requires a comment).

## Security posture (Phase 1A)

- Every endpoint except `/health` and the auth routes requires a session.
- Session is a signed, expiring **HttpOnly** cookie, not a token in `localStorage`.
- GitHub tokens are Fernet-encrypted at rest and never returned by any endpoint.
- Ownership is enforced **in the SQL query**, not by comparing in Python afterwards.
- Another user's repository returns `404`, not `403`, so the API does not confirm it exists.
- Unhandled exceptions return a generic message plus a request id; details stay in logs.
- Production config validation refuses placeholder secrets, SQLite and the unsafe sandbox.

## Limitations

- **No pull request has been created against a real repository.** The publish path is wired and
  tested against a faked GitHub Contents API, including the approval gate being re-checked at
  the push. One real run against a throwaway repository is the next step.
- **The local runner is process confinement, not isolation.** It is never called a sandbox, and
  production config validation refuses it. Remote backends (GitHub Actions, E2B) are designed
  and raise `NotImplementedError` rather than silently downgrading to the unisolated runner.
- **Retrieval quality is unmeasured.** The end-to-end run used the deterministic fake
  embedding provider because the Gemini free-tier quota was exhausted during setup. Storage,
  indexing, vector search, trigram search and fusion are all proven; semantic ranking
  quality is not. No Recall@K will be published until it is measured on real embeddings.
- **Retrieval and orchestration are not tested together.** Similarity search needs Postgres;
  the orchestrator tests run on SQLite and inject a retriever that returns nothing.
- **A run sent back for revision cannot later be approved.** `uq_approval_per_run` allows one
  decision per run. Fixing it means versioning approvals the way plans are versioned.
- **Private repositories cannot be cloned.** `workspace.clone` passes a bare clone URL with no
  credentials on purpose, so a token can never end up in `.git/config` or a process listing. The
  cost is that only public repositories work today. The fix is a short-lived credential supplied
  through `GIT_ASKPASS`, never interpolated into the URL.
- **A run awaiting review holds its checkout on disk**, because the pull request is built from
  those files. Fine at this scale, and the reason remote sandboxes matter beyond it.
- **No evaluation harness yet** (Phase 6B): no task success rate, no autonomous resolution
  rate, no cost per run. Those numbers will be published only once measured, with N stated.
- **No `ruff format` gate in CI.** The code was formatted by hand and the formatter would rewrite
  47 files, so the gate would be red on day one. CI runs `ruff check` and `mypy`, both clean.
- **Architecture diagrams are still text**, not drawn.
- Query latency measured at 1.0–2.9s, dominated by network round trips to `us-east-2` and
  Neon's cold start after scale-to-zero rather than by the index.
- OAuth CSRF state is held in process memory, which is fine for one instance and will
  move to the database if the API is ever scaled out.

## Roadmap

See [BUILD_PLAN.md](./BUILD_PLAN.md) for all phases, exit criteria and the interview
questions each phase must make answerable.
