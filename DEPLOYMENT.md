# Deployment

## Read this first: the worker is not deployed, on purpose

The worker runs the test suite of whatever repository it is working on. That is **arbitrary
third-party code**, and the only sandbox implementation in this project is a plain subprocess on
the host with a stripped environment and a command allowlist. That is process confinement, not
isolation, and it is never called a sandbox anywhere in the codebase.

Putting that on a public server would mean arbitrary code executing next to the database
credentials and other people's encrypted GitHub tokens. The configuration layer refuses to let
it happen:

```python
if self.sandbox_backend is SandboxBackend.local_unsafe:
    problems.append(
        "SANDBOX_BACKEND=local_unsafe provides no isolation and is "
        "forbidden outside development"
    )
```

So both routes are closed, which is the correct outcome:

| Attempt | Result |
|---|---|
| `APP_ENV=production`, `SANDBOX_BACKEND=local_unsafe` | Worker refuses to start |
| `APP_ENV=production`, any remote backend | `build_sandbox` raises `NotImplementedError`, because Phase 3D is not built |
| `APP_ENV=development` on a public host | Guard bypassed. Do not do this |

**What is deployed:** the API and the frontend. You can sign in, browse repositories and issues,
read every past run's timeline, plan, diff, risk score and pull request. Starting a run enqueues
a job, and the UI states plainly that nothing is consuming the queue.

**What executes runs:** the worker, on a machine you trust, pointed at the same database. That
is a legitimate architecture — the queue is durable and the worker is deliberately a separate
process — it just means the public URL is a read-only window unless your laptop is running.

Deploying the worker properly requires a remote sandbox (GitHub Actions runner or a microVM).
Everything above it is already written against the `SandboxRunner` protocol, so that is an
implementation, not a redesign.

---

## 1. Database

Neon Postgres with the pgvector extension. You already have one. Apply migrations:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
alembic upgrade head
```

## 2. API on Render

`render.yaml` is a blueprint. In the Render dashboard: **New → Blueprint**, point it at
`github.com/DIKSHAKUMA/pullsmith`, and it reads the file.

Set these in the dashboard when prompted (they are marked `sync: false` so they never live in
the repository):

| Variable | Value |
|---|---|
| `DATABASE_URL` | Neon pooled URL, `postgresql+asyncpg://`, `?ssl=require` |
| `SESSION_SECRET` | `python -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `TOKEN_ENCRYPTION_KEY` | `python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"` |
| `GEMINI_API_KEY` | your key |
| `FRONTEND_ORIGIN` | your Vercel URL, e.g. `https://pullsmith.vercel.app` |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | from a GitHub OAuth app |
| `GITHUB_OAUTH_REDIRECT_URI` | `https://<your-api>.onrender.com/auth/github/callback` |

`TOKEN_ENCRYPTION_KEY` must be **stable**. Change it and every stored GitHub token becomes
undecryptable, and users have to reconnect. `tokens.client_for_user` raises
`CredentialUnreadableError` rather than failing vaguely when that happens.

Free-tier Render sleeps after inactivity, so the first request after idling takes 30 seconds or
so. That is the plan, not a bug.

## 3. Frontend on Vercel

**Add New → Project**, import the repo, then set **Root Directory** to `frontend`. Vercel reads
`frontend/vercel.json` for the rest.

**Edit one line in `frontend/vercel.json`** before deploying: the rewrite destination must point
at your API.

```json
{ "source": "/api/(.*)", "destination": "https://YOUR-API.onrender.com/$1" }
```

That rewrite is not decoration. The session cookie is `SameSite=Lax`, so a cross-origin
`fetch` would not send it and every request would be a 401. Proxying `/api` through Vercel keeps
the browser on one origin, which also removes CORS from the production path. It mirrors exactly
what the Vite dev server does locally.

## 4. Sign-in in production

`APP_ENV=production` disables token sign-in — `/health` reports
`dev_login_available: false` and the form disappears. So production needs a **GitHub OAuth app**:

1. <https://github.com/settings/developers> → **New OAuth App**
2. Homepage URL: your Vercel URL
3. Authorization callback URL: `https://<your-api>.onrender.com/auth/github/callback`
4. Put the client id and secret into Render

Without it the sign-in screen says no method is configured, which is accurate rather than
broken.

## 5. Running the worker against the deployed database

```powershell
cd backend
.\venv\Scripts\Activate.ps1
# APP_ENV stays development, because local_unsafe is only permitted there
python -m app.worker.main
```

It claims jobs from the same Postgres queue, so a run started from the public URL executes on
your machine. The `FOR UPDATE SKIP LOCKED` claim makes that safe even with several workers.

## What a visitor sees without a worker

- Dashboard with past runs, including the two that opened real pull requests
- Full timeline for each, every state and tool call
- The plan, the diff, the risk score, the approval, the pull request link
- An amber banner if they start a run: *"N job(s) have been waiting... No worker appears to be
  running, so runs will stay queued."*

That banner is driven by real queue depth from the database, not a flag someone has to remember
to set.
