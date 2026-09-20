"""API tests: auth enforcement, run creation, ownership isolation, event history."""

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import new_id
from app.models.core import Job, User
from app.services import job_queue


async def test_health_is_public(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_runs_require_authentication(client: AsyncClient) -> None:
    response = await client.get("/runs")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "HTTP_401"


async def test_invalid_session_cookie_is_rejected(client: AsyncClient, settings) -> None:  # noqa: ANN001
    client.cookies.set(settings.session_cookie_name, "not-a-real-token")
    response = await client.get("/runs")
    assert response.status_code == 401


async def test_create_run_returns_202_and_enqueues_job(
    client: AsyncClient, db: AsyncSession, seeded: dict[str, str], auth_cookie
) -> None:  # noqa: ANN001
    for name, value in auth_cookie(seeded["user_id"]).items():
        client.cookies.set(name, value)

    response = await client.post(
        "/runs",
        json={"repository_id": seeded["repository_id"], "issue_id": seeded["issue_id"]},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "CREATED"

    jobs = (await db.execute(select(Job).where(Job.run_id == body["run_id"]))).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].kind == job_queue.JobKind.EXECUTE_RUN
    assert jobs[0].status == job_queue.JobStatus.QUEUED


async def test_duplicate_active_run_is_conflict(
    client: AsyncClient, seeded: dict[str, str], auth_cookie
) -> None:  # noqa: ANN001
    for name, value in auth_cookie(seeded["user_id"]).items():
        client.cookies.set(name, value)

    payload = {"repository_id": seeded["repository_id"], "issue_id": seeded["issue_id"]}

    first = await client.post("/runs", json=payload)
    second = await client.post("/runs", json=payload)

    assert first.status_code == 202
    assert second.status_code == 409


async def test_cannot_start_run_on_another_users_repository(
    client: AsyncClient, db: AsyncSession, seeded: dict[str, str], auth_cookie
) -> None:  # noqa: ANN001
    intruder = User(id=new_id(), github_user_id=777, github_login="intruder")
    db.add(intruder)
    await db.commit()

    for name, value in auth_cookie(intruder.id).items():
        client.cookies.set(name, value)

    response = await client.post(
        "/runs",
        json={"repository_id": seeded["repository_id"], "issue_id": seeded["issue_id"]},
    )

    # 404, not 403: the endpoint must not confirm that someone else's repository exists.
    assert response.status_code == 404


async def test_event_history_supports_after_cursor(
    client: AsyncClient, seeded: dict[str, str], auth_cookie
) -> None:  # noqa: ANN001
    for name, value in auth_cookie(seeded["user_id"]).items():
        client.cookies.set(name, value)

    created = await client.post(
        "/runs",
        json={"repository_id": seeded["repository_id"], "issue_id": seeded["issue_id"]},
    )
    run_id = created.json()["run_id"]

    all_events = await client.get(f"/runs/{run_id}/events/history")
    assert all_events.status_code == 200
    assert len(all_events.json()) >= 1

    after_last = await client.get(f"/runs/{run_id}/events/history?after=99")
    assert after_last.json() == []


async def test_cancel_then_cancel_again_conflicts(
    client: AsyncClient, seeded: dict[str, str], auth_cookie
) -> None:  # noqa: ANN001
    for name, value in auth_cookie(seeded["user_id"]).items():
        client.cookies.set(name, value)

    created = await client.post(
        "/runs",
        json={"repository_id": seeded["repository_id"], "issue_id": seeded["issue_id"]},
    )
    run_id = created.json()["run_id"]

    first = await client.post(f"/runs/{run_id}/cancel")
    second = await client.post(f"/runs/{run_id}/cancel")

    assert first.status_code == 200
    assert first.json()["state"] == "CANCELLED"
    assert second.status_code == 409


async def test_validation_error_uses_standard_error_shape(
    client: AsyncClient, seeded: dict[str, str], auth_cookie
) -> None:  # noqa: ANN001
    for name, value in auth_cookie(seeded["user_id"]).items():
        client.cookies.set(name, value)

    response = await client.post("/runs", json={"repository_id": seeded["repository_id"]})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
