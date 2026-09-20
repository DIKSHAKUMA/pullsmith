"""Follows a run from the terminal, then shows the diff and offers to approve it.

The dashboard does this over SSE and is nicer to look at. This exists because it works without
the frontend running, and because it prints the whole story in one scrollable place, which is
what you want when something goes wrong.

Usage:
    python -m scripts.e2e_watch <run_id>
    python -m scripts.e2e_watch <run_id> --approve    # approve once it parks at the gate
"""

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.agent.states import RunState, awaits_human, is_terminal
from app.config.settings import get_settings
from app.db.session import init_engine, session_scope
from app.models.core import AgentRun
from app.services import event_service, review_service

POLL_SECONDS = 2.0

_LEVEL_MARK = {"info": " ", "warning": "!", "error": "x"}


def _print_event(event) -> None:  # noqa: ANN001
    mark = _LEVEL_MARK.get(event.level, " ")
    state = (event.state or "")[:22].ljust(22)
    print(f"{mark} {event.sequence:>4}  {state}  {event.message}")


async def follow(run_id: str, *, approve: bool) -> int:
    settings = get_settings()
    init_engine(settings)

    seen = 0
    print(f"following run {run_id}. Ctrl-C to stop.\n")

    while True:
        async with session_scope() as session:
            run = (
                await session.execute(select(AgentRun).where(AgentRun.id == run_id))
            ).scalar_one_or_none()

            if run is None:
                print(f"no such run: {run_id}")
                return 1

            for event in await event_service.list_events(session, run_id, after_sequence=seen):
                _print_event(event)
                seen = event.sequence

            state = RunState(run.state)

            if is_terminal(state):
                print(f"\nfinished: {state}")

                if run.failure_category:
                    print(f"category:  {run.failure_category}")
                    print(f"detail:    {run.failure_detail}")

                await _show_review(session, run_id)
                return 0 if state is RunState.COMPLETED else 1

            if awaits_human(state):
                print(f"\nparked at {state}, waiting for a human")
                await _show_review(session, run_id)

                if not approve:
                    print("\nRe-run with --approve to approve this change, or use the dashboard.")
                    return 0

                if state is not RunState.WAITING_FOR_APPROVAL:
                    print("\n--approve only applies to the final approval gate.")
                    return 0

                try:
                    await review_service.approve(
                        session, run=run, user_id=run.user_id, comment="approved from e2e_watch"
                    )
                    await session.commit()
                except review_service.ReviewError as error:
                    print(f"\ncannot approve: {error}")
                    return 1

                print("\napproved. A create_pull_request job is queued; the worker will take it.")
                print("Expect a GitHub 401 unless a real OAuth token is linked.")
                approve = False

        await asyncio.sleep(POLL_SECONDS)


async def _show_review(session, run_id: str) -> None:  # noqa: ANN001
    plan = await review_service.latest_plan(session, run_id)
    diff = await review_service.latest_diff(session, run_id)

    if plan is not None:
        print("\n--- plan ---")
        print(f"understanding:  {plan.problem_understanding}")
        print(f"suspected cause: {plan.suspected_root_cause}")
        print(f"confidence:      {plan.root_cause_confidence}")
        print(f"verification:    {plan.verification_strategy}")

    if diff is None:
        return

    print("\n--- review ---")
    print(
        f"{diff.files_changed} file(s), +{diff.lines_added}/-{diff.lines_removed}, "
        f"risk {diff.risk_level} (score {diff.risk_score})"
    )

    for reason in diff.risk_reasons or []:
        print(f"  reason:  {reason}")
    for warning in diff.risk_warnings or []:
        print(f"  warning: {warning}")

    if diff.risk_blocking:
        print("  BLOCKING: a possible credential was detected; this cannot be approved")

    print("\n--- diff ---")
    print(diff.diff_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="approve the change when the run parks at the approval gate",
    )
    arguments = parser.parse_args()

    try:
        sys.exit(asyncio.run(follow(arguments.run_id, approve=arguments.approve)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
