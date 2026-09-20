"""Turning a stored GitHub connection into a usable client.

The worker needs this as much as the API does: creating a pull request happens in the worker,
long after the HTTP request that approved it has finished, so the token has to be read from the
database and decrypted there too. Keeping the logic in one place means the API and the worker
cannot drift on how a credential is handled.

The plaintext token exists only as a local variable and is passed straight into the client. It
is never returned to a caller, logged, or written anywhere.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.github.client import GitHubClient
from app.models.core import GitHubConnection
from app.security.crypto import build_cipher

logger = logging.getLogger(__name__)


class ConnectionMissingError(RuntimeError):
    """No GitHub account is linked for this user."""


class CredentialUnreadableError(RuntimeError):
    """The stored token cannot be decrypted, usually a rotated encryption key."""


async def client_for_user(
    session: AsyncSession, *, settings: Settings, user_id: str
) -> GitHubClient:
    connection = (
        await session.execute(
            select(GitHubConnection).where(GitHubConnection.user_id == user_id)
        )
    ).scalar_one_or_none()

    if connection is None:
        raise ConnectionMissingError("GitHub account not connected")

    try:
        token = build_cipher(settings).decrypt(connection.encrypted_token)
    except ValueError as exc:
        # Happens if TOKEN_ENCRYPTION_KEY was rotated without re-linking accounts. Reported
        # as itself rather than as a generic failure, because the fix is specific.
        raise CredentialUnreadableError(
            "Stored GitHub credential is unreadable; reconnect GitHub"
        ) from exc

    return GitHubClient(token, api_base=settings.github_api_base)
