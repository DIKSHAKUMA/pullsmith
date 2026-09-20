"""Token encryption and session signing.

GitHub tokens are encrypted at rest with Fernet (AES-CBC + HMAC) rather than stored in
plaintext, because a read-only leak of the database would otherwise hand over write
access to every connected repository.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from cryptography.fernet import Fernet, InvalidToken

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class TokenCipher:
    def __init__(self, key: str | None) -> None:
        if not key or key == "change-me":
            # Development convenience only. validate_for_runtime() blocks this in
            # production so a missing key can never silently disable encryption.
            logger.warning("TOKEN_ENCRYPTION_KEY not set; generating an ephemeral key")
            key = Fernet.generate_key().decode()

        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("Stored token could not be decrypted") from exc


class SessionCodec:
    """Signed, expiring session tokens carried in an HttpOnly cookie.

    A JWT is used as a signed envelope, not as an identity provider. It carries only the
    internal user id, so nothing sensitive is exposed if it is read.
    """

    algorithm = "HS256"

    def __init__(self, secret: str, ttl_seconds: int) -> None:
        self._secret = secret
        self._ttl = ttl_seconds

    def issue(self, user_id: str) -> str:
        now = datetime.now(UTC)
        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=self._ttl)).timestamp()),
        }
        return jwt.encode(payload, self._secret, algorithm=self.algorithm)

    def verify(self, token: str) -> str:
        claims: dict[str, Any] = jwt.decode(token, self._secret, algorithms=[self.algorithm])
        user_id = claims.get("sub")

        if not isinstance(user_id, str) or not user_id:
            raise ValueError("Session token missing subject")

        return user_id


def build_cipher(settings: Settings) -> TokenCipher:
    return TokenCipher(settings.token_encryption_key)


def build_session_codec(settings: Settings) -> SessionCodec:
    return SessionCodec(settings.session_secret, settings.session_ttl_seconds)
