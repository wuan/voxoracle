"""Typed errors for the shared Mistral cloud client.

These cover transport and HTTP failure modes for the Mistral audio APIs so the
STT (WP4) and TTS (WP5) backends surface uniform, catchable errors to the
session loop.
"""

from __future__ import annotations


class MistralError(Exception):
    """Base class for all Mistral client errors."""


class MistralConnectionError(MistralError):
    """Mistral could not be reached after bounded retries."""


class MistralTimeoutError(MistralError):
    """Mistral did not answer within the configured timeout."""


class MistralAuthError(MistralError):
    """Mistral rejected the credentials (HTTP 401/403)."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Mistral authentication failed (HTTP {status_code}): {body[:200]}")


class MistralRateLimitError(MistralError):
    """Mistral rate-limited the request (HTTP 429)."""

    def __init__(self, body: str) -> None:
        self.body = body
        super().__init__(f"Mistral rate limit exceeded (HTTP 429): {body[:200]}")


class MistralServerError(MistralError):
    """Mistral returned a server error (HTTP 5xx)."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Mistral server error (HTTP {status_code}): {body[:200]}")


class MistralStatusError(MistralError):
    """Mistral returned an unexpected non-success status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Mistral returned HTTP {status_code}: {body[:200]}")


class MistralResponseError(MistralError):
    """Mistral returned a body that does not match the expected shape."""

    def __init__(self, errors: str) -> None:
        self.errors = errors
        super().__init__(f"invalid Mistral response: {errors}")
