"""Shared async HTTP client for the Mistral cloud audio APIs.

Both the STT backend (``voxoracle.stt.mistral``) and the TTS backend
(``voxoracle.tts.mistral``) reuse this client for authentication, request
timeouts, bounded exponential backoff retries, streaming responses, and a
uniform typed-error surface. It knows nothing about individual endpoints beyond
their paths; callers supply the path and payload.

Retry policy: timeouts, connection errors, HTTP 408, HTTP 429 and HTTP 5xx are
retried up to ``retries`` extra times. Delays grow exponentially
(``retry_delay * backoff_factor**attempt``, capped at ``max_retry_delay``) with
optional jitter, and a numeric ``Retry-After`` header is honored on any retried
response that carries one (in practice 429 and 5xx). Authentication and other
permanent client errors fail immediately.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from voxoracle.mistral.errors import (
    MistralAuthError,
    MistralConnectionError,
    MistralError,
    MistralRateLimitError,
    MistralResponseError,
    MistralServerError,
    MistralStatusError,
    MistralTimeoutError,
)

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"


@dataclass(frozen=True)
class _ResponseFailure:
    """A non-2xx response classified into the shared retry/error policy."""

    error: MistralError
    retryable: bool
    retry_after: float | None = None


class MistralAudioClient:
    """Authenticated async client for Mistral endpoints under ``base_url``.

    A custom ``transport`` can be injected (e.g. ``httpx.MockTransport``) to run
    without network access; no API key is required in that case.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        *,
        retries: int = 2,
        retry_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_retry_delay: float = 30.0,
        jitter: float = 0.1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError(f"retries must be non-negative, got {retries}")
        if retry_delay < 0:
            raise ValueError(f"retry_delay must be non-negative, got {retry_delay}")
        if backoff_factor < 1:
            raise ValueError(f"backoff_factor must be >= 1, got {backoff_factor}")
        if not 0.0 <= jitter <= 1.0:
            raise ValueError(f"jitter must be in [0, 1], got {jitter}")
        self._retries = retries
        self._retry_delay = retry_delay
        self._backoff_factor = backoff_factor
        self._max_retry_delay = max_retry_delay
        self._jitter = jitter
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=False,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def post_multipart(
        self,
        path: str,
        *,
        files: Mapping[str, tuple[str, bytes, str]],
        data: Mapping[str, str],
    ) -> Any:
        """POST a multipart form (file upload) and return the parsed JSON body."""
        return await self._request("POST", path, files=files, data=data)

    async def post_json(self, path: str, *, json: Mapping[str, Any]) -> Any:
        """POST a JSON body and return the parsed JSON response."""
        return await self._request("POST", path, json=dict(json))

    async def stream_post_json(self, path: str, *, json: Mapping[str, Any]) -> AsyncIterator[bytes]:
        """POST a JSON body and stream the raw response body as byte chunks.

        Used for endpoints that return ``text/event-stream`` (e.g. Mistral speech
        synthesis). The request is established with the same bounded retry policy
        as :meth:`post_json`; once the response starts streaming, its body is
        yielded as-is. A mid-stream timeout or transport failure is surfaced as
        the same typed error as the buffered path but is not retried, since a
        partly-consumed streaming response cannot be safely retried.
        """
        async for chunk in self._stream_request(path, json=dict(json)):
            yield chunk

    def _classify(self, response: httpx.Response) -> _ResponseFailure | None:
        """Classify a fully-buffered response into the shared retry/error policy.

        Returns ``None`` for a success (2xx) status.
        """
        if response.status_code < 300:
            return None
        return self._classify_status(response.status_code, response.text, response)

    def _classify_status(
        self, status: int, text: str, response: httpx.Response
    ) -> _ResponseFailure:
        """Classify a non-2xx status into the shared retry/error policy.

        ``text`` is the already-read body; ``response`` provides the headers (for
        ``Retry-After``). Authentication and other client errors (4xx) are never
        retried; 408, 429 and 5xx are.
        """
        if status in (401, 403):
            return _ResponseFailure(MistralAuthError(status, text), retryable=False)
        if status == 429:
            return _ResponseFailure(
                MistralRateLimitError(status, text),
                retryable=True,
                retry_after=self._retry_after_seconds(response),
            )
        if status >= 500:
            return _ResponseFailure(
                MistralServerError(status, text),
                retryable=True,
                retry_after=self._retry_after_seconds(response),
            )
        if status == 408:
            return _ResponseFailure(MistralStatusError(status, text), retryable=True)
        if status >= 400:
            return _ResponseFailure(MistralStatusError(status, text), retryable=False)
        # e.g. a 3xx redirect we do not follow; surface it clearly rather than
        # failing later on a non-JSON body.
        return _ResponseFailure(
            MistralStatusError(status, f"unexpected non-success status (redirect?): {text}"),
            retryable=False,
        )

    def _delay_for(self, attempt: int, retry_after: float | None) -> float:
        """Delay before the next attempt: exponential backoff, capped, with jitter.

        Jitter is applied to the backoff component only, then a numeric
        ``retry_after`` (from a retried 429/5xx response) is honored as a lower
        bound (itself capped at ``max_retry_delay``). The returned delay is
        therefore never shorter than ``min(retry_after, max_retry_delay)``, so a
        server-mandated wait is never undercut by jitter.
        """
        backoff = min(self._retry_delay * self._backoff_factor**attempt, self._max_retry_delay)
        if self._jitter > 0:
            backoff *= 1 - self._jitter * random.random()
        if retry_after is not None:
            backoff = max(backoff, min(retry_after, self._max_retry_delay))
        return backoff

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None  # HTTP-date form is not handled; fall back to backoff

    async def _request(
        self,
        method: str,
        path: str,
        *,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
        data: Mapping[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        last_error: MistralError | None = None
        for attempt in range(self._retries + 1):
            retry_after: float | None = None
            try:
                response = await self._client.request(
                    method,
                    path,
                    files=dict(files) if files is not None else None,
                    data=dict(data) if data is not None else None,
                    json=json,
                )
            except httpx.TimeoutException as exc:
                last_error = MistralTimeoutError(f"Mistral timed out at {path}: {exc}")
            except httpx.TransportError as exc:
                last_error = MistralConnectionError(f"cannot reach Mistral at {path}: {exc}")
            else:
                failure = self._classify(response)
                if failure is None:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise MistralResponseError(str(exc)) from exc
                if not failure.retryable:
                    raise failure.error
                last_error = failure.error
                retry_after = failure.retry_after
            if attempt < self._retries:
                await asyncio.sleep(self._delay_for(attempt, retry_after))
        assert last_error is not None
        raise last_error

    async def _stream_request(self, path: str, *, json: dict[str, Any]) -> AsyncIterator[bytes]:
        """Establish a streamed POST, retrying only the establishment step."""
        last_error: MistralError | None = None
        for attempt in range(self._retries + 1):
            retry_after: float | None = None
            try:
                request = self._client.build_request("POST", path, json=json)
                response = await self._client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                last_error = MistralTimeoutError(f"Mistral timed out at {path}: {exc}")
            except httpx.TransportError as exc:
                last_error = MistralConnectionError(f"cannot reach Mistral at {path}: {exc}")
            else:
                if response.is_success:
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    except httpx.TimeoutException as exc:
                        # A failure after the headers arrived cannot be retried
                        # (the body is already partly consumed); surface it as the
                        # same typed error as the buffered path.
                        raise MistralTimeoutError(f"Mistral timed out at {path}: {exc}") from exc
                    except httpx.DecodingError as exc:
                        # e.g. a corrupt compressed body; a malformed response, not
                        # a connection failure.
                        raise MistralResponseError(
                            f"cannot decode Mistral stream at {path}: {exc}"
                        ) from exc
                    except httpx.RequestError as exc:
                        # Any other request-level failure (transport, connection
                        # reset, ...) is surfaced as the same typed error as the
                        # buffered path, so callers catching MistralError see it.
                        raise MistralConnectionError(
                            f"connection lost mid-stream at {path}: {exc}"
                        ) from exc
                    finally:
                        await response.aclose()
                    return
                try:
                    body = (await response.aread()).decode(errors="replace")
                except httpx.HTTPError as exc:  # pragma: no cover - defensive
                    last_error = MistralConnectionError(
                        f"cannot read Mistral error at {path}: {exc}"
                    )
                else:
                    failure = self._classify_status(response.status_code, body, response)
                    if not failure.retryable:
                        raise failure.error
                    last_error = failure.error
                    retry_after = failure.retry_after
                finally:
                    await response.aclose()
            if attempt < self._retries:
                await asyncio.sleep(self._delay_for(attempt, retry_after))
        assert last_error is not None
        raise last_error
