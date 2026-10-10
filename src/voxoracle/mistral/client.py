"""Shared async HTTP client for the Mistral cloud audio APIs.

Both the STT backend (``voxoracle.stt.mistral``) and, later, the TTS backend
reuse this client for authentication, request timeouts, bounded exponential
backoff retries, and a uniform typed-error surface. It knows nothing about
individual endpoints beyond their paths; callers supply the path and payload.

Retry policy: timeouts, connection errors, HTTP 408, HTTP 429 and HTTP 5xx are
retried up to ``retries`` extra times. Delays grow exponentially
(``retry_delay * backoff_factor**attempt``, capped at ``max_retry_delay``) with
optional jitter, and a numeric ``Retry-After`` header on 429/503 is honored when
it exceeds the computed delay. Authentication and other client errors (4xx) fail
immediately.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
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

    def _delay_for(self, attempt: int, retry_after: float | None) -> float:
        """Delay before the next attempt: exponential backoff, capped, with jitter.

        Jitter is applied to the backoff component only, then a numeric
        ``retry_after`` (from a 429/503 header) is honored as a lower bound (itself
        capped at ``max_retry_delay``). The returned delay is therefore never
        shorter than ``min(retry_after, max_retry_delay)``, so a server-mandated
        wait is never undercut by jitter.
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
                status = response.status_code
                if status in (401, 403):
                    raise MistralAuthError(status, response.text)
                if status == 429:
                    last_error = MistralRateLimitError(response.text)
                    retry_after = self._retry_after_seconds(response)
                elif status >= 500:
                    last_error = MistralServerError(status, response.text)
                    retry_after = self._retry_after_seconds(response)
                elif status == 408:
                    last_error = MistralStatusError(status, response.text)
                elif status >= 400:
                    raise MistralStatusError(status, response.text)
                elif status >= 300:
                    # e.g. a 3xx redirect we do not follow; surface it clearly
                    # rather than failing later on a non-JSON body.
                    raise MistralStatusError(
                        status, f"unexpected non-success status (redirect?): {response.text}"
                    )
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise MistralResponseError(str(exc)) from exc
            if attempt < self._retries:
                await asyncio.sleep(self._delay_for(attempt, retry_after))
        assert last_error is not None
        raise last_error
