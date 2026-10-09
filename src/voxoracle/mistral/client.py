"""Shared async HTTP client for the Mistral cloud audio APIs.

Both the STT backend (``voxoracle.stt.mistral``) and, later, the TTS backend
reuse this client for authentication, request timeouts, bounded fixed-delay
retries, and a uniform typed-error surface. It knows nothing about individual
endpoints beyond their paths; callers supply the path and payload.

Retry policy: timeouts, connection errors, HTTP 429 and HTTP 5xx are retried up
to ``retries`` extra times with a fixed ``retry_delay``. Authentication and other
client errors (4xx) fail immediately.
"""

from __future__ import annotations

import asyncio
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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError(f"retries must be non-negative, got {retries}")
        self._retries = retries
        self._retry_delay = retry_delay
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"},
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
                if response.status_code in (401, 403):
                    raise MistralAuthError(response.status_code, response.text)
                if response.status_code == 429:
                    last_error = MistralRateLimitError(response.text)
                elif response.status_code >= 500:
                    last_error = MistralServerError(response.status_code, response.text)
                elif response.status_code >= 400:
                    raise MistralStatusError(response.status_code, response.text)
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise MistralResponseError(str(exc)) from exc
            if attempt < self._retries and self._retry_delay > 0:
                await asyncio.sleep(self._retry_delay)
        assert last_error is not None
        raise last_error
