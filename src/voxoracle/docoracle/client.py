"""Async HTTP client for the DocOracle API.

The client is the only component that knows DocOracle's transport details; the
rest of VoxOracle depends on the typed models in ``voxoracle.docoracle``.
Transient failures (connection, timeout, HTTP 5xx) are retried a bounded number
of times, and every failure is surfaced as a typed error.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from voxoracle.docoracle.models import AskRequest, AskResponse, HealthResponse, InfoResponse

T = TypeVar("T", bound=BaseModel)


class DocOracleError(Exception):
    """Base class for typed DocOracle client errors."""


class DocOracleConnectionError(DocOracleError):
    """DocOracle could not be reached after bounded retries."""


class DocOracleTimeoutError(DocOracleError):
    """DocOracle did not answer within the configured timeout."""


class DocOracleStatusError(DocOracleError):
    """DocOracle returned an error HTTP status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"DocOracle returned HTTP {status_code}: {body[:200]}")


class DocOracleResponseError(DocOracleError):
    """DocOracle returned a body that does not match the expected model."""

    def __init__(self, errors: str) -> None:
        self.errors = errors
        super().__init__(f"invalid DocOracle response: {errors}")


class DocOracleClient:
    """Typed async client for ``/ask``, ``/health`` and ``/info``.

    A custom ``transport`` can be injected (e.g. ``httpx.MockTransport``) to
    run without network access.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        *,
        retries: int = 1,
        retry_delay: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._retries = retries
        self._retry_delay = retry_delay
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ask(self, request: AskRequest) -> AskResponse:
        body = request.model_dump(mode="json", exclude_none=True)
        return await self._request("POST", "/ask", model=AskResponse, json=body)

    async def health(self) -> HealthResponse:
        return await self._request("GET", "/health", model=HealthResponse)

    async def info(self) -> InfoResponse:
        return await self._request("GET", "/info", model=InfoResponse)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        model: type[T],
        json: dict[str, Any] | None = None,
    ) -> T:
        last_error: DocOracleError | None = None
        for attempt in range(self._retries + 1):
            try:
                response = await self._client.request(method, url, json=json)
            except httpx.TimeoutException as exc:
                last_error = DocOracleTimeoutError(f"DocOracle timed out at {url}: {exc}")
            except httpx.ConnectError as exc:
                last_error = DocOracleConnectionError(f"cannot reach DocOracle at {url}: {exc}")
            else:
                if response.status_code >= 500:
                    last_error = DocOracleStatusError(response.status_code, response.text)
                elif response.status_code >= 400:
                    raise DocOracleStatusError(response.status_code, response.text)
                else:
                    try:
                        return model.model_validate(response.json())
                    except (ValidationError, ValueError) as exc:
                        raise DocOracleResponseError(str(exc)) from exc
            if attempt < self._retries and self._retry_delay > 0:
                await asyncio.sleep(self._retry_delay)
        assert last_error is not None
        raise last_error
