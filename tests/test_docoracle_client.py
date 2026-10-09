"""DocOracle client tests using a mocked HTTP transport (no network)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from voxoracle.docoracle.client import (
    DocOracleClient,
    DocOracleConnectionError,
    DocOracleResponseError,
    DocOracleStatusError,
    DocOracleTimeoutError,
)
from voxoracle.docoracle.models import AskRequest, AskResponse, HealthResponse, InfoResponse


def body_of(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content)


def run(coro):
    return asyncio.run(coro)


def ask_via(handler) -> AskResponse:
    async def scenario() -> AskResponse:
        client = DocOracleClient(
            "http://localhost:8000",
            retries=0,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.ask(AskRequest(question="Hallo"))
        finally:
            await client.aclose()

    return run(scenario())


def test_ask_success_omits_unset_optional_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ask"
        assert body_of(request) == {"question": "Hallo"}
        return httpx.Response(
            200,
            json={
                "question": "Hallo",
                "answer": "Antwort",
                "sources": [],
                "source_details": [],
                "retrieved_count": 0,
                "retrieval_mode": "hybrid",
            },
        )

    response = ask_via(handler)
    assert response.answer == "Antwort"


def test_ask_forwards_set_optional_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = body_of(request)
        assert body["module"] == "voxoracle"
        assert body["retrieval"] == "hybrid"
        assert body["k"] == 3
        assert body["show_sources"] is True
        assert "component" not in body
        return httpx.Response(
            200,
            json={
                "question": "Hallo",
                "answer": "Antwort",
                "sources": [],
                "source_details": [],
                "retrieved_count": 0,
                "retrieval_mode": "hybrid",
            },
        )

    async def scenario() -> None:
        client = DocOracleClient(
            "http://localhost:8000",
            retries=0,
            transport=httpx.MockTransport(handler),
        )
        request = AskRequest(
            question="Hallo",
            module="voxoracle",
            retrieval="hybrid",
            k=3,
            show_sources=True,
        )
        try:
            await client.ask(request)
        finally:
            await client.aclose()

    run(scenario())


def test_health_and_info() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=(
                    {
                        "total_chunks": 10,
                        "semantic_chunks": 4,
                        "bm25_chunks": 6,
                        "retrieval_mode": "hybrid",
                        "store_path": "/tmp/store",
                    }
                    if request.url.path == "/info"
                    else {"status": "healthy"}
                ),
            )
        )
        client = DocOracleClient("http://localhost:8000", retries=0, transport=transport)
        try:
            health: HealthResponse = await client.health()
            info: InfoResponse = await client.info()
        finally:
            await client.aclose()
        assert health.status == "healthy"
        assert info.total_chunks == 10
        assert info.semantic_chunks == 4

    run(scenario())


def test_timeout_raises_and_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(DocOracleTimeoutError):
        ask_via(handler)
    # retries=0 in ask_via, so a single attempt by default
    assert calls == 1


def test_timeout_with_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    async def scenario() -> None:
        client = DocOracleClient(
            "http://localhost:8000",
            retries=2,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(DocOracleTimeoutError):
                await client.ask(AskRequest(question="Hallo"))
        finally:
            await client.aclose()

    run(scenario())
    assert calls == 3


def test_connection_error_with_retry_then_success() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(
            200,
            json={
                "question": "Hallo",
                "answer": "Antwort",
                "sources": [],
                "source_details": [],
                "retrieved_count": 0,
                "retrieval_mode": "hybrid",
            },
        )

    async def scenario() -> None:
        client = DocOracleClient(
            "http://localhost:8000",
            retries=2,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await client.ask(AskRequest(question="Hallo"))
        finally:
            await client.aclose()
        assert response.answer == "Antwort"

    run(scenario())
    assert calls == 3


def test_connection_error_exhausted() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    async def scenario() -> None:
        client = DocOracleClient(
            "http://localhost:8000",
            retries=2,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(DocOracleConnectionError):
                await client.ask(AskRequest(question="Hallo"))
        finally:
            await client.aclose()

    run(scenario())
    assert calls == 3


def test_http_5xx_raises_status_error_after_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="boom")

    with pytest.raises(DocOracleStatusError) as excinfo:
        ask_via(handler)
    assert excinfo.value.status_code == 500
    assert calls == 1  # ask_via retries=0


def test_other_transport_errors_are_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    with pytest.raises(DocOracleConnectionError):
        ask_via(handler)


def test_remote_protocol_error_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected", request=request)

    with pytest.raises(DocOracleConnectionError):
        ask_via(handler)


def test_http_4xx_raises_immediately() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    async def scenario() -> None:
        client = DocOracleClient(
            "http://localhost:8000",
            retries=2,
            retry_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(DocOracleStatusError) as excinfo:
                await client.ask(AskRequest(question="Hallo"))
        finally:
            await client.aclose()
        assert excinfo.value.status_code == 404

    run(scenario())


def test_invalid_json_body_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with pytest.raises(DocOracleResponseError):
        ask_via(handler)


def test_schema_mismatch_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": 12345})

    with pytest.raises(DocOracleResponseError):
        ask_via(handler)
