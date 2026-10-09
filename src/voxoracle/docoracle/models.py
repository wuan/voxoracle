"""Typed request/response models for the DocOracle HTTP API.

Field names and defaults mirror the models in ``docoracle/server/main.py`` so
the client stays faithful to the server contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["hybrid", "semantic", "bm25"]


class AskRequest(BaseModel):
    """Body for ``POST /ask``; unset optional fields are omitted on the wire."""

    question: str
    module: str | None = None
    component: str | None = None
    version: str | None = None
    k: int | None = None
    retrieval: RetrievalMode | None = None
    show_sources: bool | None = None
    show_context: bool | None = None


class SourceDetail(BaseModel):
    link: str
    url: str | None = None
    module: str
    component: str
    breadcrumb: str | None = None
    section_title: str | None = None
    text_preview: str
    sources: dict[str, int] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    reasoning: str | None = None
    sources: list[str] = Field(default_factory=list)
    source_details: list[SourceDetail] = Field(default_factory=list)
    retrieved_count: int = 0
    retrieval_mode: str = "hybrid"


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    total_chunks: int
    semantic_chunks: int
    bm25_chunks: int
    retrieval_mode: str
    store_path: str
    modules: dict[str, int] = Field(default_factory=dict)
    components: dict[str, int] = Field(default_factory=dict)
    modules_by_component: dict[str, dict[str, int]] = Field(default_factory=dict)
