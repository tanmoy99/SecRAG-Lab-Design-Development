"""Core data types shared across the package.

Design rule (decision D12): anything only the *evaluator* may know -- such as
whether a document is poisoned -- is NOT stored on these objects. It lives in
separate evaluator-side records (see `secrag.eval`), so the pipeline under test
cannot read it even by accident.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TrustLevel(IntEnum):
    """Provenance trust levels, ordered so that higher means more trusted."""

    UNVERIFIED = 0  # e.g. member forum posts, public uploads
    STAFF = 1  # e.g. newsletters, event pages written by staff
    OFFICIAL = 2  # e.g. policies and fee schedules from the policy library

    @classmethod
    def parse(cls, value: str | int | TrustLevel) -> TrustLevel:
        """Parse 'official' / 'staff' / 'unverified' (case-insensitive) or an int."""
        if isinstance(value, TrustLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]

    def label(self) -> str:
        return self.name.lower()


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 text. Used for content hashes and cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """A source document as the organisation's knowledge base stores it."""

    doc_id: str
    title: str
    text: str
    source: str  # where it came from, e.g. "policy-library", "member-forum"
    author: str  # author or last editor
    trust_level: TrustLevel
    created_at: str  # ISO date, YYYY-MM-DD
    doc_type: str = "page"  # e.g. "policy", "faq", "news", "forum-post"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Hash of title + body; an integrity defence can compare it to a manifest."""
        return sha256_text(self.title + "\n" + self.text)


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document, carrying the document's metadata."""

    chunk_id: str  # "<doc_id>#<position>"
    doc_id: str
    position: int  # 0-based index of the chunk within its document
    text: str  # chunk body only
    title: str
    source: str
    author: str
    trust_level: TrustLevel
    created_at: str
    doc_type: str
    doc_hash: str  # content hash of the parent document

    @property
    def index_text(self) -> str:
        """Text that is embedded / BM25-indexed: the title gives each chunk context."""
        return f"{self.title}\n\n{self.text}"

    @property
    def chunk_hash(self) -> str:
        return sha256_text(self.text)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retriever, with its rank and score(s)."""

    chunk: Chunk
    rank: int  # 1-based
    score: float  # the score the retriever ranked by (cosine, BM25, RRF, ...)
    scores: dict[str, float] = field(default_factory=dict)  # component scores, for display/analysis
